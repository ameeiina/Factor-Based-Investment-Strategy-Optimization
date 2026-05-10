# gp_optimize.py — Bayesian optimisation of 11-factor {-1, 0, +1} QRUMBLE strategies
#
# Runs gp_minimize (scikit-optimize) over 3 configurations × 3 seeds on 2015-2019
# training data. The scalar objective J(w) combines Sharpe, annualized return, and VaR
# with variance-normalised weights. A shared eval cache avoids redundant QRUMBLE calls.
# Selects the best configuration by cross-seed stability, validates the top 10 solutions
# on 2019-2020, and writes the recommended strategy to gp_final_recommendation.pkl.
#
# Output : outputs/gp/
# DO NOT use python3 — QRUMBLE is only available in the conda env named Q

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import pickle
import logging

os.environ["MKL_VERBOSE"] = "0"

from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

import qrumble as q
from qrumble.factors import *
import qrumble.universe
import qrumble.optimizer

q.optimizer.logging(logging.INFO)
os.makedirs("outputs/gp", exist_ok=True)

# ── Setup ─────────────────────────────────────────────────────────────────────

q.universe.load("../../data/UNIVERSE.2015_2019.pickle", index="stoxx600")
funda_train = q.load("../../data/FUNDAMENTALS.2015_2019.pickle")
ohlcv_train = q.load("../../data/stoxx600.jan2014_dec2019.pickle")

funda_test  = q.load("../../data/FUNDAMENTALS.2019_2020.pickle")
ohlcv_test  = q.load("../../data/stoxx600.jan2018_apr2021.pickle")

RF = -0.6

factor_universe = [
    Yield(), ROC(), EarningsYield(), RS('6m'),
    ROA(), ROA_diff(), AccrualRatio(),
    LTDebt_to_Assets_diff(), CurrentRatio_diff(),
    OpMgn_diff(), AssetTurnover_diff()
]

factor_names = [
    "Yield", "ROC", "EarningsYield", "RS_6m",
    "ROA", "ROA_diff", "AccrualRatio",
    "LTDebt_Assets_diff", "CurrentRatio_diff",
    "OpMgn_diff", "AssetTurnover_diff"
]

# ── Objective weights ─────────────────────────────────────────────────────────
# Derived from training data so each term contributes equally to variance of J(w):
#   alpha / std(S) = beta / std(A) = gamma / std(V)
# Computed from gp_scaled_score.csv (400-call training run):
#   std(S) = 0.232, std(A) = 6.707, std(V) = 2.518
ALPHA = 1.000   # Sharpe
BETA  = 0.035   # Annualized return  (= 1/std(A) / (1/std(S)))
GAMMA = 0.092   # Variance           (= 1/std(V) / (1/std(S)))

print(f"Objective: J(w) = -{ALPHA}*S - {BETA}*A + {GAMMA}*V")
print(f"Each term contributes equally to variance of J(w)")

# ── Factor selection ──────────────────────────────────────────────────────────

def get_selection(f, coefficients):
    ranking_criteria = []
    for i, factor in enumerate(factor_universe):
        if coefficients[i] == 1:
            ranking_criteria.append(factor(order=ascending))
        elif coefficients[i] == -1:
            ranking_criteria.append(factor(order=descending))
    if ranking_criteria:
        f.Ranking(sum(ranking_criteria), top=30)
    return f

# ── Evaluation cache ──────────────────────────────────────────────────────────
# Shared across runs to avoid re-evaluating identical weight vectors.

eval_cache = {}

def run_qrumble(weights, funda, ohlcv, start_date="jan2015", period="5y"):
    key = tuple(weights)
    if key in eval_cache:
        return eval_cache[key]
    try:
        perf, _ = q.qrumble(
            "gp:{DATE}/rank/30", "gp:{DATE}/play/30/1y",
            start_date, period, rebalance="1y",
            fundamentals=funda, ohlcv=ohlcv, Rf=RF,
            universe_list=q.universe.fetch,
            universe=lambda f: f.Screening(~Sector().isin([
                'banking', 'insurance', 'real estate',
                'financial', 'utilities', 'waste'
            ])),
            criteria=lambda f: get_selection(f, weights),
            dataframe=True
        )
        result = (perf['sharpe'], perf['annualized'], perf['var'])
        eval_cache[key] = result
        return result
    except Exception as e:
        print(f"  Failed for {weights}: {e}")
        return None

# ── Single config runner ──────────────────────────────────────────────────────

def run_gp_config(n_calls, n_initial_points, acq_func,
                  base_estimator, seed, log_csv=None):

    print(f"\n{'='*60}")
    print(f"  n_calls={n_calls}, n_init={n_initial_points}, "
          f"acq={acq_func}, base={base_estimator}, seed={seed}")
    print(f"{'='*60}")

    call_log = []

    if log_csv and not os.path.isfile(log_csv):
        with open(log_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [f"w{i}" for i in range(11)] +
                ["sharpe", "annualized", "var", "J", "call_number"]
            )

    call_counter = [0]

    space = [Categorical([-1, 0, 1], name=f"w{i}") for i in range(11)]

    @use_named_args(space)
    def objective(**params):
        weights = [params[f"w{i}"] for i in range(11)]
        call_counter[0] += 1

        if all(w == 0 for w in weights):
            return 1e6

        result = run_qrumble(weights, funda_train, ohlcv_train)
        if result is None:
            return 1e6

        sharpe, annual, var = result
        J = -ALPHA * sharpe - BETA * annual + GAMMA * var

        call_log.append({
            "call":   call_counter[0],
            "sharpe": sharpe,
            "ann":    annual,
            "var":    var,
            "J":      J
        })

        print(f"  [{call_counter[0]:3d}] Sharpe={sharpe:.3f}, "
              f"Ann={annual:.3f}, Var={var:.3f}, J={J:.4f}")

        if log_csv:
            with open(log_csv, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    weights + [sharpe, annual, var, J, call_counter[0]]
                )
        return J

    gp_result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        acq_func=acq_func,
        base_estimator=base_estimator,
        random_state=seed
    )

    best_weights = list(gp_result.x)
    best_result  = run_qrumble(best_weights, funda_train, ohlcv_train)
    best_sharpe, best_ann, best_var = best_result if best_result else (None, None, None)

    df_calls          = pd.DataFrame(call_log)
    running_best_J    = df_calls["J"].cummin().tolist()
    running_best_sharpe = []
    best_s = -np.inf
    for s in df_calls["sharpe"]:
        best_s = max(best_s, s)
        running_best_sharpe.append(best_s)

    # First call where improvement stalls for 20+ consecutive steps.
    plateau_call = None
    for i in range(1, len(running_best_J)):
        if abs(running_best_J[i] - running_best_J[i-1]) < 0.001:
            if plateau_call is None:
                stall_start = i
            if i - stall_start >= 20:
                plateau_call = stall_start
                break
        else:
            plateau_call = None
            stall_start  = i

    n_unique = len(eval_cache)

    print(f"\n  Best Sharpe     : {best_sharpe:.3f}")
    print(f"  Best J(w)       : {gp_result.fun:.4f}")
    print(f"  Unique cache    : {n_unique}")
    print(f"  Plateau at call : {plateau_call}")

    return {
        "n_calls":            n_calls,
        "n_initial_points":   n_initial_points,
        "acq_func":           acq_func,
        "base_estimator":     base_estimator,
        "seed":               seed,
        "best_sharpe":        best_sharpe,
        "best_ann":           best_ann,
        "best_var":           best_var,
        "best_J":             gp_result.fun,
        "n_unique_evals":     n_unique,
        "complexity":         int(np.count_nonzero(best_weights)),
        "weights":            tuple(best_weights),
        "running_best_J":     running_best_J,
        "running_best_sharpe":running_best_sharpe,
        "plateau_call":       plateau_call,
        "gp_result":          gp_result
    }

# ── Hyperparameter grid ───────────────────────────────────────────────────────
# 9 runs total (3 configs x 3 seeds).
# Config 1: baseline (EI, GP surrogate)
# Config 2: acquisition function (LCB vs EI)
# Config 3: warm-up budget (100 vs 50 initial points)
# n_calls=200 justified by convergence analysis (plateau ~150 calls).

configs = [
    # (n_calls, n_initial_points, acq_func, base_estimator)
    (200,  50, "EI",  "GP"),    # baseline
    (200,  50, "LCB", "GP"),    # test acquisition function
    (200, 100, "EI",  "GP"),    # test longer random warm-up
]

seeds = [42, 123, 999]

print(f"\nTotal runs   : {len(configs) * len(seeds)}")
print(f"Max qrumble  : {len(configs) * len(seeds) * 200} "
      f"(reduced by shared cache)")

run_results = []
for (n_calls, n_init, acq, base), seed in itertools.product(configs, seeds):
    label = f"nc{n_calls}_ni{n_init}_{acq}_{base}_s{seed}"
    r = run_gp_config(
        n_calls=n_calls,
        n_initial_points=n_init,
        acq_func=acq,
        base_estimator=base,
        seed=seed,
        log_csv=f"outputs/gp/gp_{label}.csv"
    )
    run_results.append(r)

# ── Stability analysis ────────────────────────────────────────────────────────

df_runs = pd.DataFrame([
    {k: v for k, v in r.items()
     if k not in ("running_best_J", "running_best_sharpe", "gp_result")}
    for r in run_results
])

stability = df_runs.groupby(
    ["n_calls", "n_initial_points", "acq_func", "base_estimator"]
).agg(
    mean_sharpe  = ("best_sharpe",    "mean"),
    std_sharpe   = ("best_sharpe",    "std"),
    max_sharpe   = ("best_sharpe",    "max"),
    mean_ann     = ("best_ann",       "mean"),
    mean_evals   = ("n_unique_evals", "mean"),
    mean_complex = ("complexity",     "mean"),
    mean_plateau = ("plateau_call",   "mean")
).reset_index()

stability["score"] = (
    stability["mean_sharpe"]
    - 0.5  * stability["std_sharpe"]
    - 0.05 * stability["mean_complex"]
)

stability = stability.sort_values("score", ascending=False).reset_index(drop=True)

print("\nGP Stability Analysis:")
print(stability.to_string(index=False))
stability.to_csv("outputs/gp/gp_stability_analysis.csv", index=False)

# ── Plot 1 — convergence by running best Sharpe ───────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
axes = axes.flatten()

for ax_idx, (n_calls, n_init, acq, base) in enumerate(configs):
    ax = axes[ax_idx]
    for r in run_results:
        if (r["n_calls"] == n_calls and
            r["n_initial_points"] == n_init and
            r["acq_func"] == acq and
            r["base_estimator"] == base):
            ax.plot(r["running_best_sharpe"],
                    label=f"seed={r['seed']}", alpha=0.8)
    ax.set_title(f"n_init={n_init}, acq={acq}, base={base}")
    ax.set_xlabel("Evaluation call")
    ax.set_ylabel("Best Sharpe so far")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("GP Convergence — Running Best Sharpe",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/gp/gp_convergence_sharpe.png", dpi=150)
plt.close()
print("Saved: outputs/gp/gp_convergence_sharpe.png")

# ── Plot 2 — convergence by running best J(w) ────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
axes = axes.flatten()

for ax_idx, (n_calls, n_init, acq, base) in enumerate(configs):
    ax = axes[ax_idx]
    for r in run_results:
        if (r["n_calls"] == n_calls and
            r["n_initial_points"] == n_init and
            r["acq_func"] == acq and
            r["base_estimator"] == base):
            ax.plot(r["running_best_J"],
                    label=f"seed={r['seed']}", alpha=0.8)
    ax.set_title(f"n_init={n_init}, acq={acq}, base={base}")
    ax.set_xlabel("Evaluation call")
    ax.set_ylabel("Best J(w) so far")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("GP Convergence — Running Best J(w)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/gp/gp_convergence_J.png", dpi=150)
plt.close()
print("Saved: outputs/gp/gp_convergence_J.png")

# ── Best config selection ─────────────────────────────────────────────────────

best_cfg       = stability.iloc[0]
best_n_calls   = int(best_cfg["n_calls"])
best_n_init    = int(best_cfg["n_initial_points"])
best_acq       = best_cfg["acq_func"]
best_base      = best_cfg["base_estimator"]

print(f"\nBest config: n_calls={best_n_calls}, n_init={best_n_init}, "
      f"acq={best_acq}, base={best_base}")

best_logs = []
for seed in seeds:
    label = (f"nc{best_n_calls}_ni{best_n_init}_"
             f"{best_acq}_{best_base}_s{seed}")
    fname = f"outputs/gp/gp_{label}.csv"
    if os.path.isfile(fname):
        best_logs.append(pd.read_csv(fname))

weight_cols = [f"w{i}" for i in range(11)]
df_best = pd.concat(best_logs, ignore_index=True).drop_duplicates(
    subset=weight_cols
)
df_best = df_best[df_best["sharpe"] < 1e5]
df_best.to_csv("outputs/gp/gp_all_train_evaluations.csv", index=False)
print(f"  Unique training evaluations: {len(df_best)}")

top10 = df_best.nlargest(10, "sharpe").reset_index(drop=True)

# ── Validate top 10 on 2019-2020 ─────────────────────────────────────────────

print(f"\nValidating top 10 solutions on 2019-2020...")
val_records = []

for i, row in top10.iterrows():
    weights = row[weight_cols].values.astype(int).tolist()
    print(f"  [{i+1}/10] {weights}")
    try:
        perf, _ = q.qrumble(
            "gp:{DATE}/rank/30", "gp:{DATE}/play/30/1y",
            "jan2019", "1y", rebalance="1y",
            fundamentals=funda_test, ohlcv=ohlcv_test, Rf=RF,
            universe_list=q.universe.fetch,
            universe=lambda f: f.Screening(~Sector().isin([
                'banking', 'insurance', 'real estate',
                'financial', 'utilities', 'waste'
            ])),
            criteria=lambda f: get_selection(f, weights),
            dataframe=True
        )
        val_sharpe = perf['sharpe']
        val_ann    = perf['annualized']
        val_var    = perf['var']
    except Exception as e:
        print(f"  Validation failed: {e}")
        val_sharpe = val_ann = val_var = None

    val_records.append({
        **{f"w{j}": int(weights[j]) for j in range(11)},
        "train_sharpe": row["sharpe"],
        "train_ann":    row["annualized"],
        "train_var":    row["var"],
        "val_sharpe":   val_sharpe,
        "val_ann":      val_ann,
        "val_var":      val_var,
        "degradation":  round(row["sharpe"] - val_sharpe, 4)
                        if val_sharpe else None,
        "complexity":   int(np.count_nonzero(weights))
    })

df_val = pd.DataFrame(val_records)
df_val["robustness_score"] = (
    df_val["val_sharpe"]
    - 0.3  * df_val["degradation"]
    - 0.05 * df_val["complexity"]
)
df_val = df_val.sort_values(
    "robustness_score", ascending=False
).reset_index(drop=True)
df_val.to_csv("outputs/gp/gp_validation_results.csv", index=False)

print("\nGP Validation Results (sorted by robustness):")
print(df_val[["train_sharpe", "train_ann", "val_sharpe",
              "val_ann", "degradation", "complexity",
              "robustness_score"]].to_string(index=False))

# ── Train vs val scatter ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    df_val["train_sharpe"], df_val["val_sharpe"],
    c=df_val["complexity"], cmap="RdYlGn_r",
    s=100, edgecolors="black", linewidths=0.5
)
plt.colorbar(sc, ax=ax, label="# active factors")

best_row = df_val.iloc[0]
ax.annotate("Best robust",
            xy=(best_row["train_sharpe"], best_row["val_sharpe"]),
            xytext=(10, -15), textcoords="offset points",
            fontsize=9, color="darkgreen",
            arrowprops=dict(arrowstyle="->", color="darkgreen"))

all_vals = pd.concat([df_val["train_sharpe"], df_val["val_sharpe"]])
lims = [all_vals.min() - 0.05, all_vals.max() + 0.05]
ax.plot(lims, lims, "k--", alpha=0.4, label="No degradation")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Train Sharpe (2015-2019)")
ax.set_ylabel("Val Sharpe (2019-2020)")
ax.set_title("GP: Train vs Validation Sharpe\n(colour = # active factors)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/gp/gp_train_vs_val_scatter.png", dpi=150)
plt.close()
print("Saved: outputs/gp/gp_train_vs_val_scatter.png")

# ── Final recommendation ──────────────────────────────────────────────────────

best         = df_val.iloc[0]
best_weights = [int(best[f"w{j}"]) for j in range(11)]

print("\n" + "="*60)
print("GP RECOMMENDED STRATEGY")
print("="*60)
for j, (name, w) in enumerate(zip(factor_names, best_weights)):
    if w != 0:
        direction = "ascending  ↑" if w == 1 else "descending ↓"
        print(f"  w{j:<3} {name:<25} {direction}")
print(f"\n  Train Sharpe    : {best['train_sharpe']:.3f}")
print(f"  Val   Sharpe    : {best['val_sharpe']:.3f}")
print(f"  Degradation     : {best['degradation']:.3f}")
print(f"  Complexity      : {int(best['complexity'])} active factors")
print(f"  Config          : n_calls={best_n_calls}, "
      f"n_init={best_n_init}, acq={best_acq}, base={best_base}")
print("="*60)

with open("outputs/gp/gp_final_recommendation.pkl", "wb") as f:
    pickle.dump({
        "weights":      best_weights,
        "train_stats":  {"sharpe":     best["train_sharpe"],
                         "annualized": best["train_ann"],
                         "var":        best["train_var"]},
        "val_stats":    {"sharpe":     best["val_sharpe"],
                         "annualized": best["val_ann"],
                         "var":        best["val_var"]},
        "robustness_score": best["robustness_score"],
        "best_config":  {"n_calls":          best_n_calls,
                         "n_initial_points": best_n_init,
                         "acq_func":         best_acq,
                         "base_estimator":   best_base},
        "objective_weights": {"alpha": ALPHA,
                              "beta":  BETA,
                              "gamma": GAMMA}
    }, f)

print("\nAll outputs saved to outputs/gp/")
print("   - outputs/gp/gp_stability_analysis.csv")
print("   - outputs/gp/gp_convergence_sharpe.png")
print("   - outputs/gp/gp_convergence_J.png")
print("   - outputs/gp/gp_all_train_evaluations.csv")
print("   - outputs/gp/gp_validation_results.csv")
print("   - outputs/gp/gp_train_vs_val_scatter.png")
print("   - outputs/gp/gp_final_recommendation.pkl")
print("\nNext step: include gp_final_recommendation.pkl in Section 7 comparison")
