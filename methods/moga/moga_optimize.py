# moga_optimize.py — NSGA-II multi-objective optimisation of 11-factor QRUMBLE strategies
#
# Runs NSGA-II (pymoo) over a 2×2 population/generation grid × 3 seeds = 12 runs on
# 2015-2019 training data. Three objectives: maximise Sharpe, maximise annualised return,
# minimise VaR. A taboo set prevents re-evaluating identical weight vectors. Early stopping
# based on Pareto-front hypervolume. Selects the best configuration by cross-seed stability,
# builds Pareto fronts from best and all configurations, and saves them for moga_validate.py.
#
# Output : outputs/
# DO NOT use python3 — QRUMBLE is only available in the conda env named Q

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import pickle

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.hv import HV

import qrumble as q
from qrumble.factors import *

# ── Setup ─────────────────────────────────────────────────────────────────────

os.makedirs("outputs", exist_ok=True)

q.universe.load("../../data/UNIVERSE.2015_2019.pickle", index="stoxx600")
funda_train = q.load("../../data/FUNDAMENTALS.2015_2019.pickle")
ohlcv_train = q.load("../../data/stoxx600.jan2014_dec2019.pickle")

RF = -0.6

factor_universe = [
    Yield(), ROC(), EarningsYield(), RS('6m'),
    ROA(), ROA_diff(), AccrualRatio(),
    LTDebt_to_Assets_diff(), CurrentRatio_diff(),
    OpMgn_diff(), AssetTurnover_diff()
]

taboo_set = set()

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

# ── Problem definition ────────────────────────────────────────────────────────

class QrumbleFlatProblem(ElementwiseProblem):
    def __init__(self, start_date="jan2015", period="5y",
                 funda=None, ohlcv=None, log_csv=None):
        super().__init__(
            n_var=11, n_obj=3, n_constr=0,
            xl=np.array([-1]*11),
            xu=np.array([1]*11),
            type_var=int
        )
        self.start_date = start_date
        self.period     = period
        self.funda      = funda
        self.ohlcv      = ohlcv
        self.log_csv    = log_csv

        if log_csv and not os.path.isfile(log_csv):
            with open(log_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = [f"w{i}" for i in range(11)] + ["sharpe", "annualized", "var"]
                writer.writerow(header)

    def _evaluate(self, x, out, *args, **kwargs):
        x   = np.clip(np.round(x), -1, 1).astype(int)
        key = tuple(x)

        if key in taboo_set:
            out["F"] = [1e6, 1e6, 1e6]
            return
        taboo_set.add(key)

        try:
            perf, df = q.qrumble(
                "moga:{DATE}/rank/30", "moga:{DATE}/play/30/1y",
                self.start_date, self.period, rebalance="1y",
                fundamentals=self.funda, ohlcv=self.ohlcv, Rf=RF,
                universe_list=q.universe.fetch,
                universe=lambda f: f.Screening(~Sector().isin([
                    'banking', 'insurance', 'real estate',
                    'financial', 'utilities', 'waste'
                ])),
                criteria=lambda f: get_selection(f, x),
                dataframe=True
            )
            sharpe = perf['sharpe']
            ann    = perf['annualized']
            var    = perf['var']
            print(f"  Sharpe: {sharpe:.3f}, Ann: {ann:.3f}, Var: {var:.3f} | {x}")
            out["F"] = [-sharpe, -ann, var]

            if self.log_csv:
                with open(self.log_csv, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(list(x) + [sharpe, ann, var])

        except Exception as e:
            print(f"  Failed for {x}: {e}")
            out["F"] = [1e6, 1e6, 1e6]

# ── Early stop callback ───────────────────────────────────────────────────────

class EarlyStopCallback(Callback):
    def __init__(self, patience=5, min_delta=0.01):
        super().__init__()
        self.patience       = patience
        self.min_delta      = min_delta
        self.best_hv        = -np.inf
        self.stall          = 0
        self.history        = []
        self.sharpe_history = []

    def notify(self, algorithm):
        F = algorithm.opt.get("F")

        # ref_point must dominate all solutions:
        # F[:,0]=-sharpe, F[:,1]=-annualized (both negative, 0 is safe upper bound)
        # F[:,2]=var (~-20 to -32, so 50 is safe upper bound)
        ref_point  = np.array([0.0, 0.0, 50.0])
        hv         = HV(ref_point=ref_point)
        current_hv = hv(F)
        self.history.append(current_hv)

        self.sharpe_history.append(-min(F[:, 0]))

        if current_hv - self.best_hv > self.min_delta:
            self.best_hv = current_hv
            self.stall   = 0
        else:
            self.stall  += 1

        if self.stall >= self.patience:
            print(f"  Early stop at gen {algorithm.n_gen} "
                  f"(HV no improvement for {self.patience} gens, "
                  f"best HV={self.best_hv:.4f})")
            algorithm.termination.force_termination = True

# ── Pareto filter ─────────────────────────────────────────────────────────────

def get_pareto_front(df):
    """Return only non-dominated solutions from a dataframe."""
    df = df.copy().reset_index(drop=True)
    pareto_mask = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            if (other['sharpe']     >= row['sharpe'] and
                other['annualized'] >= row['annualized'] and
                other['var']        <= row['var'] and
                (other['sharpe']     > row['sharpe'] or
                 other['annualized'] > row['annualized'] or
                 other['var']        < row['var'])):
                dominated = True
                break
        if not dominated:
            pareto_mask.append(i)
    return df.loc[pareto_mask].copy()

# ── Single config runner ──────────────────────────────────────────────────────

def run_config(pop_size, n_gen, seed, log_csv=None):
    print(f"\n{'='*55}")
    print(f"  Running: pop={pop_size}, gen={n_gen}, seed={seed}")
    print(f"{'='*55}")

    taboo_set.clear()

    problem  = QrumbleFlatProblem(
        start_date="jan2015", period="5y",
        funda=funda_train, ohlcv=ohlcv_train,
        log_csv=log_csv
    )
    callback = EarlyStopCallback(patience=3, min_delta=0.01)

    result = minimize(
        problem,
        NSGA2(pop_size=pop_size),
        get_termination("n_gen", n_gen),
        seed=seed,
        callback=callback,
        save_history=True,
        verbose=False
    )

    best_idx = result.F[:, 0].argmin()

    return {
        "pop_size":       pop_size,
        "n_gen":          n_gen,
        "seed":           seed,
        "actual_gens":    len(callback.history),
        "best_sharpe":   -result.F[best_idx, 0],
        "best_ann":      -result.F[best_idx, 1],
        "best_var":       result.F[best_idx, 2],
        "n_evals":        len(taboo_set),
        "complexity":     int(np.count_nonzero(result.X[best_idx])),
        "weights":        tuple(result.X[best_idx].astype(int)),
        "hv_history":     callback.history,
        "sharpe_history": callback.sharpe_history,
        "result_obj":     result
    }

# ── Hyperparameter grid search ────────────────────────────────────────────────

pop_sizes = [30, 50]
n_gens    = [8, 12]
seeds     = [42, 123, 999]

configs   = list(itertools.product(pop_sizes, n_gens, seeds))
print(f"\nTotal configs to run: {len(configs)}")

run_results = []
for pop, gen, seed in configs:
    r = run_config(
        pop, gen, seed,
        log_csv=f"outputs/hyperparam_pop{pop}_gen{gen}_seed{seed}_5.csv"
    )
    run_results.append(r)

# ── Stability analysis ────────────────────────────────────────────────────────

df_runs = pd.DataFrame([
    {k: v for k, v in r.items()
     if k not in ("hv_history", "sharpe_history", "result_obj")}
    for r in run_results
])

stability = df_runs.groupby(["pop_size", "n_gen"]).agg(
    mean_sharpe  = ("best_sharpe", "mean"),
    std_sharpe   = ("best_sharpe", "std"),
    max_sharpe   = ("best_sharpe", "max"),
    mean_ann     = ("best_ann",    "mean"),
    mean_evals   = ("n_evals",     "mean"),
    mean_complex = ("complexity",  "mean"),
    mean_gens    = ("actual_gens", "mean")
).reset_index()

stability["score"] = (
    stability["mean_sharpe"]
    - 0.5  * stability["std_sharpe"]
    - 0.05 * stability["mean_complex"]
)

stability = stability.sort_values("score", ascending=False).reset_index(drop=True)

print("\nStability Analysis:")
print(stability.to_string(index=False))
stability.to_csv("outputs/stability_analysis_5.csv", index=False)

# ── Plot 1 — convergence by Sharpe ───────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
axes = axes.flatten()
config_labels = [(p, g) for p, g in itertools.product(pop_sizes, n_gens)]

for ax_idx, (pop, gen) in enumerate(config_labels):
    ax = axes[ax_idx]
    for r in run_results:
        if r["pop_size"] == pop and r["n_gen"] == gen:
            ax.plot(r["sharpe_history"], label=f"seed={r['seed']}", alpha=0.8)
    ax.set_title(f"pop={pop}, gen={gen}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Sharpe")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Convergence — Best Sharpe by Generation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/convergence_sharpe_5.png", dpi=150)
plt.close()
print("Saved: outputs/convergence_sharpe_5.png")

# ── Plot 2 — convergence by hypervolume ──────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
axes = axes.flatten()

for ax_idx, (pop, gen) in enumerate(config_labels):
    ax = axes[ax_idx]
    for r in run_results:
        if r["pop_size"] == pop and r["n_gen"] == gen:
            ax.plot(r["hv_history"], label=f"seed={r['seed']}", alpha=0.8)
    ax.set_title(f"pop={pop}, gen={gen}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Convergence — Pareto Front Hypervolume by Generation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/convergence_hv.png", dpi=150)
plt.close()
print("Saved: outputs/convergence_hv.png")

# ── Best config + Pareto fronts ───────────────────────────────────────────────

best_config = stability.iloc[0]
best_pop    = int(best_config["pop_size"])
best_gen    = int(best_config["n_gen"])

print(f"\nBest config: pop={best_pop}, gen={best_gen} "
      f"(score={best_config['score']:.3f})")

best_logs = []
for r in run_results:
    if r["pop_size"] == best_pop and r["n_gen"] == best_gen:
        fname = f"outputs/hyperparam_pop{best_pop}_gen{best_gen}_seed{r['seed']}_5.csv"
        if os.path.isfile(fname):
            best_logs.append(pd.read_csv(fname))

df_best = pd.concat(best_logs, ignore_index=True).drop_duplicates()
df_best = df_best[df_best["sharpe"] < 1e5]

pareto_best = get_pareto_front(df_best).sort_values("sharpe", ascending=False).reset_index(drop=True)
pareto_best.to_csv("outputs/pareto_front_best_config_5.csv", index=False)
print(f"  Pareto front (best config): {len(pareto_best)} solutions")

all_logs = []
for pop in pop_sizes:
    for gen in n_gens:
        for seed in seeds:
            fname = f"outputs/hyperparam_pop{pop}_gen{gen}_seed{seed}_5.csv"
            if os.path.isfile(fname):
                all_logs.append(pd.read_csv(fname))

df_all = pd.concat(all_logs, ignore_index=True).drop_duplicates()
df_all = df_all[df_all["sharpe"] < 1e5]
df_all.to_csv("outputs/all_train_evaluations_5.csv", index=False)

pareto_all = get_pareto_front(df_all).sort_values("sharpe", ascending=False).reset_index(drop=True)
pareto_all.to_csv("outputs/pareto_front_all_configs_5.csv", index=False)
print(f"  Pareto front (all configs):  {len(pareto_all)} solutions")
print(f"  Total unique evaluations:    {len(df_all)}")

weight_cols = [f"w{i}" for i in range(11)]
print("\nTop 10 Pareto solutions (best config):")
print(pareto_best.head(10)[weight_cols + ["sharpe", "annualized", "var"]].to_string(index=False))

print("\nTop 10 Pareto solutions (all configs):")
print(pareto_all.head(10)[weight_cols + ["sharpe", "annualized", "var"]].to_string(index=False))

# ── Save taboo set ────────────────────────────────────────────────────────────

with open("outputs/taboo_set.pkl", "wb") as f:
    pickle.dump(taboo_set, f)

print("\nAll outputs saved to outputs/")
print("   - stability_analysis_5.csv")
print("   - convergence_sharpe_5.png")
print("   - convergence_hv_5.png")
print("   - pareto_front_best_config_5.csv")
print("   - pareto_front_all_configs_5.csv")
print("   - all_train_evaluations_5.csv")
print("   - taboo_set_5.pkl")
print("\nNext step: run moga_validate_5.py")
