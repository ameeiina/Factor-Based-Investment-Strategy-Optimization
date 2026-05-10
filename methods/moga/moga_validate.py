# moga_validate.py — Validate MOGA Pareto-front solutions on 2019-2020 held-out data
#
# Loads pareto_front_best_config.csv and pareto_front_all_configs.csv from moga_optimize.py,
# runs each solution in QRUMBLE on the 2019-2020 validation period, and scores by robustness
# (val_sharpe − degradation penalty − complexity penalty). Saves ranked results, a train vs
# validation scatter plot, and the recommended strategy to final_recommendation.pkl.
#
# Input  : outputs/pareto_front_*.csv  (produced by moga_optimize.py)
# Output : outputs/
# DO NOT use python3 — QRUMBLE is only available in the conda env named Q

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import qrumble as q
from qrumble.factors import *

# ── Setup ─────────────────────────────────────────────────────────────────────

os.makedirs("outputs", exist_ok=True)

q.universe.load("../../data/UNIVERSE.2015_2019.pickle", index="stoxx600")
funda_test = q.load("../../data/FUNDAMENTALS.2019_2020.pickle")
ohlcv_test = q.load("../../data/stoxx600.jan2018_apr2021.pickle")

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

weight_cols = [f"w{i}" for i in range(11)]

# ── Load Pareto fronts from optimise step ─────────────────────────────────────

pareto_best = pd.read_csv("outputs/pareto_front_best_config_5.csv")
pareto_all  = pd.read_csv("outputs/pareto_front_all_configs_5.csv")

print(f"Pareto front (best config) : {len(pareto_best)} solutions")
print(f"Pareto front (all configs) : {len(pareto_all)} solutions")

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

# ── Validation helpers ────────────────────────────────────────────────────────

def validate_weights(weights, start_date="jan2019", period="1y"):
    weights = np.array(weights).astype(int)
    try:
        perf, _ = q.qrumble(
            "moga:{DATE}/rank/30", "moga:{DATE}/play/30/1y",
            start_date, period, rebalance="1y",
            fundamentals=funda_test, ohlcv=ohlcv_test, Rf=RF,
            universe_list=q.universe.fetch,
            universe=lambda f: f.Screening(~Sector().isin([
                'banking', 'insurance', 'real estate',
                'financial', 'utilities', 'waste'
            ])),
            criteria=lambda f: get_selection(f, weights),
            dataframe=True
        )
        return perf['sharpe'], perf['annualized'], perf['var']
    except Exception as e:
        print(f"  Validation failed for {weights}: {e}")
        return None, None, None


def validate_pareto(pareto_df, label):
    print(f"\nValidating [{label}] — {len(pareto_df)} solutions on 2019-2020...")
    val_records = []

    for i, row in pareto_df.reset_index(drop=True).iterrows():
        weights = row[weight_cols].values.astype(int)
        print(f"  [{i+1}/{len(pareto_df)}] {weights}")

        val_sharpe, val_ann, val_var = validate_weights(weights)

        val_records.append({
            **{f"w{j}": int(weights[j]) for j in range(11)},
            "train_sharpe":  row["sharpe"],
            "train_ann":     row["annualized"],
            "train_var":     row["var"],
            "val_sharpe":    val_sharpe,
            "val_ann":       val_ann,
            "val_var":       val_var,
            "degradation":   round(row["sharpe"] - val_sharpe, 4) if val_sharpe else None,
            "complexity":    int(np.count_nonzero(weights)),
            "source":        label
        })

    df = pd.DataFrame(val_records)
    df["robustness_score"] = (
        df["val_sharpe"]
        - 0.3  * df["degradation"]
        - 0.05 * df["complexity"]
    )
    return df.sort_values("robustness_score", ascending=False).reset_index(drop=True)

# ── Run validation on both Pareto fronts ──────────────────────────────────────

df_val_best = validate_pareto(pareto_best, label="best_config")
df_val_all  = validate_pareto(pareto_all,  label="all_configs")

df_val_best.to_csv("outputs/validation_best_config_5.csv", index=False)
df_val_all.to_csv("outputs/validation_all_configs_5.csv",  index=False)

df_combined = pd.concat([df_val_best, df_val_all], ignore_index=True)
df_combined = (df_combined
               .sort_values("robustness_score", ascending=False)
               .drop_duplicates(subset=weight_cols)
               .reset_index(drop=True))
df_combined.to_csv("outputs/validation_combined_5.csv", index=False)

# ── Print results ─────────────────────────────────────────────────────────────

display_cols = ["train_sharpe", "train_ann", "val_sharpe", "val_ann",
                "degradation", "complexity", "robustness_score", "source"]

print("\nValidation — Best Config Pareto:")
print(df_val_best[display_cols].to_string(index=False))

print("\nValidation — All Configs Pareto:")
print(df_val_all[display_cols].to_string(index=False))

print("\nValidation — Combined (deduplicated, sorted by robustness):")
print(df_combined[display_cols].to_string(index=False))

# ── Train vs val scatter plot ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, df, title in zip(
    axes,
    [df_val_best, df_val_all],
    ["Best Config (pop=50, gen=8)", "All Configs Combined"]
):
    sc = ax.scatter(
        df["train_sharpe"], df["val_sharpe"],
        c=df["complexity"], cmap="RdYlGn_r",
        s=100, edgecolors="black", linewidths=0.5,
        vmin=3, vmax=11
    )
    plt.colorbar(sc, ax=ax, label="# active factors")

    best_row = df.iloc[0]
    ax.annotate(
        "Best robust",
        xy=(best_row["train_sharpe"], best_row["val_sharpe"]),
        xytext=(10, -15), textcoords="offset points",
        fontsize=8, color="darkgreen",
        arrowprops=dict(arrowstyle="->", color="darkgreen")
    )

    all_vals = pd.concat([df["train_sharpe"], df["val_sharpe"]])
    lims = [all_vals.min() - 0.05, all_vals.max() + 0.05]
    ax.plot(lims, lims, "k--", alpha=0.4, label="No degradation")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Train Sharpe (2015-2019)")
    ax.set_ylabel("Val Sharpe (2019-2020)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Train vs Validation Sharpe\n(color = # active factors)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/train_vs_val_scatter_5.png", dpi=150)
plt.close()
print("\nSaved: outputs/train_vs_val_scatter_5.png")

# ── Final recommendation ──────────────────────────────────────────────────────

best = df_combined.iloc[0]
best_weights = [int(best[f"w{j}"]) for j in range(11)]

print("\n" + "="*55)
print("RECOMMENDED STRATEGY")
print("="*55)
for j, (name, w) in enumerate(zip(factor_names, best_weights)):
    if w != 0:
        direction = "ascending  ↑" if w == 1 else "descending ↓"
        print(f"  w{j:<3} {name:<25} {direction}")
print(f"\n  Train Sharpe    : {best['train_sharpe']:.3f}")
print(f"  Val   Sharpe    : {best['val_sharpe']:.3f}")
print(f"  Degradation     : {best['degradation']:.3f}")
print(f"  Complexity      : {int(best['complexity'])} active factors")
print(f"  Found by        : {best['source']}")
print("="*55)

with open("outputs/final_recommendation.pkl", "wb") as f:
    pickle.dump({
        "weights":     best_weights,
        "train_stats": {
            "sharpe":     best["train_sharpe"],
            "annualized": best["train_ann"],
            "var":        best["train_var"]
        },
        "val_stats": {
            "sharpe":     best["val_sharpe"],
            "annualized": best["val_ann"],
            "var":        best["val_var"]
        },
        "robustness_score": best["robustness_score"],
        "source":           best["source"]
    }, f)

print("\nAll outputs saved to outputs/")
print("   - validation_best_config_5.csv")
print("   - validation_all_configs_5.csv")
print("   - validation_combined_5.csv")
print("   - train_vs_val_scatter_5.png")
print("   - final_recommendation_5.pkl")
print("\nNext step: run moga_final_oos.py (true out-of-sample)")
