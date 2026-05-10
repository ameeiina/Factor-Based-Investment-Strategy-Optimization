# random_search.py — One-round QRUMBLE factor search with manual pfactor control
#
# Run once per round. After each round, inspect the SOM heatmaps produced by
# visualize.py, then edit PFACTORS below to bias the next round toward the
# factor directions that appear in high-Sharpe map regions. Change ROUND and
# SAMPLES before each run. Combines all 3 seeds into a deduplicated .data file
# for MasterSOM.java. Linear normalisation fixed across all rounds.
#
# CLI    : conda run -n Q python random_search.py
# Output : outputs/round{ROUND}/
# DO NOT use python3 — QRUMBLE is only available in the conda env named Q

import os
import random
import logging

import numpy as np
import pandas as pd

import qrumble as q
from qrumble.factors import *
import qrumble.universe
import qrumble.optimizer

os.environ["MKL_VERBOSE"] = "0"

# ── Search parameters ─────────────────────────────────────────────────────────

ROUND   = 1              # change to 2 or 3 for subsequent rounds
SEEDS   = [42, 123, 999]
SAMPLES = 2000           # round 1: 2000  |  round 2: 1500  |  round 3: 500

# ── Data paths ────────────────────────────────────────────────────────────────

UNIVERSE     = "../../data/UNIVERSE.2015_2019.pickle"
FUNDAMENTALS = "../../data/FUNDAMENTALS.2015_2019.pickle"
OHLCV        = "../../data/stoxx600.jan2014_dec2019.pickle"
RF           = -0.6

q.optimizer.logging(logging.INFO)

os.makedirs(f"outputs/round{ROUND}", exist_ok=True)

# ── Pfactors — edit after inspecting the SOM heatmaps ────────────────────────
#
# Each entry sets the sampling probability [p(0), p(+1), p(-1)] for one factor.
#
#   eq     : uniform — factor unexplored or signal unclear
#   prob_1 : bias toward -1 — descending rank preferred in high-Sharpe region
#   prob1  : bias toward +1 — ascending rank preferred in high-Sharpe region
#   prob0  : bias toward  0 — factor confirmed inactive or harmful
#
# Round 1: keep all as eq (uniform exploration — do not change).
# Round 2+: replace eq with prob_1 / prob1 / prob0 based on heatmap inspection.

eq     = q.optimizer.proba([0, 1, -1], p=[0.34, 0.33, 0.33])
prob_1 = q.optimizer.proba([0, 1, -1], p=[0.10, 0.10, 0.80])
prob1  = q.optimizer.proba([0, 1, -1], p=[0.10, 0.80, 0.10])
prob0  = q.optimizer.proba([0, 1, -1], p=[0.80, 0.10, 0.10])

PFACTORS = [
    eq,   # Yield
    eq,   # ROC
    eq,   # EarningsYield
    eq,   # RS(6m)
    eq,   # ROA
    eq,   # ΔROA
    eq,   # AccrualRatio
    eq,   # ΔLTDebt-to-Assets
    eq,   # ΔCurrentRatio
    eq,   # ΔOpMgn
    eq,   # ΔAssetTurnover
]

# ── Factor and sector configuration ──────────────────────────────────────────

FACTOR_UNIVERSE = [
    Yield(), ROC(), EarningsYield(), RS('6m'),
    ROA(), ROA_diff(), AccrualRatio(),
    LTDebt_to_Assets_diff(), CurrentRatio_diff(),
    OpMgn_diff(), AssetTurnover_diff()
]

FACTOR_NAMES = [
    "Yield", "ROC", "EarningsYield", "RS(6m)",
    "ROA", "ΔROA", "AccrualRatio",
    "ΔLTDebt-to-Assets", "ΔCurrentRatio",
    "ΔOpMgn", "ΔAssetTurnover",
]

SECTOR_EXCLUSIONS = [
    'utilities', 'waste',
    'banking', 'insurance', 'real estate', 'financial',
]

VARIABLE_NAMES = FACTOR_NAMES + [
    "annualized", "mean", "std", "sharpe",
    "alpha", "beta", "var", "tvar",
]
N_INPUTS  = 19
N_OUTPUTS = 0

RENAME_MAP = {
    "ROA_diff":           "ΔROA",
    "LTDebt_Assets_diff": "ΔLTDebt-to-Assets",
    "CurrentRatio_diff":  "ΔCurrentRatio",
    "OpMgn_diff":         "ΔOpMgn",
    "AssetTurnover_diff": "ΔAssetTurnover",
    "RS_6m":              "RS(6m)",
}

INTERVALS = {
    "annualized": (-1.1,  66.0),
    "mean":       ( 0.9,  55.0),
    "std":        (10.8,  33.0),
    "sharpe":     ( 0.09,  1.98),
    "alpha":      (-44.0, 220.0),
    "beta":       ( 0.72,  1.43),
    "var":        (-39.6, -18.0),
    "tvar":       (-73.7, -26.1),
}

# ── Random search subclass ────────────────────────────────────────────────────

class MFRandom(q.optimizer.RandomSearch):
    def name(self, top, rnkformula):
        return f"magicformula:({top},{rnkformula})"

    def call(self, univ, funda, ohlcv, top, rnkformula):
        ranking_layout = self.name(top, rnkformula) + f":{{DATE}}/rank/{top}"
        investm_layout = self.name(top, rnkformula) + f":{{DATE}}/play/{top}/1y"
        q.universe.load(**univ)
        return q.qrumble(
            ranking_layout, investm_layout,
            "jan2015", "5y", rebalance="1y",
            fundamentals=funda, ohlcv=ohlcv, Rf=RF,
            universe_list=q.universe.fetch,
            universe=lambda f: f.Screening(~Sector().isin(SECTOR_EXCLUSIONS)),
            criteria=lambda f: f.Ranking(rnkformula.code(), top=top)
        )

# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_raw(df):
    metrics_norm = (
        df.iloc[:, -12:]
        .drop(columns=["roi", "alpha_daily", "var_daily", "tvar_daily"])
        .apply(lambda col: np.clip(
            1000 * (col - INTERVALS[col.name][0]) /
            (INTERVALS[col.name][1] - INTERVALS[col.name][0]),
            0, 1000
        ))
        .astype(int)
    )
    factor_cols = (
        df.iloc[:, :-12]
        .drop(columns=["top"], errors="ignore")
        .replace({-1: 0, 0: 500, 1: 1000})
    )
    return pd.concat([factor_cols, metrics_norm], axis=1)

# ── .data file writer ─────────────────────────────────────────────────────────

def write_data_file(data, output_path, n_unique):
    header  = "---\n"
    header += f'name: "SOM Round {ROUND} — {n_unique} samples"\n'
    header += f"description: Factor-based strategy search Round {ROUND}\n"
    header += "variables:\n"
    for name in VARIABLE_NAMES:
        header += f"    - {name}\n"
    header += f"inputs: {N_INPUTS}\n"
    header += f"outputs: {N_OUTPUTS}\n"
    header += "---\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for row in data.values:
            f.write(",".join(map(str, row)) + "\n")
    print(f"  .data file  : {output_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  SOM Random Search — Round {ROUND}")
    print(f"  Seeds   : {SEEDS}  |  Samples per seed: {SAMPLES}")
    print(f"  Total   : {len(SEEDS) * SAMPLES} evaluations")
    print(f"{'='*60}\n")

    univ  = q.universe.load(UNIVERSE, index="stoxx600")
    funda = q.load(FUNDAMENTALS)
    ohlcv = q.load(OHLCV)

    all_raw_dfs  = []
    all_norm_dfs = []

    for seed in SEEDS:
        print(f"\n{'─'*60}")
        print(f"  Seed {seed}")
        for i, (name, p) in enumerate(zip(FACTOR_NAMES, PFACTORS)):
            tag = next(
                label for label, obj in [
                    ("eq    ", eq), ("prob_1", prob_1),
                    ("prob1 ", prob1), ("prob0 ", prob0)
                ] if p is obj
            )
            print(f"    w{i:<2} {name:<22} {tag}")
        print(f"{'─'*60}")

        np.random.seed(seed)
        random.seed(seed)

        rnkformulas = q.optimizer.Solutions(FACTOR_UNIVERSE, PFACTORS)
        rnd = MFRandom(top=[30], rnkformula=rnkformulas)
        rnd.run(univ, funda, ohlcv, samples=SAMPLES)

        for metric in ("sharpe", "annualized"):
            rnd.metric = metric
            print(f"  Best '{metric}': {rnd.best_}")

        raw_csv  = f"outputs/round{ROUND}/seed{seed}_{SAMPLES}_raw.csv"
        norm_csv = f"outputs/round{ROUND}/seed{seed}_{SAMPLES}_normalized.csv"

        rnd.to_csv(raw_csv)
        print(f"  Raw CSV     : {raw_csv}")

        df_raw  = pd.read_csv(raw_csv)
        df_norm = normalize_raw(df_raw)
        df_norm.to_csv(norm_csv, index=False)
        print(f"  Norm CSV    : {norm_csv}")

        all_raw_dfs.append(df_raw)
        all_norm_dfs.append(df_norm)

    # ── Combine and deduplicate ───────────────────────────────────────────────

    combined_raw  = pd.concat(all_raw_dfs,  ignore_index=True)
    combined_norm = pd.concat(all_norm_dfs, ignore_index=True)

    factor_col_names = list(combined_norm.columns[:11])
    combined_norm_dedup = (
        combined_norm
        .drop_duplicates(subset=factor_col_names)
        .reset_index(drop=True)
    )
    combined_raw_dedup = (
        combined_raw
        .iloc[combined_norm_dedup.index]
        .reset_index(drop=True)
    )

    n_unique  = len(combined_norm_dedup)
    n_total   = len(combined_norm)

    print(f"\n{'='*60}")
    print(f"  Combined Round {ROUND} dataset")
    print(f"  Total evaluations : {n_total}")
    print(f"  Unique configs    : {n_unique}")
    print(f"  Duplicates removed: {n_total - n_unique}")

    combined_raw_dedup.to_csv(
        f"outputs/round{ROUND}/combined_{n_unique}_raw.csv", index=False
    )
    combined_norm_dedup.to_csv(
        f"outputs/round{ROUND}/combined_{n_unique}_normalized.csv", index=False
    )

    # Rename diff columns and write .data file
    data = combined_norm_dedup.rename(columns=RENAME_MAP)[VARIABLE_NAMES]
    write_data_file(data, f"outputs/round{ROUND}/combined_{n_unique}.data", n_unique)

    # ── Summary ───────────────────────────────────────────────────────────────

    raw_metrics = combined_raw_dedup.iloc[:, -12:].drop(
        columns=["roi", "alpha_daily", "var_daily", "tvar_daily"]
    )
    print(f"\n  Performance metric ranges (combined):")
    for col in ("sharpe", "annualized", "var"):
        print(
            f"    {col:<12}: "
            f"min={raw_metrics[col].min():.3f}  "
            f"max={raw_metrics[col].max():.3f}  "
            f"mean={raw_metrics[col].mean():.3f}"
        )

    print(f"\n{'='*60}")
    print(f"  Next step:")
    print(f"  1. Open MasterSOM.java")
    print(f"  2. Set dataset path to:")
    print(f"     outputs/round{ROUND}/combined_{n_unique}.data")
    print(f"  3. Run MasterSOM.java, then run visualize.py on the output")
    print(f"  4. Inspect heatmaps and update PFACTORS for round {ROUND + 1}")
    print(f"{'='*60}\n")
