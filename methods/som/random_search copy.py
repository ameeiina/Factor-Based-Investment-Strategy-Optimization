# random_search copy.py — Original single-round search script for Round 2 (archived)
#
# Earlier version of random_search.py before it was generalised to all three rounds.
# Hardcodes ROUND=2 with per-seed pfactor configurations (seeds 43, 124, 998) derived
# from manual inspection of the Round 1 SOM heatmaps. Uses sigmoid normalisation instead
# of the linear normalisation adopted in the final pipeline. Kept for reference only —
# use random_search.py for all new runs.
#
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

# ============================================================
# CONFIGURATION
# ============================================================
ROUND   = 2
SEEDS   = [43, 124, 998]   # one seed per pfactors configuration
SAMPLES = 500          # evaluations per seed

# ============================================================
# DATA
# ============================================================
UNIVERSE     = "data-stoxx600/UNIVERSE.2015_2019.pickle"
FUNDAMENTALS = "data-stoxx600/FUNDAMENTALS.2015_2019.pickle"
OHLCV        = "data-stoxx600/stoxx600.jan2014_dec2019.pickle"
RF           = -0.6

q.optimizer.logging(logging.INFO)

os.makedirs("outputs/som", exist_ok=True)

# ============================================================
# SECTOR EXCLUSIONS  (identical to GP and NSGA-II)
# ============================================================
utilities  = ['utilities', 'waste']
financials = ['banking', 'insurance', 'real estate', 'financial']

# ============================================================
# FACTOR UNIVERSE  (identical to GP and NSGA-II)
# ============================================================
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

# ============================================================
# RANDOM SEARCH SUBCLASS
# ============================================================
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
            universe=lambda f: f.Screening(
                ~Sector().isin(financials + utilities)
            ),
            criteria=lambda f: f.Ranking(rnkformula.code(), top=top)
        )

# ============================================================
# SIGMOID NORMALIZATION
# Same fixed intervals as round 1 — do not change.
# ============================================================
def sigmoid_from_interval(x, a, b):
    x0 = (a + b) / 2
    k  = 4 / (b - a)
    return 1 / (1 + np.exp(-k * (x - x0)))

INTERVALS = {
    'annualized': (-1.1,  66.0),
    'mean':       ( 0.9,  55.0),
    'std':        (10.8,  33.0),
    'sharpe':     ( 0.09,  1.98),
    'alpha':      (-44.0, 220.0),
    'beta':       ( 0.72,  1.43),
    'var':        (-39.6, -18.0),
    'tvar':       (-73.7, -26.1),
}

# ============================================================
# NORMALIZE HELPER
# ============================================================
def normalize_raw(df):
    metrics_norm = (
        df.iloc[:, -12:]
        .drop(columns=["roi", "alpha_daily", "var_daily", "tvar_daily"])
        .apply(
            lambda col: sigmoid_from_interval(
                col, *INTERVALS[col.name]
            ) * 1000
        )
        .astype(int)
    )
    factor_cols = (
        df.iloc[:, :-12]
        .drop(columns=["top"], errors="ignore")
        .replace({-1: 0, 0: 500, 1: 1000})
    )
    return pd.concat([factor_cols, metrics_norm], axis=1)


# ============================================================
# WRITE .data FILE
# ============================================================
def write_data_file(combined_df, path, round_num, total_samples):
    variables = list(combined_df.columns)
    n_total   = len(variables)
    n_inputs  = 11
    n_outputs = n_total - n_inputs
    assert n_inputs + n_outputs == n_total, \
        f"Header mismatch: {n_inputs} + {n_outputs} != {n_total}"

    header  = "---\n"
    header += f'name: "SOM Round {round_num} — {total_samples} samples'\
              f' (3 seeds combined)"\n'
    header += f"description: Factor-based strategy search Round {round_num}\n"
    header += "variables:\n"
    for var in variables:
        header += f"    - {var}\n"
    header += f"inputs: {n_inputs}\n"
    header += f"outputs: {n_outputs}\n"
    header += "---\n"

    with open(path, 'w') as f:
        f.write(header)
        for _, row in combined_df.iterrows():
            f.write(",".join(map(str, row.values)) + "\n")
    print(f".data file saved to: {path}")


# ============================================================
# PFACTORS — one configuration per seed
# ── Change these based on your round 1 SOM inspection ───────
#
# Rule applied here (relevance interpretation):
#   prob1_1 → factor active in high-Sharpe region (far from 500)
#   eq      → factor inactive or noisy (near 500)
#
# Seed 42  — main config from SOM inspection
# Seed 123 — slightly more conservative (fewer prob1_1)
# Seed 999 — slightly more aggressive (more prob1_1)
# ============================================================
eq      = q.optimizer.proba([0, 1, -1], p=[0.34, 0.33, 0.33])
prob_1  = q.optimizer.proba([0, 1, -1], p=[0.10, 0.10, 0.80])
prob1   = q.optimizer.proba([0, 1, -1], p=[0.10, 0.80, 0.10])
prob0   = q.optimizer.proba([0, 1, -1], p=[0.80, 0.10, 0.10])

PFACTORS_PER_SEED = {
    # ── Seed 42 — main config from SOM inspection ───────────
    # w0,w1,w2,w3,w6,w7 active; w4,w5,w8,w9,w10 uniform
        43: [
        prob_1,  # Yield                - clear blue in high-Sharpe region
        prob_1,  # ROC                  - clear blue in high-Sharpe region
        eq,      # EarningsYield        - mixed in high-Sharpe region
        eq,      # RS(6m)               - no coherent pattern
        prob_1,  # ROA                  - clear blue in high-Sharpe region
        eq,      # ROA_diff             - inconsistent
        eq,      # AccrualRatio         - no strong gradient
        prob0,   # LTDebt_to_Assets_diff - flat, confirmed by Table 6.3
        eq,      # CurrentRatio_diff    - mixed
        eq,      # OpMgn_diff           - weak signal
        prob1,   # AssetTurnover_diff   - strongest signal, red = +1 in high-Sharpe region
    ],
    # ── Seed 123 — conservative: only clearest signals ──────
    # Only w0 and w1 have very clear signals; rest kept uniform
        124: [
        prob_1,  # Yield                - consistent with seed 42
        prob_1,  # ROC                  - consistent with seed 42
        eq,      # EarningsYield        - consistent with seed 42
        eq,      # RS(6m)               - consistent with seed 42
        prob_1,  # ROA                  - consistent with seed 42
        eq,      # ROA_diff             - consistent with seed 42
        eq,      # AccrualRatio         - consistent with seed 42
        prob0,   # LTDebt_to_Assets_diff - consistent with seed 42
        eq,      # CurrentRatio_diff    - consistent with seed 42
        eq,      # OpMgn_diff           - consistent with seed 42
        prob_1,  # AssetTurnover_diff   - FLIPPED from seed 42
    ],
    # ── Seed 999 — aggressive: all non-noisy factors active ─
    # Includes borderline factors w9 and w3
    998: [
        prob_1,  # Yield                - unanimous across seeds
        prob_1,  # ROC                  - unanimous across seeds
        eq,      # EarningsYield        - unanimous across seeds
        eq,      # RS(6m)               - unanimous across seeds
        prob_1,  # ROA                  - unanimous across seeds
        eq,      # ROA_diff             - unanimous across seeds
        eq,      # AccrualRatio         - unanimous across seeds
        prob0,   # LTDebt_to_Assets_diff - unanimous across seeds
        eq,      # CurrentRatio_diff    - unanimous across seeds
        eq,      # OpMgn_diff           - unanimous across seeds
        prob_1,  # AssetTurnover_diff   - majority 2/3 seeds
    ],
}

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  SOM Random Search — Round {ROUND}")
    print(f"  Seeds   : {SEEDS}  |  Samples per seed: {SAMPLES}")
    print(f"  Total   : {len(SEEDS) * SAMPLES} evaluations")
    print(f"  top=[30]  |  3 different pfactors configurations")
    print(f"{'='*60}\n")

    # ---- Load data once ----
    univ  = q.universe.load(UNIVERSE, index="stoxx600")
    funda = q.load(FUNDAMENTALS)
    ohlcv = q.load(OHLCV)

    all_raw_dfs  = []
    all_norm_dfs = []

    for seed in SEEDS:
        pfactors = PFACTORS_PER_SEED[seed]

        print(f"\n{'─'*60}")
        print(f"  Seed {seed}")
        print(f"  Active factors (prob_1):")
        for i, (name, p) in enumerate(zip(factor_names, pfactors)):
            tag = "prob_1" if p is prob_1 else "eq     "
            print(f"    w{i:<3} {name:<25} {tag}")
        print(f"{'─'*60}")

        np.random.seed(seed)
        random.seed(seed)

        rnkformulas = q.optimizer.Solutions(factor_universe, pfactors)
        rnd = MFRandom(top=[30], rnkformula=rnkformulas)
        rnd.run(univ, funda, ohlcv, samples=SAMPLES)

        # ---- Per-seed best ----
        print()
        for metric in ('sharpe', 'annualized'):
            rnd.metric = metric
            print(f"  Best on '{metric}': {rnd.best_}")

        # ---- Save per-seed raw CSV ----
        raw_csv = (
            f"outputs/som/round{ROUND}_seed{seed}_{SAMPLES}_raw.csv"
        )
        rnd.to_csv(raw_csv)
        print(f"  Raw CSV: {raw_csv}")

        # ---- Normalize ----
        df_raw  = pd.read_csv(raw_csv)
        df_norm = normalize_raw(df_raw)

        norm_csv = (
            f"outputs/som/round{ROUND}_seed{seed}_{SAMPLES}_normalized.csv"
        )
        df_norm.to_csv(norm_csv, index=False)
        print(f"  Norm CSV: {norm_csv}")

        all_raw_dfs.append(df_raw)
        all_norm_dfs.append(df_norm)

    # --------------------------------------------------------
    # Combine all three seeds
    # --------------------------------------------------------
    combined_raw  = pd.concat(all_raw_dfs,  ignore_index=True)
    combined_norm = pd.concat(all_norm_dfs, ignore_index=True)

    # Deduplicate on factor direction columns
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

    total_unique = len(combined_norm_dedup)
    total_raw    = len(combined_norm)

    print(f"\n{'='*60}")
    print(f"  Combined Round {ROUND} dataset")
    print(f"{'='*60}")
    print(f"  Total evaluations (3 seeds) : {total_raw}")
    print(f"  Unique configurations       : {total_unique}")
    print(f"  Duplicates removed          : {total_raw - total_unique}")

    # ---- Save combined CSVs ----
    combined_raw_path  = (
        f"outputs/som/round{ROUND}_combined_{total_unique}_raw.csv"
    )
    combined_norm_path = (
        f"outputs/som/round{ROUND}_combined_{total_unique}_normalized.csv"
    )
    combined_raw_dedup.to_csv(combined_raw_path,  index=False)
    combined_norm_dedup.to_csv(combined_norm_path, index=False)
    print(f"\n  Combined raw CSV  : {combined_raw_path}")
    print(f"  Combined norm CSV : {combined_norm_path}")

    # ---- Write .data file for MasterSOM.java ----
    data_file = (
        f"outputs/som/round{ROUND}_combined_{total_unique}.data"
    )
    write_data_file(
        combined_norm_dedup, data_file, ROUND, total_unique
    )

    # --------------------------------------------------------
    # SUMMARY STATISTICS
    # --------------------------------------------------------
    raw_metrics = combined_raw_dedup.iloc[:, -12:].drop(
        columns=["roi", "alpha_daily", "var_daily", "tvar_daily"]
    )

    print(f"\n  Performance metric ranges (combined):")
    for col in ['sharpe', 'annualized', 'var']:
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
    print(f"     {data_file}")
    print(f"  3. Run MasterSOM.java — train UbiSOM on combined")
    print(f"     round 1 + round 2 data for final inspection")
    print(f"{'='*60}\n")