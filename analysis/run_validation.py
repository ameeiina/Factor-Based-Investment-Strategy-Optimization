# run_validation.py — True out-of-sample validation of top strategies from all three methods
#
# Takes one or more top-N CSV files (top6_SOM.csv, top6_GP.csv, top6_MOGA.csv), runs each
# strategy in QRUMBLE on the 2020-2023 out-of-sample period, and saves a structured results
# CSV alongside a full text log. Each row in the output records both training-period metrics
# (from the input CSV) and OOS metrics (Sharpe, return, VaR, alpha, beta, etc.) for direct
# comparison. Results feed Table 7.4 in the thesis.
#
# Input  : data/top6_SOM.csv  data/top6_GP.csv  data/top6_MOGA.csv
# Output : outputs/validation_results_<timestamp>.csv
#          outputs/validation_log_<timestamp>.txt
# CLI    : conda run -n Q python run_validation.py data/top6_SOM.csv data/top6_GP.csv data/top6_MOGA.csv
# DO NOT use python3 — QRUMBLE is only available in the conda env named Q

import os
import sys
import logging
import datetime
import argparse
import traceback

import pandas as pd

import qrumble as q
from qrumble.factors import *
import qrumble.universe

# ── Out-of-sample data paths ──────────────────────────────────────────────────

UNIVERSE     = "../data/UNIVERSE.2020_2023.pickle"
FUNDAMENTALS = "../data/FUNDAMENTALS.2020_2023.pickle"
OHLCV        = "../data/stoxx600.jan2019_jun2024.pickle"

OOS_START    = "feb2020"
OOS_DURATION = "53m"       # Feb 2020 – Jun 2024
REBALANCE    = "1y"
RF           = 0.6        # risk-free rate (%)
TOP          = 30          # portfolio size

# ── Sector exclusions (same as training) ──────────────────────────────────────

UTILITIES  = ['utilities', 'waste', 'electricity']
FINANCIALS = ['banking', 'insurance', 'real estate', 'financial']

# ── Factor universe (same order as training, indices 0-10) ────────────────────

FACTOR_OBJECTS = [
    Yield(),                    # w0
    ROC(),                      # w1
    EarningsYield(),            # w2
    RS('6m'),                   # w3
    ROA(),                      # w4
    ROA_diff(),                 # w5
    AccrualRatio(),             # w6
    LTDebt_to_Assets_diff(),    # w7
    CurrentRatio_diff(),        # w8
    OpMgn_diff(),               # w9
    AssetTurnover_diff(),       # w10
]

FACTOR_COLS = [
    'Yield', 'ROC', 'EarningsYield', 'RS(6m)', 'ROA',
    'ΔROA', 'AccrualRatio', 'ΔLTDebt-to-Assets',
    'ΔCurrentRatio', 'ΔOpMgn', 'ΔAssetTurnover',
]

# ── Output directory ──────────────────────────────────────────────────────────

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TIMESTAMP   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_CSV = os.path.join(OUT_DIR, f"validation_results_{TIMESTAMP}.csv")
LOG_TXT     = os.path.join(OUT_DIR, f"validation_log_{TIMESTAMP}.txt")

# ── Logging setup ─────────────────────────────────────────────────────────────

class TeeLogger:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log      = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ── Criteria builder ──────────────────────────────────────────────────────────

def make_criteria(directions):
    """
    Returns a criteria function to pass as criteria= to q.qrumble().

    Matches the pattern from 3_MOGA_validate.py (get_selection):
      - factor expressions built INSIDE the function at call time
      - f.Ranking() called as a side-effect
      - returns f

    Mapping:
        +1 → Factor()(ascending)   — prefer stocks with HIGH values
        -1 → Factor()(descending)  — prefer stocks with LOW values
         0 → excluded from ranking
    """
    dirs = [int(d) for d in directions]

    def criteria(f):
        terms = []
        for factor_obj, d in zip(FACTOR_OBJECTS, dirs):
            if d == 1:
                terms.append(factor_obj(ascending))
            elif d == -1:
                terms.append(factor_obj(descending))
        if not terms:
            raise ValueError("All factor directions are 0 — no ranking criteria.")
        f.Ranking(sum(terms), top=TOP)
        return f

    return criteria


def strategy_id(row, source_file):
    """Human-readable identifier for a strategy row."""
    fname = os.path.basename(source_file).replace(".csv", "")
    return f"{fname}_rank{int(row['rank'])}"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run QRUMBLE out-of-sample validation for strategies in CSV files."
    )
    parser.add_argument(
        "files", nargs="+",
        help="Path(s) to top-N CSV files (e.g. top6_SOM.csv top6_GP.csv)"
    )
    args = parser.parse_args()

    tee = TeeLogger(LOG_TXT)
    sys.stdout = tee

    print("=" * 70)
    print("  QRUMBLE Out-of-Sample Validation Runner")
    print(f"  Started : {TIMESTAMP}")
    print(f"  Period  : {OOS_START}  duration={OOS_DURATION}  rebalance={REBALANCE}")
    print(f"  Top     : {TOP}  |  RF={RF}")
    print(f"  Log     : {LOG_TXT}")
    print("=" * 70)

    q.logging(logging.BASIC)

    print(f"\nLoading universe  : {UNIVERSE}")
    univ  = q.universe.load(UNIVERSE, index="stoxx600")
    print(f"Loading fundamentals: {FUNDAMENTALS}")
    funda = q.load(FUNDAMENTALS)
    print(f"Loading OHLCV     : {OHLCV}")
    ohlcv = q.load(OHLCV)
    print("Data loaded.\n")

    results = []

    for fpath in args.files:
        if not os.path.exists(fpath):
            print(f"\nWARNING: File not found, skipping: {fpath}")
            continue

        df_strats = pd.read_csv(fpath)
        method    = os.path.basename(fpath).replace(".csv", "")
        print(f"\n{'─' * 70}")
        print(f"  File   : {fpath}")
        print(f"  Method : {method}  |  Strategies: {len(df_strats)}")
        print(f"{'─' * 70}")

        missing = [c for c in FACTOR_COLS if c not in df_strats.columns]
        if missing:
            print(f"  WARNING: Missing factor columns {missing} — skipping file.")
            continue

        for _, row in df_strats.iterrows():
            sid       = strategy_id(row, fpath)
            dirs      = [row[c] for c in FACTOR_COLS]
            criterion = f"selected_for={row.get('selected_for', '?')}"

            print(f"\n  [{sid}]  {criterion}")
            print(f"  Directions : {[int(d) for d in dirs]}")
            print(f"  Train      : sharpe={row.get('sharpe', '?')}  "
                  f"return={row.get('annualized', '?')}  "
                  f"var={row.get('var', '?')}")

            try:
                criteria_fn = make_criteria(dirs)

                ranking_layout = f"{sid}:{{DATE}}/rank/{TOP}"
                investm_layout = f"{sid}:{{DATE}}/play/{TOP}/1y"

                # Re-register the universe before each q.qrumble() call,
                # exactly as done in 1_RandomSearchProb.py (MFRandom.call).
                q.universe.load(**univ)

                perf = q.qrumble(
                    ranking_layout, investm_layout,
                    OOS_START, OOS_DURATION,
                    rebalance=REBALANCE,
                    fundamentals=funda,
                    ohlcv=ohlcv,
                    Rf=RF,
                    universe_list=q.universe.fetch,
                    universe=lambda f: f.Screening(
                        ~Sector().isin(FINANCIALS + UTILITIES)
                    ),
                    criteria=criteria_fn,
                )

                print(f"  Saved: OOS result: {perf}")

                results.append({
                    "method":          method,
                    "strategy_id":     sid,
                    "selected_for":    row.get("selected_for", ""),
                    "rank":            int(row["rank"]),
                    **{c: int(row[c]) for c in FACTOR_COLS},
                    "train_sharpe":    row.get("sharpe",     None),
                    "train_return":    row.get("annualized", None),
                    "train_var":       row.get("var",        None),
                    "val_sharpe":      perf.get("sharpe",     None),
                    "val_return":      perf.get("annualized", None),
                    "val_var":         perf.get("var",        None),
                    "val_roi":         perf.get("roi",        None),
                    "val_mean":        perf.get("mean",       None),
                    "val_std":         perf.get("std",        None),
                    "val_alpha":       perf.get("alpha",      None),
                    "val_beta":        perf.get("beta",       None),
                    "val_tvar":        perf.get("tvar",       None),
                })

            except Exception as e:
                print(f"  Error: {e}")
                traceback.print_exc()
                results.append({
                    "method":       method,
                    "strategy_id":  sid,
                    "selected_for": row.get("selected_for", ""),
                    "rank":         int(row["rank"]),
                    **{c: int(row[c]) for c in FACTOR_COLS},
                    "train_sharpe": row.get("sharpe",     None),
                    "train_return": row.get("annualized", None),
                    "train_var":    row.get("var",        None),
                    "error":        str(e),
                })

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(RESULTS_CSV, index=False)
        print(f"\n{'=' * 70}")
        print(f"  Saved: Results saved -> {RESULTS_CSV}")
        print(f"  Saved: Log saved     -> {LOG_TXT}")
        print(f"  Total strategies run: {len(results)}")

        val_cols = ["strategy_id", "selected_for",
                    "train_sharpe", "val_sharpe",
                    "train_return", "val_return",
                    "train_var",    "val_var"]
        summary_cols = [c for c in val_cols if c in df_out.columns]
        print(f"\n  Summary:\n{df_out[summary_cols].to_string(index=False)}")
        print("=" * 70)
    else:
        print("\nWARNING: No strategies were run.")

    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
