# converter_csv_to_data.py — Convert a QRUMBLE CSV to a YAML-header .data file
#
# Accepts either a raw QRUMBLE output CSV (float metric values) or an already-normalised
# CSV (integer values in [0, 1000]). Auto-detects which format is given, normalises if
# needed, renames diff columns to Greek-delta names, and writes the YAML-header .data
# file expected by MasterSOM.java. Default paths are set at the top of the script;
# override them via CLI arguments.
#
# CLI    : conda run -n Q python converter_csv_to_data.py \
#              <input_csv> <output_data> [round_num] [n_samples]
# DO NOT use python3 — QRUMBLE is only available in the conda env named Q

import os
import sys
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "som")

INPUT_CSV   = os.path.join(_BASE, "round2_seed998_500_normalized.csv")
OUTPUT_DATA = os.path.join(_BASE, "round2_seed998_500.data")

ROUND   = 2
SAMPLES = 500

if len(sys.argv) >= 3:
    INPUT_CSV   = sys.argv[1]
    OUTPUT_DATA = sys.argv[2]
if len(sys.argv) >= 4:
    ROUND   = int(sys.argv[3])
if len(sys.argv) >= 5:
    SAMPLES = int(sys.argv[4])

# top comes first (fixed at 30), then 11 factors, then 8 metrics.
# inputs=20, outputs=0 — Java treats all columns as input dimensions.
VARIABLE_NAMES = [
    "top",
    "Yield",
    "ROC",
    "EarningsYield",
    "RS(6m)",
    "ROA",
    "ΔROA",
    "AccrualRatio",
    "ΔLTDebt-to-Assets",
    "ΔCurrentRatio",
    "ΔOpMgn",
    "ΔAssetTurnover",
    "annualized",
    "mean",
    "std",
    "sharpe",
    "alpha",
    "beta",
    "var",
    "tvar",
]

N_INPUTS  = 20
N_OUTPUTS = 0

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

# ── Inlined helpers ───────────────────────────────────────────────────────────

def normalize_metrics(df, intervals):
    return (
        df.iloc[:, -12:]
        .drop(columns=["roi", "alpha_daily", "var_daily", "tvar_daily"])
        .apply(lambda col: np.clip(
            1000 * (col - intervals[col.name][0]) /
            (intervals[col.name][1] - intervals[col.name][0]),
            0, 1000
        ))
        .astype(int)
    )


def write_data_file(data, output_path, variable_names, n_inputs, n_outputs, round_num, n_samples=None):
    assert len(variable_names) == n_inputs + n_outputs
    assert data.shape[1] == len(variable_names)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    header  = "---\n"
    header += f'name: "SOM Round {round_num}'
    if n_samples is not None:
        header += f" — {n_samples} samples"
    header += '"\n'
    header += f"description: Factor-based strategy search Round {round_num}\n"
    header += "variables:\n"
    for name in variable_names:
        header += f"    - {name}\n"
    header += f"inputs: {n_inputs}\n"
    header += f"outputs: {n_outputs}\n"
    header += "---\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for row in data.values:
            f.write(",".join(map(str, row)) + "\n")
    return output_path

# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_raw_df(df):
    metrics_norm = normalize_metrics(df, INTERVALS)
    non_metric   = df.iloc[:, :-12].copy()
    factor_only  = (
        non_metric
        .drop(columns=["top"], errors="ignore")
        .replace({-1: 0, 0: 500, 1: 1000})
    )
    if "top" in non_metric.columns:
        top_col = non_metric[["top"]]
        return pd.concat([top_col, factor_only, metrics_norm], axis=1)
    else:
        top_col = pd.DataFrame({"top": [30] * len(df)})
        return pd.concat([top_col, factor_only, metrics_norm], axis=1)


def _print_data_summary(data, output_path, n_inputs, n_outputs):
    print(f"\nWritten : {output_path}")
    print(f"  Rows   : {len(data)}")
    print(f"  Columns: {data.shape[1]}  (inputs={n_inputs}, outputs={n_outputs})")
    print(f"\nFirst 3 data rows:")
    for row in data.values[:3]:
        print("  " + ",".join(map(str, row)))

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Shape  : {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Raw CSVs have float metric values (e.g. sharpe=0.87);
    # normalised CSVs have integer values in [0,1000].
    last_col_max = df.iloc[:, -1].max()

    if last_col_max > 1000:
        print("\nDetected: RAW CSV — normalizing...")
        data = normalize_raw_df(df)
    else:
        print("\nDetected: already normalized CSV.")
        if "top" not in df.columns:
            print("  Adding top=30 column...")
            top_col = pd.DataFrame({"top": [30] * len(df)})
            data = pd.concat([top_col, df], axis=1)
        else:
            cols = ["top"] + [c for c in df.columns if c != "top"]
            data = df[cols]

    rename_map = {
        "ROA_diff":              "ΔROA",
        "LTDebt_Assets_diff":    "ΔLTDebt-to-Assets",
        "CurrentRatio_diff":     "ΔCurrentRatio",
        "OpMgn_diff":            "ΔOpMgn",
        "AssetTurnover_diff":    "ΔAssetTurnover",
        "RS_6m":                 "RS(6m)",
    }
    data = data.rename(columns=rename_map)

    missing = [c for c in VARIABLE_NAMES if c not in data.columns]
    if missing:
        print(f"\nFilling {len(missing)} missing factor cols with 500: {missing}")
        for col in missing:
            data[col] = 30 if col == "top" else 500

    if not all(c in data.columns for c in VARIABLE_NAMES):
        print(f"\nERROR: cannot match all expected columns.")
        print(f"Data columns   : {list(data.columns)}")
        print(f"Expected       : {VARIABLE_NAMES}")
    else:
        data = data[VARIABLE_NAMES]
        write_data_file(
            data           = data,
            output_path    = OUTPUT_DATA,
            variable_names = VARIABLE_NAMES,
            n_inputs       = N_INPUTS,
            n_outputs      = N_OUTPUTS,
            round_num      = ROUND,
            n_samples      = SAMPLES,
        )
        _print_data_summary(data, OUTPUT_DATA, N_INPUTS, N_OUTPUTS)
