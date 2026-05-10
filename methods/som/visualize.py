# visualize.py — Heatmap visualisation of UbiSOM prototype CSV output
#
# Reads the prototypes CSV written by MasterSOM.java and saves four figure files:
#   umat_cnt.png      — U-matrix (cluster boundaries) and hit map (data density)
#   factor_planes.png — one heatmap per factor on a fixed [0, 1000] colour scale
#                       (cool=-1 descending, mid=0 inactive, warm=+1 ascending)
#   metric_planes.png — one heatmap per performance metric (auto-scaled)
#   all_planes.png    — all planes combined in a single grid figure
# Used between search rounds to inspect the SOM and decide pfactors for the next round.
#
# CLI    : conda run -n Q python visualize.py <prototypes_csv> [--save-dir <dir>]
# Output : saved_heatmaps/  (or --save-dir)
# DO NOT use python3 — QRUMBLE is only available in the conda env named Q

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────

SOM_COLS  = 20
SOM_LINES = 10

FACTOR_COLS = [
    "Yield", "ROC", "EarningsYield", "RS(6m)",
    "ROA", "ΔROA", "AccrualRatio",
    "ΔLTDebt-to-Assets", "ΔCurrentRatio",
    "ΔOpMgn", "ΔAssetTurnover",
]

METRIC_COLS = [
    "annualized", "mean", "std", "sharpe",
    "alpha", "beta", "var", "tvar",
]

# ── Prototype loading ─────────────────────────────────────────────────────────

def read_prototypes(path, cols, lines):
    df = pd.read_csv(path)
    assert len(df) == cols * lines, f"Expected {cols*lines} rows, got {len(df)}"
    return df

def to_grid(df, col, cols, lines):
    mat = np.zeros((lines, cols))
    for x in range(cols):
        for y in range(lines):
            mat[lines - y - 1, x] = df.iloc[x * lines + y][col]
    return mat

# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_plane(ax, mat, title, vmin=None, vmax=None, cmap="jet"):
    im = ax.imshow(mat, cmap=cmap, interpolation="nearest",
                   vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

# ── Save functions ────────────────────────────────────────────────────────────

def save_umat_cnt(df, cols, lines, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(axes,
                               ["UMAT", "CNT"],
                               ["U-Matrix", "Hit Map (CNT)"]):
        mat = to_grid(df, col, cols, lines)
        im  = plot_plane(ax, mat, title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("UbiSOM — U-Matrix and Hit Map", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umat_cnt.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_factor_planes(df, cols, lines, save_dir):
    n = len(FACTOR_COLS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    axes = axes.flatten()
    for i, col in enumerate(FACTOR_COLS):
        if col not in df.columns:
            axes[i].axis("off")
            continue
        mat = to_grid(df, col, cols, lines)
        im  = plot_plane(axes[i], mat, col, vmin=0, vmax=1000)
        cb  = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04,
                           ticks=[0, 500, 1000])
        cb.ax.set_yticklabels(["-1", "0", "+1"], fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Factor Direction Planes  (cool=-1 | mid=0 | warm=+1)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "factor_planes.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_metric_planes(df, cols, lines, save_dir):
    n = len(METRIC_COLS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
    axes = axes.flatten()
    for i, col in enumerate(METRIC_COLS):
        if col not in df.columns:
            axes[i].axis("off")
            continue
        mat = to_grid(df, col, cols, lines)
        im  = plot_plane(axes[i], mat, col)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Performance Metric Planes", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metric_planes.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_all_planes(df, cols, lines, save_dir):
    all_cols   = ["UMAT"] + FACTOR_COLS + METRIC_COLS + ["CNT"]
    all_titles = ["U-Matrix"] + FACTOR_COLS + METRIC_COLS + ["Hit Map"]
    ncols = 6
    nrows = (len(all_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.8))
    axes = axes.flatten()
    for i, (col, title) in enumerate(zip(all_cols, all_titles)):
        if col not in df.columns:
            axes[i].axis("off")
            continue
        mat  = to_grid(df, col, cols, lines)
        vmin = 0 if col in FACTOR_COLS else None
        vmax = 1000 if col in FACTOR_COLS else None
        im   = plot_plane(axes[i], mat, title, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    for j in range(len(all_cols), len(axes)):
        axes[j].axis("off")
    plt.suptitle("UbiSOM — All Component Planes", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_planes.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize UbiSOM prototype heatmaps.")
    parser.add_argument("prototypes_csv", help="Path to prototypes CSV from MasterSOM.java")
    parser.add_argument("--save-dir", default="saved_heatmaps",
                        help="Directory to save figures (default: saved_heatmaps/)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    df = read_prototypes(args.prototypes_csv, SOM_COLS, SOM_LINES)
    print(f"Loaded : {args.prototypes_csv}  ({len(df)} nodes)")
    print(f"Columns: {list(df.columns)}")

    save_umat_cnt(df, SOM_COLS, SOM_LINES, args.save_dir)
    save_factor_planes(df, SOM_COLS, SOM_LINES, args.save_dir)
    save_metric_planes(df, SOM_COLS, SOM_LINES, args.save_dir)
    save_all_planes(df, SOM_COLS, SOM_LINES, args.save_dir)

    print(f"Figures saved to {args.save_dir}/")
