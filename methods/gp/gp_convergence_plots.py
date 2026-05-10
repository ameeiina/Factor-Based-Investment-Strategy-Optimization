# gp_convergence_plots.py — Regenerate GP convergence figures from saved per-seed CSVs
#
# Reads the per-seed evaluation CSVs written by gp_optimize.py and produces two
# 1×3 subplot figures (one for J(w), one for Sharpe), one panel per configuration.
# These are Figures 6.4 and 6.5 in the thesis.
#
# CLI    : conda run -n Q python gp_convergence_plots.py
# Input  : outputs/gp/gp_nc200_ni*_EI_GP_s*.csv  outputs/gp/gp_nc200_ni*_LCB_GP_s*.csv
# Output : outputs/gp/gp_convergence_J_3subplots.png
#          outputs/gp/gp_convergence_sharpe_3subplots.png
# DO NOT use python3 — QRUMBLE is only available in the conda env named Q

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GP_DIR   = os.path.join(BASE_DIR, 'outputs', 'gp')
OUT_DIR  = GP_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ── Configuration matrix ──────────────────────────────────────────────────────

CONFIGS = [
    {'name': 'EI, n_init=50',  'acq': 'EI',  'ni': 50},
    {'name': 'LCB, n_init=50', 'acq': 'LCB', 'ni': 50},
    {'name': 'EI, n_init=100', 'acq': 'EI',  'ni': 100},
]

SEEDS        = [42, 123, 999]
COLORS       = {42: '#1f77b4', 123: '#ff7f0e', 999: '#2ca02c'}
SEED_LABELS  = {42: 'seed=42', 123: 'seed=123', 999: 'seed=999'}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(acq, ni):
    data = {}
    for seed in SEEDS:
        path = os.path.join(GP_DIR, f'gp_nc200_ni{ni}_{acq}_GP_s{seed}.csv')
        if os.path.exists(path):
            data[seed] = pd.read_csv(path)
        else:
            print(f"  Missing: {os.path.basename(path)}")
    return data

# ── J(w) convergence ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('GP Convergence — Running Best J(w)', fontsize=13, weight='bold', y=1.02)

for ax, cfg in zip(axes, CONFIGS):
    for seed, df in sorted(load_config(cfg['acq'], cfg['ni']).items()):
        df = df.sort_values('call_number').reset_index(drop=True)
        ax.plot(df['call_number'], df['J'].cummin(),
                color=COLORS[seed], linewidth=2, label=SEED_LABELS[seed])
    ax.set_title(cfg['name'], fontsize=11, weight='bold')
    ax.set_xlabel('Evaluation call', fontsize=10)
    ax.set_ylabel('Best J(w) so far', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'gp_convergence_J_3subplots.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── Sharpe convergence ────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('GP Convergence — Running Best Sharpe', fontsize=13, weight='bold', y=1.02)

for ax, cfg in zip(axes, CONFIGS):
    for seed, df in sorted(load_config(cfg['acq'], cfg['ni']).items()):
        df = df.sort_values('call_number').reset_index(drop=True)
        ax.plot(df['call_number'], df['sharpe'].cummax(),
                color=COLORS[seed], linewidth=2, label=SEED_LABELS[seed])
    ax.set_title(cfg['name'], fontsize=11, weight='bold')
    ax.set_xlabel('Evaluation call', fontsize=10)
    ax.set_ylabel('Best Sharpe so far', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'gp_convergence_sharpe_3subplots.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
