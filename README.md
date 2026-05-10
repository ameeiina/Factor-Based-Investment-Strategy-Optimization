# Factor-Based Investment Strategy Optimization

Source code for the master thesis *Self-Organizing Maps, Gaussian Processes and Genetic Algorithms Applied to Factor-Based Investment Strategy Optimization*, TU Berlin, 2026.

Three optimization methods search for the best combination of 11 fundamental and momentum factors applied to STOXX 600 stocks (2015–2019 training, 2020–2023 out-of-sample). Strategies are evaluated using the QRUMBLE backtesting framework.

---

## Repository structure

```
data/                          # QRUMBLE input data (universe, fundamentals, OHLCV)
methods/
  som/                         # Self-Organizing Map guided random search
    random_search.py           # Main script — run once per round, edit PFACTORS manually
    visualize.py               # Heatmap visualisation of SOM prototype output
    helper_files/              # CSV→.data converter, multi-file combiner
    UbiSOM/                    # UbiSOM Java library (brunomnsilva) + MasterSOM.java
    outputs/round{1,2,3}/      # Search results per round
    saved_heatmaps/            # Saved heatmap figures per round and seed
  gp/
    gp_optimize.py             # Bayesian optimisation (scikit-optimize gp_minimize)
    gp_convergence_plots.py    # Regenerate GP convergence subplot figures
    outputs/gp/                # Per-seed CSVs, convergence plots, recommendation
  moga/
    moga_optimize.py           # NSGA-II multi-objective optimisation (pymoo)
    moga_validate.py           # Pareto-front validation on 2019-2020 holdout
    outputs/                   # Per-config CSVs, Pareto fronts, convergence plots
analysis/
  run_validation.py            # True out-of-sample validation (2020-2023) — Table 7.4
  data/                        # Top-6 strategy CSVs per method
  outputs/                     # Validation results CSV + log, cross-method figures
requirements.txt
```

---

## Setup

QRUMBLE is a proprietary backtesting framework and must be installed separately into a conda environment named `Q`. All scripts must be run via:

```bash
conda run -n Q python <script>.py
```

Install open-source dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the methods

### SOM — human-in-the-loop random search

```bash
cd methods/som
# Round 1 — uniform exploration
conda run -n Q python random_search.py   # ROUND=1, all PFACTORS=eq

# Inspect heatmaps: run MasterSOM.java on outputs/round1/combined_*.data
# then: conda run -n Q python visualize.py <prototypes_csv>

# Round 2 — update PFACTORS in random_search.py based on heatmap inspection
# Set ROUND=2, SAMPLES=1500, edit PFACTORS, then:
conda run -n Q python random_search.py

# Round 3 — repeat for ROUND=3, SAMPLES=500
```

### Gaussian Process

```bash
cd methods/gp
conda run -n Q python gp_optimize.py
```

### NSGA-II (MOGA)

```bash
cd methods/moga
conda run -n Q python moga_optimize.py
conda run -n Q python moga_validate.py
```

### Out-of-sample validation

```bash
cd analysis
conda run -n Q python run_validation.py data/top6_SOM.csv data/top6_GP.csv data/top6_MOGA.csv
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Data handling |
| `matplotlib` | Visualisation |
| `scikit-optimize` | GP surrogate optimisation |
| `pymoo` | NSGA-II multi-objective optimisation |
| `qrumble` | Backtesting (proprietary, not in requirements.txt) |

Java 11+ and Maven are required to build and run UbiSOM (`methods/som/UbiSOM/`).
