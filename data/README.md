# Data

This directory contains STOXX Europe 600 market data used in the thesis.

## Included files (small, tracked by git)

| File | Description |
|------|-------------|
| `EXSA.DE.csv` | iShares STOXX Europe 600 ETF price data |
| `OSX6.DE.csv` | Lyxor STOXX Europe 600 ETF price data |
| `ECB_Data_Portal_20240707200042.csv` | ECB risk-free rate data |
| `get-risk-free-rates.sh` | Script to fetch ECB risk-free rates |

## Excluded files (large, NOT tracked by git)

The `.pickle` files are excluded via `.gitignore` due to size. They are produced
by the QRUMBLE data pipeline. Contact the authors or re-generate them using QRUMBLE.

| File | Period | Contents |
|------|--------|----------|
| `UNIVERSE.2015_2019.pickle` | 2015–2019 | Stock universe (training) |
| `UNIVERSE.2019_2020.pickle` | 2019–2020 | Stock universe (validation) |
| `UNIVERSE.2020_2023.pickle` | 2020–2023 | Stock universe (out-of-sample) |
| `FUNDAMENTALS.2015_2019.pickle` | 2015–2019 | Fundamental factors (training) |
| `FUNDAMENTALS.2019_2020.pickle` | 2019–2020 | Fundamental factors (validation) |
| `FUNDAMENTALS.2020_2023.pickle` | 2020–2023 | Fundamental factors (out-of-sample) |
| `stoxx600.jan2014_dec2019.pickle` | 2014–2019 | OHLCV prices (training) |
| `stoxx600.jan2018_apr2021.pickle` | 2018–2021 | OHLCV prices (validation) |
| `stoxx600.jan2019_jun2024.pickle` | 2019–2024 | OHLCV prices (out-of-sample) |
