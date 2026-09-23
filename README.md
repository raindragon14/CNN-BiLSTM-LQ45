# NeuralAlpha

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

> **Production-grade two-stage portfolio optimization for LQ45 (Indonesian equities).**
> **Self-supervised MAE pre-training → CNN-BiLSTM walk-forward preselection → Mean-Variance optimization with realistic IDX transaction costs.**

## About

**NeuralAlpha** is a deep learning research project that builds **neural network systems** to optimize investment portfolios. It combines three cutting-edge AI techniques:

- **Self-supervised learning** — A Masked Autoencoder (MAE) pre-trains on the same 8-feature market panel the supervised model consumes (price, volume, technical indicators, BI-7DRRR, JISDOR) without return labels, so the encoder transfers unchanged; pre-training data is capped at the design period (`pretrain.data_end`) so nothing from the OOS window leaks in
- **Deep sequence modeling** — A CNN-BiLSTM architecture with walk-forward validation (65 expanding windows, purge + embargo) predicts 5-day log-returns; 5-seed ensemble reduces variance
- **Mathematical optimization** — Top-k stocks are selected by ensemble predictions, then Mean-Variance optimization allocates weights under realistic Indonesian market constraints (buy 0.19%, sell 0.29%, lot 100)

The system evaluates rigorously with deflated Sharpe ratio (DSR), probability of backtest overfitting (PBO), Romano-Wolf multiple testing, and regime robustness across COVID-19 and rate-hike cycles.

---

## At a Glance

| Aspect | Details |
|--------|---------|
| **Universe** | 43/45 LQ45 constituents (Aug 2019–Jan 2020 composition) |
| **Period** | 2018–2025 (8 years, includes COVID-19, rate hike cycles) |
| **Architecture** | CNN (32→64) + BiLSTM (64×2) with MAE self-supervised pre-training on the 8-feature panel (`FEATURE_COLUMNS`) |
| **Training** | Walk-forward expanding window (65 folds), purge/embargo, 5 seeds, discriminative LR fine-tuning |
| **Optimization** | Mean-Variance (target-return), 4 covariance estimators, monthly rebalancing (21 days) |
| **Costs** | Buy 0.19%, Sell 0.29%, Lot 100 shares (IDX retail rules) |
| **Evaluation** | Sharpe (LW-HAC), Sortino, Calmar, DSR, PBO/CSCV, Romano-Wolf, 3 regime tests |

---

## Method Overview

1. **Data Pipeline**: Daily OHLCV from Yahoo Finance (`.JK`), BI-7DRRR (policy rate), JISDOR (USD/IDR) from Bank Indonesia, IHSG benchmark
2. **Feature Engineering**: 8 features per stock — price, volume, RSI(14), CCI(20), CMO(14), MFI(14), BI-7DRRR (level), JISDOR (log-return); winsorization + min-max scaling fit **only on training windows**
3. **Self-Supervised Pre-training (MAE)**: Masked Autoencoder on the same 8-feature panel, 30% masking, reconstruction loss on masked positions only — learns universal market representations without return labels (data capped at `pretrain.data_end` = 2019-12-31)
4. **Walk-Forward Fine-tuning**: CNN-BiLSTM predicts 5-day log-returns, nested walk-forward CV with purge (τ=5) and embargo (w-1=59), expanding training window, 5-seed ensemble
5. **Top-k Preselection**: Ensemble average of 5 seeds ranks stocks; select top-k (k=5,7,10 + sensitivity 3,15,20)
6. **Mean-Variance Optimization**: Minimize variance s.t. target return = ensemble mean prediction; covariance estimators: sample, ridge, Ledoit-Wolf, GMV; max weight 35% (sensitivity 25/50/100%)
7. **Rigorous Evaluation**: Out-of-sample walk-forward metrics, deflated Sharpe (LW-HAC), multiple testing correction (Romano-Wolf), probability of backtest overfitting (PBO), regime robustness (COVID 2020, Recovery/Rate Hikes 2021–2025)

---

## Quick Start

### Prerequisites
- Python 3.11+ (tested on 3.14.7)
- PyTorch 2.14+ (CPU)
- 7 GB RAM minimum (limits parallel jobs to 4)

```bash
# Clone and setup
git clone https://github.com/raindragon14/NeuralAlpha
cd NeuralAlpha
python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -e ".[dev]"
```

### Full Pipeline

```bash
# 1. Fetch raw data (requires internet: Yahoo Finance + Bank Indonesia)
python scripts/01_fetch_data.py
# Output: data/raw/prices/*.csv, data/raw/macro/*.csv, data/raw/benchmark_ihsg.csv

# 2. Build features & targets (offline)
python scripts/02_build_features.py
# Output: data/interim/prices_clean/, data/processed/features/

# 3. Self-supervised MAE pre-training (NEW)
python scripts/03_train_predict.py --mode pretrain --epochs 50 --jobs 4 --seeds 0,1,2,3,4
# Output: experiments/pretrain_<timestamp>/pretrained/encoder_seed{0-4}.pt

# 4. Walk-forward with pre-trained encoder → fine-tune (two-stage)
python scripts/03_train_predict.py --mode walk-forward-pretrain --jobs 1 --seeds 0,1,2,3,4 \
    --pretrained-dir experiments/pretrain_<timestamp>/pretrained --resume
# Or baseline without pre-training:
python scripts/03_train_predict.py --mode walk-forward --jobs 4 --seeds 0,1,2,3,4

# 5. Portfolio optimization (full grid: 1,728 configs)
python scripts/04_optimize.py --predictions experiments/<run_id>/predictions.csv
# Quick grid (12 configs): --grid main --seed-mode ensemble

# 6. Evaluation & regime analysis
python scripts/05_evaluate.py --portfolio <opt>/portfolio_returns.csv --baselines <opt>/baseline_returns.csv --weights <opt>/weights.csv
# Output: reports/metrics_table.csv, regime_table.csv, significance.csv, dsr_pbo.json, figures/
```

### Training Modes (`03_train_predict.py`)

```bash
python scripts/03_train_predict.py --mode dry-run        # Inspect 65-fold table without training
python scripts/03_train_predict.py --mode smoke          # 2 folds, 1 seed, 3 epochs (sanity check)
python scripts/03_train_predict.py --mode calibrate      # Measure sec/epoch on first/last fold
python scripts/03_train_predict.py --mode tune           # Hyperparameter grid + sensitivity on design period
python scripts/03_train_predict.py --mode pretrain       # MAE self-supervised pre-training
python scripts/03_train_predict.py --mode walk-forward-pretrain  # Pre-train → fine-tune (two-stage)
python scripts/03_train_predict.py                       # Walk-forward baseline (supervised only)
```

Key options: `--seeds 0,1,2,3,4`, `--jobs 4` (max 4, RAM 7 GiB), `--threads`, `--folds`, `--epochs`, `--resume`, `--use-tuning <dir>`, `--no-checkpoints`

---

## Repository Structure

```
configs/          YAML configs (model, split, portfolio, data, experiment, universe)
data/             raw/, interim/, processed/ (gitignored — see data/README.md)
docs/             keputusan_desain.md (decision log with citations)
scripts/          Numbered pipeline 01_fetch → 05_evaluate
src/lq45/    Package: data, features, models, portfolio, evaluation, utils
tests/            30 unit tests (pytest)
```

---

## Current Status (2026-09-22)

| Stage | Status | Experiment ID |
|-------|--------|---------------|
| Data pipeline (01–02) | ✅ Complete | — |
| Hyperparameter tuning | ✅ Complete | `tune_20260917_062213` (best: batch=64, wd=1e-4, pooling=2, units=64) |
| Walk-forward baseline | ✅ Complete | `walk-forward_20260917_081400` (65 folds × 5 seeds) |
| MAE pre-training | ✅ Fixed, re-run pending | Masking + scaling bugs fixed; regenerate encoders |
| Walk-forward + pre-train | ⏳ Pending | Requires regenerated pre-trained encoders |
| Full grid optimization | ✅ Baseline done | Main grid (12): `optimize_20260918_162336`; full grid (1,728): `optimize_20260918_050942` (baseline); pre-train grid queued |
| Evaluation & regime analysis | ⏳ Pending | Requires full grid outputs |

> **Note**: Correctness fixes (2026-09-22) changed MAE pre-training (per-sample masking), pre-trained
> encoder resolution, MV optimizer determinism, and Romano-Wolf. Correctness fixes (2026-09-23) aligned
> MAE pre-training to the 8-feature panel (encoder transfer), capped pre-training data at the design
> period (`pretrain.data_end`), restored the Jan-Apr 2020 OOS window, corrected DSR/Romano-Wolf/Sortino
> formulas, added the Ledoit-Wolf (2008) Sharpe test, a 1-day execution lag and fee-paying 1/N
> baselines, and removed `turnover_adjusted_sharpe`. All prior `experiments/pretrain_*` outputs are
> invalid. Results shown are from intermediate runs; final journal submission pending full grid
> evaluation on both baseline and pre-trained prediction sets.

---

## Key Results (Placeholders — To Be Filled After Full Grid)

| Metric | Baseline (Supervised) | NeuralAlpha (Pre-trained) | IHSG Buy-Hold | 1/N Equal Weight |
|--------|----------------------|---------------------------|---------------|------------------|
| **Annualized Return** | — | — | — | — |
| **Annualized Volatility** | — | — | — | — |
| **Sharpe Ratio (LW-HAC)** | — | — | — | — |
| **Sortino Ratio** | — | — | — | — |
| **Calmar Ratio** | — | — | — | — |
| **Max Drawdown** | — | — | — | — |
| **Turnover (mean per rebalance)** | — | — | — | — |
| **Deflated Sharpe (DSR)** | — | — | — | — |
| **PBO (Grid Utama)** | — | — | — | — |

| Regime | Baseline Sharpe | NeuralAlpha Sharpe | IHSG Sharpe |
|--------|----------------|-------------------|-------------|
| **COVID 2020 (OOS)** | — | — | — |
| **Recovery + Rate Hikes 2021–2025 (OOS)** | — | — | — |

| Statistical Test | Result |
|------------------|--------|
| Romano-Wolf (vs IHSG) | — |
| Romano-Wolf (vs 1/N) | — |
| Sharpe Difference (LW-HAC, vs IHSG) | — |

---

## Documentation

- `data/README.md` — Data sources, feature definitions, limitations
- `docs/keputusan_desain.md` — Every decision traced to literature + implementation location
- `scripts/README.md` — Pipeline contracts & I/O specs
- `ABOUT.md` — Portfolio showcase (problem, method, tech stack, results)

---

## Citation

If you use this code or findings, please cite:

```bibtex
@software{neuralalpha,
  author       = {Pandanarang, Reihan},
  title        = {NeuralAlpha: Two-Stage Portfolio Optimization with Self-Supervised MAE Pre-training and CNN-BiLSTM Preselection on LQ45},
  year         = {2026},
  url          = {https://github.com/raindragon14/NeuralAlpha},
  version      = {0.1.0},
  license      = {MIT}
}
```

Full citation metadata: [`CITATION.cff`](CITATION.cff).

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.