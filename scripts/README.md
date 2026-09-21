# scripts

> **Peran:** urutan pipeline dan keluaran tiap tahap.
> **Audiens:** publik.
> **Bukan untuk:** definisi fitur (lihat `data/README.md`) atau dasar keputusan
> (lihat `docs/keputusan_desain.md`).

Pipeline bernomor, dijalankan berurutan:

| Skrip | Status | Keluaran |
|---|---|---|
| `01_fetch_data.py` | selesai | `data/raw/` - harga `.JK`, `macro/jisdor.csv`, `macro/bi_7drrr.csv`, `benchmark_ihsg.csv`, `metadata.json` |
| `02_build_features.py` | selesai | `data/interim/prices_clean/`, `data/processed/features/` (8 fitur + target) |
| `03_train_predict.py` | selesai | `experiments/<run_id>/` - `predictions.csv`, `fold_metrics.csv`, `run_info.json`, `checkpoints/`, `pretrained/` (mode pretrain) |
| `04_optimize.py` | selesai | Grid penuh 1.728 konfigurasi di `experiments/optimize_20260918_050942/` |
| `05_evaluate.py` | selesai | `reports/` - `metrics_table.csv`, `regime_table.csv`, dsb. |

Catatan: `01_fetch_data.py` membutuhkan akses jaringan ke Yahoo Finance dan Bank Indonesia. `02_build_features.py` tidak memerlukan jaringan.

## 03_train_predict.py

Latih CNN-BiLSTM bersama untuk 43 saham lalu tulis prediksi walk-forward. **Baru:** mode `pretrain` (MAE self-supervised) dan `walk-forward-pretrain` (pre-train → fine-tune).

Spesifikasi arsitektur dan split ada di `configs/model.yaml` dan `configs/split.yaml`; dasar tiap nilai ada di `docs/keputusan_desain.md`.

```
python3 scripts/03_train_predict.py --mode dry-run        # tinjau tabel 65 lipatan
python3 scripts/03_train_predict.py --mode smoke          # 2 lipatan, 1 seed, 3 epoch
python3 scripts/03_train_predict.py --mode calibrate      # ukur laju fit terkecil/terbesar
python3 scripts/03_train_predict.py --mode tune           # grid + sensitivitas periode desain
python3 scripts/03_train_predict.py --mode pretrain       # BARU: MAE pre-training (self-supervised)
python3 scripts/03_train_predict.py --mode walk-forward-pretrain  # BARU: pre-train + fine-tune
python3 scripts/03_train_predict.py                       # walk-forward penuh (baseline)
```

Opsi penting: `--seeds 0,1,2,3,4`, `--jobs 4` (paralel; RAM 7 GiB membatasi maksimal 4), `--resume` (lewati pasangan lipatan-seed yang selesai), `--use-tuning <direktori>` (bekukan hyperparameter hasil tuning).

### Mode Baru: Pre-training

```bash
# 1. Pre-training MAE (self-supervised, tanpa label return)
python3 scripts/03_train_predict.py --mode pretrain --epochs 50 --jobs 4 --seeds 0,1,2,3,4
# Output: experiments/pretrain_<timestamp>/pretrained/encoder_seed{0-4}.pt

# 2. Walk-forward dengan pre-trained encoder → fine-tune
python3 scripts/03_train_predict.py --mode walk-forward-pretrain --jobs 1 --seeds 0,1,2,3,4 --resume
```

Kontrak keluaran untuk tahap 4: `predictions.csv` berisi satu baris per (tanggal, saham, seed) dengan kolom `date, ticker, seed, role, pred_scaled, true_scaled, pred_raw, true_raw`. Prediksi mentah (`pred_raw`) dipakai untuk peringkat top-k; `role=test` dipakai evaluasi utama tahap 5.

## 04_optimize.py

Peringkat ensemble top-k lalu backtest Mean-Variance bulanan dengan biaya IDX. Peringkat memakai rata-rata `pred_raw` lintas seed per (tanggal, saham); kovarians ditaksir dari L imbal hasil log harian sampai tanggal rebalancing (tanpa melihat masa depan). Optimasi Mean-Variance meminimalkan varians dengan kendala target return rata-rata pred_ens (Chaweewanchon & Chaysiri 2022 Section 3.1).

```
python3 scripts/04_optimize.py --predictions experiments/<run_id>/predictions.csv --grid utama --seed-mode ensemble
python3 scripts/04_optimize.py --predictions experiments/<run_id>/predictions.csv
```

Grid bawaan (`penuh`): k {5,7,10,3,15,20}, estimator sample, ridge_epsilon, ledoit_wolf, gmv, L {120,60,252}, maxw {0.35,0.25,0.50,1.0}, peringkat ensemble dan per seed. Optimizer type: target_return (MV) saja. Opsi `--grid utama` menjalankan hanya k {5,7,10}, L 120, maxw 0.35. Modal awal bawaan 100 juta rupiah.

## 05_evaluate.py

Metrik, signifikansi, dan ketahanan rezim dari keluaran tahap 4.

```
python3 scripts/05_evaluate.py --portfolio <opt>/portfolio_returns.csv --baselines <opt>/baseline_returns.csv --weights <opt>/weights.csv
```

Risk-free harian = BI-7DRRR tahunan dibagi 252; annualisasi 252. `turnover_adjusted_sharpe` = Sharpe dikali (1 - turnover) dengan turnover didefinisikan sebagai rata-rata |selisih bobot|/2 per rebalancing.