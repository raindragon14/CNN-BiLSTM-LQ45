#!/usr/bin/env python3
"""Tahap 4: Peringkat Top-K dan Optimasi Mean-Variance.

Masukan: `predictions.csv` dari tahap 3 (kolom `role` menandai set uji).
Keluaran per jalankan (`experiments/<run_id>/`):
    - `weights.csv`            : bobot rebalancing per tanggal dan konfigurasi
    - `portfolio_returns.csv`  : ekuitas dan return harian bersih per konfigurasi
    - `baseline_returns.csv`   : pool 1/N dan IHSG buy-and-hold
    - `run_info.json`          : rekaman eksekusi untuk audit

Konfigurasi grid pencarian (lihat `configs/portfolio.yaml`):
    - k: nilai {5, 7, 10, 3, 15, 20}, L: {120, 60, 252},
      max_weight: {0.35, 0.25, 0.50, 1.0}, dan mode peringkat
      ensemble serta per seed. Opsi `--grid utama` untuk eksekusi
      cepat (k {5, 7, 10}, L 120, max_weight 0.35).

Metodologi:
    Penyaringan top-k memakai prediksi CNN-BiLSTM. Optimasi
    Mean-Variance meminimalkan varians dengan kendala target_return
    rata-rata pred_ens (Chaweewanchon & Chaysiri 2022, Section 3.1).
    GMV tidak digunakan.

Jejak keputusan: docs/keputusan_desain.md bagian E dan H.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.portfolio.backtest import jalankan_backtest
from lq45.portfolio.ranking import ensemble_prediksi
from lq45.utils.config import (
    EXPERIMENT_DIR,
    INTERIM_DIR,
    ensure_dirs,
    load_config,
)


def parse_args() -> argparse.Namespace:
    """Mengurai argumen baris perintah untuk konfigurasi optimasi.

    Returns:
        Namespace berisi parameter: jalur predictions, direktori
        keluaran, cakupan grid, modal awal, dan mode peringkat seed.
    """
    parser = argparse.ArgumentParser(
        description="Peringkat top-k dan optimasi portofolio."
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, help="predictions.csv tahap 3"
    )
    parser.add_argument("--out", type=Path, default=None, help="direktori keluaran")
    parser.add_argument(
        "--grid",
        choices=["utama", "penuh"],
        default="penuh",
        help="cakupan grid (bawaan: penuh)",
    )
    parser.add_argument(
        "--modal",
        type=float,
        default=100_000_000.0,
        help="modal awal rupiah (bawaan: 100 juta)",
    )
    parser.add_argument(
        "--seed-mode",
        choices=["ensemble", "semua"],
        default="semua",
        help="peringkat ensemble saja atau ditambah per seed",
    )
    return parser.parse_args()


def git_sha() -> str:
    """Mengambil hash commit pendek (7 karakter) dari repository akar.

    Returns:
        String hash SHA pendek, atau `"unknown"` jika Git tidak tersedia.
    """
    try:
        hasil = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return hasil.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def muat_harga() -> pd.DataFrame:
    """Memuat harga penutupan adjusted (Adj Close) seluruh saham LQ45.

    Setiap file CSV di direktori interim mewakili satu ticker.
    Indeks berupa tanggal, kolom berupa ticker dengan akhiran `.JK`.

    Returns:
        DataFrame berisi harga harian yang diurutkan berdasarkan tanggal.
    """
    bingkai: dict[str, pd.Series] = {}
    for berkas in sorted((INTERIM_DIR / "prices_clean").glob("*.csv")):
        ticker = berkas.stem + ".JK"
        frame = pd.read_csv(berkas, parse_dates=["Date"]).set_index("Date")
        bingkai[ticker] = frame["Adj Close"]
    harga = pd.DataFrame(bingkai).sort_index()
    harga.index = pd.to_datetime(harga.index)
    return harga


def kerangka_prediksi(path: Path) -> pd.DataFrame:
    """Memuat hasil prediksi tahap 3 dan menyaring baris set uji.

    Mengonversi kolom `date` ke string ISO 8601 dan memfilter hanya
    baris dengan `role == "test"` untuk evaluasi out-of-sample.

    Args:
        path: Jalur ke file `predictions.csv`.

    Returns:
        DataFrame berisi data uji yang telah direset index-nya.
    """
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame[frame["role"] == "test"].reset_index(drop=True)


def bingkai_peringkat(frame: pd.DataFrame, mode_seed: str) -> dict[str, pd.DataFrame]:
    """Membangun bingkai prediksi per mode ensemble dan per seed.

    Setiap mode dipetakan ke DataFrame dengan kolom `date`, `ticker`,
    dan `pred_ens`. Mode `ensemble` merata-ratakan seluruh seed;
    mode seed individual menghasilkan satu DataFrame untuk pelaporan
    sebaran prediksi.

    Args:
        frame: DataFrame hasil `kerangka_prediksi`.
        mode_seed: `"ensemble"` untuk rata-rata, `"semua"` untuk per seed.

    Returns:
        Kamus yang memetakan nama mode ke DataFrame peringkat.
    """
    keluar: dict[str, pd.DataFrame] = {}
    keluar["ensemble"] = ensemble_prediksi(frame)
    if mode_seed == "semua":
        for seed in sorted(frame["seed"].unique().tolist()):
            potong = frame[frame["seed"] == seed][["date", "ticker", "pred_raw"]]
            keluar[f"seed{seed}"] = potong.rename(columns={"pred_raw": "pred_ens"})
    return keluar


def garis_dasar(
    harga: pd.DataFrame, tanggal_oos: list[str], modal: float
) -> pd.DataFrame:
    """Menghitung imbal hasil harian pembanding 1/N pool penuh dan IHSG.

    1/N membagi modal merata ke seluruh ticker yang tersedia pada tiap
    tanggal. IHSG dinormalkan ke modal awal pada tanggal OOS pertama.

    Args:
        harga: DataFrame harga penutupan harian.
        tanggal_oos: Daftar tanggal out-of-sample.
        modal: Modal awal dalam Rupiah.

    Returns:
        DataFrame berisi ekuitas dan return harian kedua pembanding.
    """
    tanggal = pd.to_datetime(sorted(set(tanggal_oos)))
    harga_oos = harga[harga.index.isin(tanggal)].sort_index()
    imbal = harga_oos.pct_change().fillna(0.0)
    ret_1n = imbal.mean(axis=1)
    ekuitas_1n = (1.0 + ret_1n).cumprod() * modal
    ihsg = pd.read_csv(
        ROOT / "data" / "raw" / "benchmark_ihsg.csv", parse_dates=["Date"]
    ).set_index("Date")
    ihsg.index = pd.to_datetime(ihsg.index)
    ihsg_oos = ihsg[ihsg.index.isin(tanggal)].sort_index()["Adj Close"]
    ekuitas_ihsg = ihsg_oos / ihsg_oos.iloc[0] * modal
    gabung = pd.DataFrame({"eq_1n_full": ekuitas_1n, "eq_ihsg": ekuitas_ihsg})
    gabung.index.name = "date"
    gabung = gabung.reset_index()
    gabung["ret_1n_full"] = gabung["eq_1n_full"].pct_change().fillna(0.0)
    gabung["ret_ihsg"] = gabung["eq_ihsg"].pct_change().fillna(0.0)
    return gabung


def main() -> int:
    """Menjalankan grid pencarian optimasi portofolio dan menulis hasilnya.

    Alur kerja:
        1. Memuat konfigurasi, harga, dan prediksi.
        2. Menyusun kombinasi grid (k, estimator, lookback, max_weight).
        3. Menjalankan backtest per kombinasi dengan menyimpan hasil
           ke CSV secara append untuk ketahanan terhadap kegagalan.
        4. Menghitung baseline 1/N dan IHSG.
        5. Menulis metrik eksekusi ke `run_info.json`.

    Returns:
        Kode keluar 0 jika berhasil.
    """
    args = parse_args()
    port_cfg = load_config("portfolio")
    data_cfg = load_config("data")
    opt_cfg = port_cfg["optimization"]
    mulai = time.time()
    stempel = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out or (EXPERIMENT_DIR / f"optimize_{stempel}")
    ensure_dirs(run_dir)
    if args.grid == "utama":
        daftar_k = list(port_cfg["selection"]["k_values"])
        daftar_l = [int(port_cfg["covariance"]["lookback_days"])]
        daftar_maks = [float(port_cfg["constraints"]["max_weight"])]
    else:
        daftar_k = list(port_cfg["selection"]["k_values"]) + list(
            port_cfg["selection"]["k_sensitivity"]
        )
        daftar_l = [int(port_cfg["covariance"]["lookback_days"])] + list(
            port_cfg["covariance"]["lookback_sensitivity"]
        )
        daftar_maks = [float(port_cfg["constraints"]["max_weight"])] + [
            float(x) for x in port_cfg["constraints"]["max_weight_sensitivity"]
        ]
    estimators = list(port_cfg["covariance"]["estimators"])
    optimizer_types = opt_cfg.get("types", ["target_return"])
    biaya = data_cfg["costs"]
    frame = kerangka_prediksi(args.predictions)
    harga = muat_harga()
    peringkat = bingkai_peringkat(frame, args.seed_mode)
    tanggal_oos = sorted(frame["date"].unique().tolist())
    total = (
        len(daftar_k)
        * len(estimators)
        * len(daftar_l)
        * len(daftar_maks)
        * len(peringkat)
        * len(optimizer_types)
    )
    kerja = 0
    for optimizer_type in optimizer_types:
        for nama_mode, peringkat_df in peringkat.items():
            for k in daftar_k:
                for est in estimators:
                    for panjang in daftar_l:
                        for maks in daftar_maks:
                            kerja += 1
                            bobot, nilai = jalankan_backtest(
                                peringkat_df,
                                harga,
                                int(k),
                                est,
                                int(panjang),
                                float(maks),
                                modal=args.modal,
                                fee_beli=float(biaya["buy_fee"]),
                                fee_jual=float(biaya["sell_fee"]),
                                lot=int(biaya["lot_size"]),
                                ridge_epsilon=float(
                                    port_cfg["covariance"]["ridge_epsilon"]
                                ),
                                optimizer_type=optimizer_type,
                            )
                            bobot["seed_mode"] = nama_mode
                            nilai["k"] = int(k)
                            nilai["estimator"] = est
                            nilai["lookback"] = int(panjang)
                            nilai["max_weight"] = float(maks)
                            nilai["seed_mode"] = nama_mode
                            # Menyimpan hasil per iterasi secara append
                            # untuk mencegah kehilangan data jika terjadi
                            # kegagalan proses di tengah jalan.
                            weights_path = run_dir / "weights.csv"
                            returns_path = run_dir / "portfolio_returns.csv"
                            bobot.to_csv(
                                weights_path,
                                mode="a",
                                header=not weights_path.exists(),
                                index=False,
                            )
                            nilai.to_csv(
                                returns_path,
                                mode="a",
                                header=not returns_path.exists(),
                                index=False,
                            )
                            if kerja % 100 == 0 or kerja == total:
                                print(
                                    f"[{kerja}/{total}] {optimizer_type} {nama_mode} k={k} {est} "
                                    f"L={panjang} maxw={maks}",
                                    flush=True,
                                )
    garis_dasar(harga, tanggal_oos, args.modal).to_csv(
        run_dir / "baseline_returns.csv", index=False
    )
    info = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "predictions": str(args.predictions),
        "grid": args.grid,
        "k_values": daftar_k,
        "estimators": estimators,
        "optimizer_types": optimizer_types,
        "lookbacks": daftar_l,
        "max_weights": daftar_maks,
        "seed_modes": sorted(peringkat),
        "modal": args.modal,
        "n_configs": total,
        "wall_sec": round(time.time() - mulai, 1),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"selesai: {total} konfigurasi -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
