#!/usr/bin/env python3
"""Tahap 2: bersihkan data mentah lalu bangun fitur dan target.

Menghasilkan:
- ``data/interim/prices_clean/<TICKER>.csv``: OHLCV selaras kalender bursa
- ``data/processed/features/<TICKER>.csv``  : 8 fitur + kolom ``target``

Spesifikasi fitur: configs/experiment.yaml
Jejak keputusan: docs/keputusan_desain.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from lq45.features import build_macro, build_panel, clean_prices
from lq45.utils.config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_dirs,
    load_config,
)


def parse_args() -> argparse.Namespace:
    """Baca argumen baris perintah."""
    parser = argparse.ArgumentParser(
        description="Bangun fitur dan target dari data mentah."
    )
    return parser.parse_args()


def nama_berkas(ticker: str) -> str:
    """Nama berkas tanpa akhiran `.JK`."""
    return ticker.replace(".JK", "")


def main() -> int:
    """Bangun fitur dan target untuk seluruh saham kandidat."""
    parse_args()

    data_cfg = load_config("data")
    exp_cfg = load_config("experiment")
    model_cfg = load_config("model")
    tickers = list(load_config("universe")["tickers"])

    ffill_days = int(data_cfg["cleaning"]["forward_fill_max_days"])
    horizon = int(model_cfg["horizon_days"])
    periods = exp_cfg["features"]["indicators"]
    macro_transform = exp_cfg["features"]["macro_transform"]

    # Kalender bursa diambil dari IHSG.
    ihsg = pd.read_csv(RAW_DIR / "benchmark_ihsg.csv", parse_dates=["Date"])
    calendar = pd.DatetimeIndex(ihsg["Date"]).sort_values()
    print(
        f"Kalender bursa : {len(calendar)} hari"
        f" ({calendar.min().date()} s.d. {calendar.max().date()})"
    )

    jisdor = pd.read_csv(RAW_DIR / "macro" / "jisdor.csv", parse_dates=["date"])
    bi_rate = pd.read_csv(RAW_DIR / "macro" / "bi_7drrr.csv", parse_dates=["date"])
    macro = build_macro(jisdor, bi_rate, calendar, macro_transform)

    mentah: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = RAW_DIR / "prices" / f"{nama_berkas(ticker)}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        mentah[ticker] = clean_prices(frame, calendar, ffill_days)

    panel = build_panel(mentah, calendar, macro, horizon, periods)

    interim_dir = INTERIM_DIR / "prices_clean"
    processed_dir = PROCESSED_DIR / "features"
    ensure_dirs(interim_dir, processed_dir)

    for ticker, bersih in mentah.items():
        bersih.to_csv(interim_dir / f"{nama_berkas(ticker)}.csv")
    for ticker, fitur in panel.items():
        fitur.to_csv(processed_dir / f"{nama_berkas(ticker)}.csv")

    kolom = list(next(iter(panel.values())).columns)
    ringkas = pd.DataFrame(
        {
            "terisi": {
                k: int(sum(fitur[k].notna().sum() for fitur in panel.values()))
                for k in kolom
            },
            "kosong": {
                k: int(sum(fitur[k].isna().sum() for fitur in panel.values()))
                for k in kolom
            },
        }
    )
    ringkas["total"] = ringkas["terisi"] + ringkas["kosong"]
    lengkap = int(
        sum(fitur[kolom].notna().all(axis=1).sum() for fitur in panel.values())
    )
    print(f"\nSaham diproses : {len(panel)}")
    print(
        f"Baris lengkap  : {lengkap} dari {ringkas['total'].max()}"
        " (seluruh fitur + target terisi)"
    )
    print(f"Kolom keluaran : {', '.join(kolom)}")
    print("\nRekap per kolom (agregat lintas saham):")
    print(ringkas.to_string())
    print(f"\nInterim   : {interim_dir}")
    print(f"Processed : {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
