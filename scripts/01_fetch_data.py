#!/usr/bin/env python3
"""Tahap 1: unduh data mentah ke ``data/raw/``.

Menghasilkan:
- ``data/raw/prices/<TICKER>.csv``  : OHLCV harian (kolom ``Adj Close``)
- ``data/raw/macro/jisdor.csv``     : kurs referensi resmi USD/IDR (BI)
- ``data/raw/macro/bi_7drrr.csv``   : riwayat keputusan BI-7DRRR
- ``data/raw/benchmark_ihsg.csv``   : IHSG untuk pembanding beli-dan-tahan
- ``data/raw/metadata.json``        : ringkasan unduhan (waktu, jumlah baris)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from lq45.data import fetch_bi_rate, fetch_equities, fetch_jisdor
from lq45.utils.config import RAW_DIR, ensure_dirs, load_config


def parse_args() -> argparse.Namespace:
    """Baca argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Unduh data mentah LQ45 dan makro.")
    parser.add_argument("--start", default=None, help="tanggal awal (YYYY-MM-DD)")
    parser.add_argument(
        "--end", default=None, help="tanggal akhir inklusif (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="daftar ticker dipisah koma; default dari configs/universe.yaml",
    )
    return parser.parse_args()


def main() -> int:
    """Unduh harga, kurs, suku bunga acuan, dan tolok ukur ke ``data/raw/``."""
    args = parse_args()
    data_cfg = load_config("data")

    start = args.start or data_cfg["period"]["start"]
    end = args.end or data_cfg["period"]["end"]
    tickers = (
        [t.strip() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else list(load_config("universe")["tickers"])
    )

    # Yahoo memakai batas akhir eksklusif.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    prices_dir = RAW_DIR / "prices"
    macro_dir = RAW_DIR / "macro"
    ensure_dirs(prices_dir, macro_dir)

    print(f"Periode   : {start} s.d. {end}")
    print(f"Ticker    : {len(tickers)} saham")

    frames = fetch_equities(tickers, start, end_exclusive)
    tersimpan, kosong = [], []
    for ticker, frame in frames.items():
        if frame is None or frame.empty:
            kosong.append(ticker)
            continue
        nama = ticker.replace(".JK", "")
        frame.to_csv(prices_dir / f"{nama}.csv")
        tersimpan.append((nama, len(frame)))

    print("\nHarga saham tersimpan:")
    for nama, jumlah in tersimpan:
        print(f"  {nama:<6} {jumlah:>5} baris")
    if kosong:
        print(f"  TIDAK ADA DATA: {', '.join(kosong)}")

    fx = fetch_jisdor(start, end)
    if not fx.empty:
        fx.to_csv(macro_dir / "jisdor.csv", index=False)
        print(f"JISDOR USD/IDR: {len(fx)} baris -> {macro_dir / 'jisdor.csv'}")
        print(
            f"  rentang {fx['date'].min().date()} s.d. {fx['date'].max().date()}"
            f" | kurs {fx['rate'].min():,.0f}-{fx['rate'].max():,.0f}"
        )

    bi = fetch_bi_rate()
    bi = bi[(bi["date"] >= pd.Timestamp(start)) & (bi["date"] <= pd.Timestamp(end))]
    bi.to_csv(macro_dir / "bi_7drrr.csv", index=False)
    print(f"BI-7DRRR     : {len(bi)} pertemuan -> {macro_dir / 'bi_7drrr.csv'}")
    if not bi.empty:
        print(
            f"  rentang {bi['date'].min().date()} s.d. {bi['date'].max().date()}"
            f" | suku bunga {bi['rate'].min():.2f}-{bi['rate'].max():.2f}%"
        )

    benchmark = data_cfg["sources"].get("benchmark", "^JKSE")
    indeks = fetch_equities([benchmark], start, end_exclusive)[benchmark]
    baris_indeks = 0
    if indeks is not None and not indeks.empty:
        indeks.to_csv(RAW_DIR / "benchmark_ihsg.csv")
        baris_indeks = len(indeks)
        print(
            f"\nBenchmark {benchmark:<6}: {baris_indeks} baris"
            f" -> {RAW_DIR / 'benchmark_ihsg.csv'}"
        )

    metadata = {
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "period": {"start": start, "end": end},
        "tickers": list(tickers),
        "missing": kosong,
        "rows": {nama: jumlah for nama, jumlah in tersimpan},
        "benchmark": {"ticker": benchmark, "rows": baris_indeks},
        "sources": {
            "prices": "Yahoo Finance (.JK, adjusted close)",
            "fx": "Bank Indonesia (JISDOR, kurs referensi resmi)",
            "bi_rate": "Bank Indonesia (bi-rate.aspx)",
            "benchmark": "Yahoo Finance (^JKSE)",
        },
    }
    (RAW_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nMetadata     : {RAW_DIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
