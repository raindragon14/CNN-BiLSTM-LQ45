"""Panel fitur dan jendela geser untuk model bersama seluruh saham.

Keputusan yang mendasari (rincian: docs/keputusan_desain.md):
- Satu model bersama untuk semua saham: Espiga-Fernandez et al. (2024)
  memakai tensor `lookback x instrumen x fitur`; Huang et al. (2024)
  memakai tensor `32 x 15`.
- Target log-return: Sen & Dutta (2021) menghitung imbal hasil sebagai
  log return.
- Target dihitung ulang dari kolom `close` agar parameter horizon menjadi
  sumber acuan tunggal; saat horizon 5, hasilnya dicocokkan dengan kolom
  `target` buatan tahap 2 sebagai pemeriksaan silang.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "close",
    "volume",
    "rsi",
    "cci",
    "cmo",
    "mfi",
    "bi_7drrr",
    "jisdor",
)


@dataclass
class PanelData:
    """Panel seluruh saham pada satu kalender bursa.

    Setiap larik berindeks tanggal yang sama (`dates`) dan bertipe float32.
    """

    tickers: list[str]
    dates: pd.DatetimeIndex
    features: dict[str, np.ndarray]  # (T, 8), urutan FEATURE_COLUMNS
    target: dict[str, np.ndarray]  # (T,) log-return horizon hari
    close: dict[str, np.ndarray]  # (T,) harga penutupan terkoreksi


@dataclass
class RawStockData:
    """Data mentah per saham untuk pre-training (kalender independen)."""

    ticker: str
    dates: pd.DatetimeIndex
    features: np.ndarray  # (T, 7) = OHLCV + BI-7DRRR + JISDOR
    close: np.ndarray  # (T,)


def nama_berkas(ticker: str) -> str:
    """Nama berkas tanpa akhiran `.JK`."""
    return ticker.replace(".JK", "")


def forward_log_return(close: np.ndarray, horizon: int) -> np.ndarray:
    """Target log-return `horizon` hari ke depan; NaN pada ujung kanan."""
    hasil = np.full_like(close, np.nan)
    if horizon <= 0 or len(close) <= horizon:
        return hasil
    hasil[:-horizon] = np.log(close[horizon:] / close[:-horizon])
    return hasil


def load_panel(processed_dir: Path, tickers: Sequence[str], horizon: int) -> PanelData:
    """Muat panel fitur terproses dan hitung ulang target.

    `tickers` memakai akhiran `.JK` (bentuk di `configs/universe.yaml`);
    berkas terproses tidak memakai akhiran itu.
    """
    if not tickers:
        raise ValueError("daftar ticker kosong")
    fitur: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    close: dict[str, np.ndarray] = {}
    dates: pd.DatetimeIndex | None = None

    for ticker in tickers:
        path = processed_dir / f"{nama_berkas(ticker)}.csv"
        if not path.exists():
            raise FileNotFoundError(f"fitur terproses tidak ditemukan: {path}")
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        if dates is None:
            dates = frame.index
        elif not frame.index.equals(dates):
            raise ValueError(f"kalender {ticker} tidak sama dengan kalender acuan")
        harga = frame["close"].to_numpy(dtype=np.float32)
        hasil_target = forward_log_return(harga, horizon)
        if horizon == 5:
            csv_target = frame["target"].to_numpy(dtype=np.float32)
            if not np.allclose(
                hasil_target,
                csv_target,
                equal_nan=True,
                rtol=1e-4,
                atol=1e-7,
            ):
                raise ValueError(
                    f"target hitung ulang tidak cocok dengan kolom target "
                    f"CSV: {ticker}"
                )
        fitur[ticker] = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        target[ticker] = hasil_target
        close[ticker] = harga

    return PanelData(
        tickers=list(tickers),
        dates=dates,
        features=fitur,
        target=target,
        close=close,
    )


def build_windows(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    start: int,
    stop: int,
    banned: Sequence[tuple[int, int]] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jendela untuk tanggal `[start, stop)` di luar rentang `banned`.

    Sampel tanggal `t` memakai fitur `[t-lookback+1 .. t]` dan label
    `target[t]`; tanggal yang lebih muda dari `lookback-1` tidak memiliki
    jendela lengkap sehingga tidak ikut. Jendela yang memuat NaN dibuang.

    Mengembalikan `(X, y, idx)` dengan `X` berbentuk `(N, F, lookback)`,
    `y` label, dan `idx` posisi tanggal dalam larik masukan.
    """
    awal_t = max(start, lookback - 1)
    n = stop - awal_t
    kosong = np.empty((0, features.shape[1], lookback), dtype=np.float32)
    if n <= 0:
        return kosong, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

    segmen = features[awal_t - lookback + 1 : stop]
    jendela = np.lib.stride_tricks.sliding_window_view(segmen, lookback, axis=0)
    idx = np.arange(awal_t, stop, dtype=np.int64)

    pakai = np.ones(n, dtype=bool)
    for a, b in banned:
        pakai[(idx >= a) & (idx < b)] = False
    jendela = jendela[pakai]
    idx = idx[pakai]
    label = target[idx]

    valid = ~np.isnan(jendela).any(axis=(1, 2)) & ~np.isnan(label)
    return jendela[valid], label[valid], idx[valid]


# Pretrain channels: OHLCV + BI-7DRRR + JISDOR (7 channel)
PRETRAIN_CHANNELS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bi_7drrr",
    "jisdor",
)


def load_raw_stocks(raw_dir: Path, tickers: Sequence[str]) -> list[RawStockData]:
    """Muat data mentah OHLCV + makro per saham untuk pre-training.

    Setiap saham diproses independen dengan kalendernya sendiri.
    Channel: 7 (OHLCV + BI-7DRRR + JISDOR).
    """
    if not tickers:
        raise ValueError("daftar ticker kosong")

    prices_dir = raw_dir / "prices"
    if not prices_dir.exists():
        raise FileNotFoundError(f"direktori prices tidak ditemukan: {prices_dir}")

    macro_dir = raw_dir / "macro"
    bi_path = macro_dir / "bi_7drrr.csv"
    jisdor_path = macro_dir / "jisdor.csv"

    if not bi_path.exists() or not jisdor_path.exists():
        raise FileNotFoundError(f"file makro tidak ditemukan: {bi_path}, {jisdor_path}")

    bi_df = pd.read_csv(bi_path, parse_dates=["date"]).set_index("date")
    jisdor_df = pd.read_csv(jisdor_path, parse_dates=["date"]).set_index("date")

    if "rate" in bi_df.columns:
        bi_df = bi_df.rename(columns={"rate": "bi_7drrr"})
    if "rate" in jisdor_df.columns:
        jisdor_df = jisdor_df.rename(columns={"rate": "jisdor"})

    stocks: list[RawStockData] = []

    for ticker in tickers:
        path = prices_dir / f"{nama_berkas(ticker)}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        dates = frame.index

        # Align macro to this stock's calendar
        bi_aligned = bi_df.reindex(dates, method="ffill")["bi_7drrr"].to_numpy(
            dtype=np.float32
        )
        jisdor_aligned = jisdor_df.reindex(dates, method="ffill")["jisdor"].to_numpy(
            dtype=np.float32
        )

        # Handle initial NaN (dates before first macro observation)
        if np.isnan(bi_aligned).any():
            bi_series = pd.Series(bi_aligned)
            bi_aligned = bi_series.ffill().bfill().to_numpy(dtype=np.float32)
        if np.isnan(jisdor_aligned).any():
            jisdor_series = pd.Series(jisdor_aligned)
            jisdor_aligned = jisdor_series.ffill().bfill().to_numpy(dtype=np.float32)

        feat = np.column_stack(
            [
                frame["Open"].to_numpy(dtype=np.float32),
                frame["High"].to_numpy(dtype=np.float32),
                frame["Low"].to_numpy(dtype=np.float32),
                frame["Close"].to_numpy(dtype=np.float32),
                frame["Volume"].to_numpy(dtype=np.float32),
                bi_aligned,
                jisdor_aligned,
            ]
        )

        stocks.append(
            RawStockData(
                ticker=ticker,
                dates=dates,
                features=feat,
                close=frame["Close"].to_numpy(dtype=np.float32),
            )
        )

    return stocks


def build_pretrain_windows(
    features: np.ndarray,
    lookback: int,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bangun jendela untuk pre-training (tanpa label, tanpa banned).

    Returns:
        X: (N, C, lookback)
        idx: (N,) posisi tanggal
    """
    awal_t = max(start, lookback - 1)
    n = stop - awal_t
    kosong = np.empty((0, features.shape[1], lookback), dtype=np.float32)
    if n <= 0:
        return kosong, np.empty(0, dtype=np.int64)

    segmen = features[awal_t - lookback + 1 : stop]
    jendela = np.lib.stride_tricks.sliding_window_view(segmen, lookback, axis=0)
    idx = np.arange(awal_t, stop, dtype=np.int64)

    valid = ~np.isnan(jendela).any(axis=(1, 2))
    return jendela[valid], idx[valid]
