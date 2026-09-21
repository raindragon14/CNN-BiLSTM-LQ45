"""Penyusunan fitur harga, indikator, makro, dan target.

Sumber tiap pilihan (rincian: docs/keputusan_desain.md):
- Harga `Adj Close` level dan volume: Sebastian & Tantia (2024); Sen & Dutta (2021).
- RSI, CCI, CMO, MFI: Espiga-Fernandez et al. (2024) Lampiran B.
- BI-7DRRR level dan JISDOR log-return: ekstensi untuk celah
  Chaweewanchon & Chaysiri (2022).
- Penyesuaian OHLC dengan faktor `Adj Close / Close`: prinsip mencegah
  lompatan akibat dividen dan pemecahan saham.
- Kalender bursa dari IHSG; forward-fill maksimum 3 hari.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from lq45.features.indicators import cci, cmo, mfi, rsi

# Espiga-Fernandez et al. (2024) tidak mematok periode tiap indikator,
# jadi dipakai nilai standar: RSI 14, CCI 20, CMO 14, MFI 14.
INDICATOR_PERIODS: dict[str, int] = {"rsi": 14, "cci": 20, "cmo": 14, "mfi": 14}

KOLOM_HARGA = ("Open", "High", "Low", "Close", "Adj Close")


def clean_prices(
    frame: pd.DataFrame, calendar: pd.DatetimeIndex, ffill_days: int
) -> pd.DataFrame:
    """Selaraskan ke kalender bursa lalu isi celah maksimum `ffill_days` hari."""
    frame = frame.reindex(calendar)
    frame[list(KOLOM_HARGA)] = frame[list(KOLOM_HARGA)].ffill(limit=ffill_days)
    frame["Volume"] = frame["Volume"].ffill(limit=ffill_days).fillna(0.0)
    return frame


def build_stock_features(
    frame: pd.DataFrame, periods: Mapping[str, int] | None = None
) -> pd.DataFrame:
    """Fitur harga dan indikator untuk satu saham (8 kolom)."""
    periods = dict(periods or INDICATOR_PERIODS)

    # Sesuaikan OHLC dengan faktor dari kolom Adj Close (mencegah lompatan
    # akibat dividen dan pemecahan saham).
    faktor = frame["Adj Close"] / frame["Close"]
    high = frame["High"] * faktor
    low = frame["Low"] * faktor
    close = frame["Adj Close"]
    volume = frame["Volume"]

    hasil = pd.DataFrame(index=frame.index)
    hasil["close"] = close
    hasil["volume"] = volume
    hasil["rsi"] = rsi(close, periods["rsi"])
    hasil["cci"] = cci(high, low, close, periods["cci"])
    hasil["cmo"] = cmo(close, periods["cmo"])
    hasil["mfi"] = mfi(high, low, close, volume, periods["mfi"])
    return hasil


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Target: log-return `horizon` hari ke depan."""
    return np.log(close.shift(-horizon) / close)


def build_macro(
    jisdor: pd.DataFrame,
    bi_rate: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    transform: Mapping[str, str],
) -> pd.DataFrame:
    """Selaraskan makro ke kalender bursa menurut tanggal publikasi.

    BI-7DRRR diteruskan tanpa batas karena suku bunga acuan berlaku sampai
    keputusan berikutnya. Kurs JISDOR diubah menjadi log-return harian
    karena levelnya non-stasioner.
    """
    bi = bi_rate.set_index("date")["rate"].sort_index()
    kurs = jisdor.set_index("date")["rate"].sort_index()

    bi_harian = bi.reindex(calendar, method="ffill")
    kurs_harian = kurs.reindex(calendar, method="ffill")

    if transform.get("jisdor") == "log_return":
        kurs_harian = np.log(kurs_harian / kurs_harian.shift(1))

    return pd.DataFrame({"bi_7drrr": bi_harian, "jisdor": kurs_harian})


def build_panel(
    prices: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    macro: pd.DataFrame,
    horizon: int,
    periods: Mapping[str, int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Susun panel fitur + makro + target untuk setiap saham."""
    panel: dict[str, pd.DataFrame] = {}
    for ticker, mentah in prices.items():
        fitur = build_stock_features(mentah, periods)
        fitur = fitur.join(macro, how="left")
        fitur["target"] = forward_log_return(fitur["close"], horizon)
        panel[ticker] = fitur
    return panel
