"""Indikator teknikal.

Rumus mengikuti Espiga-Fernandez et al. (2024), Lampiran B: RSI pada B.3,
CCI pada B.4, CMO pada B.5, dan MFI pada B.6. Nomor B.x menunjuk lampiran
artikel tersebut, bukan kode internal.
"""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100), periode 14."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    nilai = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss = 0 berarti tidak ada penurunan pada jendela, RSI = 100.
    return nilai.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Commodity Channel Index, periode 20."""
    typical = (high + low + close) / 3.0
    rata = typical.rolling(period, min_periods=period).mean()
    simpangan = (typical - rata).abs().rolling(period, min_periods=period).mean()
    return (typical - rata) / (0.015 * simpangan)


def cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator (-100 sampai 100), periode 14."""
    delta = close.diff()
    naik = delta.clip(lower=0.0).rolling(period, min_periods=period).sum()
    turun = (-delta.clip(upper=0.0)).rolling(period, min_periods=period).sum()
    penyebut = naik + turun
    return (100.0 * (naik - turun) / penyebut.where(penyebut != 0.0)).where(
        penyebut.notna()
    )


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index (0-100), memakai harga dan volume; periode 14."""
    typical = (high + low + close) / 3.0
    aliran = typical * volume
    perubahan = typical.diff()
    positif = aliran.where(perubahan > 0, 0.0).rolling(period, min_periods=period).sum()
    negatif = aliran.where(perubahan < 0, 0.0).rolling(period, min_periods=period).sum()
    rasio = positif / negatif.replace(0.0, float("nan"))
    nilai = 100.0 - 100.0 / (1.0 + rasio)
    return nilai.where(negatif != 0.0, 100.0).where(positif.notna())
