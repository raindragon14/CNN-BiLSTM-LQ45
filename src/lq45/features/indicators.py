"""Technical indicators.

Formulas follow Espiga-Fernandez et al. (2024), Appendix B: RSI in B.3,
CCI in B.4, CMO in B.5, and MFI in B.6. The B.x numbers refer to the
appendix of that article, not to internal code.
"""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100), 14-period."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    value = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss = 0 means no decline in the window, so RSI = 100.
    return value.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Commodity Channel Index, 20-period."""
    typical = (high + low + close) / 3.0
    mean = typical.rolling(period, min_periods=period).mean()
    deviation = (typical - mean).abs().rolling(period, min_periods=period).mean()
    return (typical - mean) / (0.015 * deviation)


def cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator (-100 to 100), 14-period."""
    delta = close.diff()
    up = delta.clip(lower=0.0).rolling(period, min_periods=period).sum()
    down = (-delta.clip(upper=0.0)).rolling(period, min_periods=period).sum()
    denominator = up + down
    return (100.0 * (up - down) / denominator.where(denominator != 0.0)).where(
        denominator.notna()
    )


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index (0-100), using price and volume; 14-period."""
    typical = (high + low + close) / 3.0
    flow = typical * volume
    change = typical.diff()
    positive = flow.where(change > 0, 0.0).rolling(period, min_periods=period).sum()
    negative = flow.where(change < 0, 0.0).rolling(period, min_periods=period).sum()
    ratio = positive / negative.replace(0.0, float("nan"))
    value = 100.0 - 100.0 / (1.0 + ratio)
    return value.where(negative != 0.0, 100.0).where(positive.notna())
