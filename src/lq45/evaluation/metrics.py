"""Portfolio evaluation metrics: risk, return, and implicit costs.

Basis (details: docs/keputusan_desain.md):
- Sharpe, Sortino, Calmar, MDD, turnover: Malhotra et al. (2023) use
  Sharpe/Sortino/Omega as mutual-fund standards; Wang & Liu (2025)
  define risk-sensitive evaluation (Sharpe/Sortino/Calmar).
- Daily risk-free = annual BI-7DRRR divided by 252 (E8, official BI source).
- Annualization over 252 trading days (E9, standard convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown_and_duration(equity: pd.Series) -> tuple[float, int]:
    """Maximum drawdown (fraction) and day count from peak to trough."""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak.replace(0.0, np.nan)
    drawdown = drawdown.fillna(0.0)
    mdd = float(drawdown.min())
    trough_position = int(drawdown.to_numpy().argmin()) if len(drawdown) else 0
    peak_position = (
        int(peak.to_numpy()[: trough_position + 1].argmax()) if len(drawdown) else 0
    )
    return mdd, max(0, trough_position - peak_position)


def summarize_series(
    returns: pd.Series,
    risk_free: pd.Series | float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Summarize a daily return series into annualized metrics.

    `risk_free` is a daily series or a fixed number. Sortino uses a
    target of zero on the downside. Calmar uses annual return divided
    by |MDD| (zero when MDD is zero).
    """
    r = returns.to_numpy(dtype=float)
    if isinstance(risk_free, pd.Series):
        rf = risk_free.reindex(returns.index).fillna(0.0).to_numpy(dtype=float)
    else:
        rf = np.full_like(r, float(risk_free))
    excess = r - rf
    n = len(r)
    cumulative = float(np.prod(1.0 + r) - 1.0) if n else 0.0
    annual = float((1.0 + cumulative) ** (periods_per_year / n) - 1.0) if n else 0.0
    vol = float(np.std(excess, ddof=1)) if n > 1 else 0.0
    sharpe = float(np.sqrt(periods_per_year) * excess.mean() / vol) if vol > 0 else 0.0
    down = excess[excess < 0.0]
    vol_down = float(np.std(down, ddof=1)) if len(down) > 1 else 0.0
    sortino = (
        float(np.sqrt(periods_per_year) * excess.mean() / vol_down)
        if vol_down > 0
        else 0.0
    )
    equity = pd.Series(np.cumprod(1.0 + r), index=returns.index)
    mdd, _ = max_drawdown_and_duration(equity)
    calmar = float(annual / abs(mdd)) if mdd < 0 else 0.0
    return {
        "cumulative_return": cumulative,
        "annual_return": annual,
        "annual_vol": float(vol * np.sqrt(periods_per_year)) if n > 1 else 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "n_days": float(n),
    }


def daily_risk_free_rate(
    date: pd.DatetimeIndex, rate_file: str, periods_per_year: int = 252
) -> pd.Series:
    """Daily risk-free series from the BI-7DRRR decision history.

    The file uses columns `date, rate` (annual percent on the decision
    date); values are forward-filled and then divided by 100 and 252.
    """
    rate = pd.read_csv(rate_file, parse_dates=["date"]).set_index("date")
    rate.index = pd.to_datetime(rate.index)
    daily = rate["rate"].reindex(date, method="ffill").bfill()
    return (daily / 100.0 / periods_per_year).astype(float)
