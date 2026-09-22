"""Feature panel and sliding windows for the pooled model across all stocks.

Underlying decisions (details: docs/keputusan_desain.md):
- A single pooled model for all stocks: Espiga-Fernandez et al. (2024)
  use a `lookback x instruments x features` tensor; Huang et al. (2024)
  use a `32 x 15` tensor.
- Log-return target: Sen & Dutta (2021) compute returns as log returns.
- The target is recomputed from the `close` column so the horizon
  parameter is the single source of truth; at horizon 5 the result is
  matched against the stage-2 `target` column as a cross-check.
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
    """Panel of all stocks on a single exchange calendar.

    Every array shares the same date index (`dates`) and is float32.
    """

    tickers: list[str]
    dates: pd.DatetimeIndex
    features: dict[str, np.ndarray]  # (T, 8), in FEATURE_COLUMNS order
    target: dict[str, np.ndarray]  # (T,) horizon-day log-return
    close: dict[str, np.ndarray]  # (T,) adjusted closing price


@dataclass
class RawStockData:
    """Raw per-stock data for pre-training (independent calendar)."""

    ticker: str
    dates: pd.DatetimeIndex
    features: np.ndarray  # (T, 7) = OHLCV + BI-7DRRR + JISDOR
    close: np.ndarray  # (T,)


def stock_file_name(ticker: str) -> str:
    """File name without the `.JK` suffix."""
    return ticker.replace(".JK", "")


def forward_log_return(close: np.ndarray, horizon: int) -> np.ndarray:
    """Forward `horizon`-day log-return target; NaN on the right edge."""
    result = np.full_like(close, np.nan)
    if horizon <= 0 or len(close) <= horizon:
        return result
    result[:-horizon] = np.log(close[horizon:] / close[:-horizon])
    return result


def load_panel(processed_dir: Path, tickers: Sequence[str], horizon: int) -> PanelData:
    """Load the processed feature panel and recompute the target.

    `tickers` carry the `.JK` suffix (as in `configs/universe.yaml`); the
    processed files do not use that suffix.
    """
    if not tickers:
        raise ValueError("empty ticker list")
    features: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    close: dict[str, np.ndarray] = {}
    dates: pd.DatetimeIndex | None = None

    for ticker in tickers:
        path = processed_dir / f"{stock_file_name(ticker)}.csv"
        if not path.exists():
            raise FileNotFoundError(f"processed features not found: {path}")
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        if dates is None:
            dates = frame.index
        elif not frame.index.equals(dates):
            raise ValueError(
                f"calendar for {ticker} differs from the reference calendar"
            )
        close_prices = frame["close"].to_numpy(dtype=np.float32)
        computed_target = forward_log_return(close_prices, horizon)
        if horizon == 5:
            csv_target = frame["target"].to_numpy(dtype=np.float32)
            if not np.allclose(
                computed_target,
                csv_target,
                equal_nan=True,
                rtol=1e-4,
                atol=1e-7,
            ):
                raise ValueError(
                    f"recomputed target does not match the CSV target column: "
                    f"{ticker}"
                )
        features[ticker] = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        target[ticker] = computed_target
        close[ticker] = close_prices

    return PanelData(
        tickers=list(tickers),
        dates=dates,
        features=features,
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
    """Windows for dates `[start, stop)` outside the `banned` ranges.

    A sample at date `t` uses features `[t-lookback+1 .. t]` and label
    `target[t]`; dates younger than `lookback-1` have no full window and
    are skipped. Windows containing NaN are dropped.

    Returns `(X, y, idx)` where `X` has shape `(N, F, lookback)`, `y` is
    the label, and `idx` is the date position within the input array.
    """
    first_index = max(start, lookback - 1)
    n = stop - first_index
    empty = np.empty((0, features.shape[1], lookback), dtype=np.float32)
    if n <= 0:
        return empty, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

    segment = features[first_index - lookback + 1 : stop]
    windows = np.lib.stride_tricks.sliding_window_view(segment, lookback, axis=0)
    idx = np.arange(first_index, stop, dtype=np.int64)

    keep = np.ones(n, dtype=bool)
    for a, b in banned:
        keep[(idx >= a) & (idx < b)] = False
    windows = windows[keep]
    idx = idx[keep]
    label = target[idx]

    valid = ~np.isnan(windows).any(axis=(1, 2)) & ~np.isnan(label)
    return windows[valid], label[valid], idx[valid]


# Pretrain channels: OHLCV + BI-7DRRR + JISDOR (7 channels)
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
    """Load raw per-stock OHLCV + macro data for pre-training.

    Each stock is processed independently on its own calendar.
    Channels: 7 (OHLCV + BI-7DRRR + JISDOR).
    """
    if not tickers:
        raise ValueError("empty ticker list")

    prices_dir = raw_dir / "prices"
    if not prices_dir.exists():
        raise FileNotFoundError(f"prices directory not found: {prices_dir}")

    macro_dir = raw_dir / "macro"
    bi_path = macro_dir / "bi_7drrr.csv"
    jisdor_path = macro_dir / "jisdor.csv"

    if not bi_path.exists() or not jisdor_path.exists():
        raise FileNotFoundError(f"macro files not found: {bi_path}, {jisdor_path}")

    bi_df = pd.read_csv(bi_path, parse_dates=["date"]).set_index("date")
    jisdor_df = pd.read_csv(jisdor_path, parse_dates=["date"]).set_index("date")

    if "rate" in bi_df.columns:
        bi_df = bi_df.rename(columns={"rate": "bi_7drrr"})
    if "rate" in jisdor_df.columns:
        jisdor_df = jisdor_df.rename(columns={"rate": "jisdor"})

    stocks: list[RawStockData] = []

    for ticker in tickers:
        path = prices_dir / f"{stock_file_name(ticker)}.csv"
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
    """Build windows for pre-training (no labels, no banned ranges).

    Returns:
        X: (N, C, lookback)
        idx: (N,) date positions
    """
    first_index = max(start, lookback - 1)
    n = stop - first_index
    empty = np.empty((0, features.shape[1], lookback), dtype=np.float32)
    if n <= 0:
        return empty, np.empty(0, dtype=np.int64)

    segment = features[first_index - lookback + 1 : stop]
    windows = np.lib.stride_tricks.sliding_window_view(segment, lookback, axis=0)
    idx = np.arange(first_index, stop, dtype=np.int64)

    valid = ~np.isnan(windows).any(axis=(1, 2))
    return windows[valid], idx[valid]
