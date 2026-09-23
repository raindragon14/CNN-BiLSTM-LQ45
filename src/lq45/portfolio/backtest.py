"""Monthly backtest engine: ranking, optimization, costs, holding.

Flow at each rebalancing date `d` (all using information up to `d`):
1. Rank the ensemble pred_raw on date `d`, take the top-k (stocks
   without a prediction are excluded because of suspension).
2. Estimate the covariance from L daily log returns up to `d`.
3. Optimize minimum variance or mean-variance under long-only, sum-to-one,
   and maximum-weight constraints.
4. Execute `execution_lag_days` trading days after `d`: round to lots of
   100, deduct a 0.19% buy fee and a 0.29% sell fee.
5. Hold until the next rebalancing; daily value follows prices.

Rebalancing dates use the out-of-sample prediction calendar every `step`
days to align with the stage-3 test windows.
"""

from __future__ import annotations

import pandas as pd

from lq45.portfolio.costs import target_shares, transaction_value
from lq45.portfolio.covariance import (
    estimate_covariance,
    returns_matrix,
)
from lq45.portfolio.optimize import (
    mean_variance_target_return_weights,
    minimum_variance_weights,
)
from lq45.portfolio.ranking import top_k


def rebalance_dates(oos_dates: list[str], step: int = 21) -> list[str]:
    """Take every `step`-th date from the sorted OOS calendar."""
    ordered = sorted(set(oos_dates))
    return ordered[::step]


def _target_return_from_predictions(
    predictions: pd.DataFrame,
    date: str,
    selected: list[str],
) -> float:
    """Compute the average pred_ens target return for the selected stocks."""
    rows = predictions[
        (predictions["date"] == date) & (predictions["ticker"].isin(selected))
    ]
    if rows.empty:
        return 0.0
    return float(rows["pred_ens"].mean())


def run_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    k: int,
    estimator: str,
    lookback: int,
    max_weight: float,
    capital: float = 100_000_000.0,
    buy_fee: float = 0.0019,
    sell_fee: float = 0.0029,
    lot: int = 100,
    step: int = 21,
    ridge_epsilon: float = 1e-4,
    optimizer_type: str = "target_return",
    execution_lag_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest a single configuration; return (weights, daily value).

    `predictions` uses columns `date, ticker, pred_ens`; `prices` is
    indexed by date with one column per ticker (Adj Close). `weights`
    has one row per rebalancing date plus the execution date used for the
    trade; the daily value uses columns `date, equity, return`.
    `optimizer_type`: "target_return" (MV) or "min_variance" (GMV).
    Trades execute `execution_lag_days` trading days after the signal
    date, so the decision never trades on the same bar it was made from
    (Kim et al. 2025 use a two-day lag).
    """
    oos_dates = sorted(predictions["date"].unique().tolist())
    schedule = rebalance_dates(oos_dates, step)
    price_index = prices.index
    cash = float(capital)
    holdings = pd.Series(dtype=float)
    weight_rows: list[dict] = []
    value_rows: list[dict] = []
    last_date = oos_dates[-1]

    for i, date in enumerate(schedule):
        selected = top_k(predictions, date, k)
        if selected:
            returns = returns_matrix(prices[selected], date, lookback)
            returns = returns.dropna(axis=1)
            selected = returns.columns.tolist()
        if not selected:
            weights = pd.Series(dtype=float)
        else:
            cov = estimate_covariance(returns, estimator, ridge_epsilon)
            if optimizer_type == "target_return":
                expected_returns = predictions[predictions["date"] == date].set_index(
                    "ticker"
                )["pred_ens"]
                target_ret = _target_return_from_predictions(
                    predictions, date, selected
                )
                weights = mean_variance_target_return_weights(
                    cov, expected_returns, target_ret, max_weight
                )
            else:
                weights = minimum_variance_weights(cov, max_weight)
            weights = weights.reindex(selected).fillna(0.0)

        future = price_index[price_index > date]
        execution_date = None
        if len(weights) and len(future) >= execution_lag_days:
            execution_date = future[execution_lag_days - 1]

        boundary = schedule[i + 1] if i + 1 < len(schedule) else last_date
        period_days = price_index[(price_index >= date) & (price_index <= boundary)]
        if execution_date is not None:
            pre_trade_days = period_days[period_days < execution_date]
            post_trade_days = period_days[period_days >= execution_date]
        else:
            pre_trade_days = period_days
            post_trade_days = period_days[:0]

        for day in pre_trade_days:
            day_prices = prices.loc[day]
            holdings_value = float(
                (holdings * day_prices.reindex(holdings.index).fillna(0.0)).sum()
                if len(holdings)
                else 0.0
            )
            value_rows.append({"date": day, "equity": cash + holdings_value})

        if execution_date is not None:
            exec_prices = prices.loc[execution_date]
            equity = cash + float(
                (holdings * exec_prices.reindex(holdings.index).fillna(0.0)).sum()
                if len(holdings)
                else 0.0
            )
            target = target_shares(weights, exec_prices, equity, lot)
            target = target.reindex(holdings.index.union(target.index)).fillna(0.0)
            old = holdings.reindex(target.index).fillna(0.0)
            cash_flow, fee = transaction_value(
                old, target, exec_prices, buy_fee, sell_fee
            )
            cash = cash + cash_flow - fee
            holdings = target
            for ticker, w in weights.items():
                weight_rows.append(
                    {
                        "date": date,
                        "execution_date": pd.Timestamp(execution_date).strftime(
                            "%Y-%m-%d"
                        ),
                        "ticker": ticker,
                        "weight": float(w),
                        "k": k,
                        "estimator": estimator,
                        "lookback": lookback,
                        "max_weight": max_weight,
                        "optimizer_type": optimizer_type,
                    }
                )

        for day in post_trade_days:
            day_prices = prices.loc[day]
            holdings_value = float(
                (holdings * day_prices.reindex(holdings.index).fillna(0.0)).sum()
                if len(holdings)
                else 0.0
            )
            value_rows.append({"date": day, "equity": cash + holdings_value})

    value = pd.DataFrame(value_rows)
    value["date"] = pd.to_datetime(value["date"])
    value = value.sort_values("date").drop_duplicates("date")
    value["return"] = value["equity"].pct_change().fillna(0.0)
    weights_frame = pd.DataFrame(
        weight_rows,
        columns=[
            "date",
            "execution_date",
            "ticker",
            "weight",
            "k",
            "estimator",
            "lookback",
            "max_weight",
            "optimizer_type",
        ],
    )
    return weights_frame, value
