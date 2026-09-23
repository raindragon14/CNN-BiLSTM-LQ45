"""Portfolio module tests; run directly: python3 tests/test_portfolio.py.

The data here is synthetic and only validates the pipeline, not results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.portfolio.backtest import (
    rebalance_dates,
    run_backtest,
)
from lq45.portfolio.costs import target_shares, transaction_value
from lq45.portfolio.covariance import estimate_covariance
from lq45.portfolio.optimize import (
    mean_variance_target_return_weights,
    minimum_variance_weights,
)
from lq45.portfolio.ranking import ensemble_predictions, top_k


def test_ensemble_and_topk() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 6,
            "ticker": ["A", "B", "C"] * 2,
            "pred_raw": [0.1, 0.3, 0.2, 0.3, 0.1, 0.2],
        }
    )
    ensemble = ensemble_predictions(frame)
    assert len(ensemble) == 3
    assert top_k(ensemble, "2020-01-01", 2) == ["A", "B"]
    assert top_k(ensemble, "2020-02-01", 2) == []


def test_covariance_and_weights() -> None:
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("ABCD"))
    for method in ("sample", "ridge_epsilon", "ledoit_wolf", "gmv"):
        cov = estimate_covariance(returns, method)
        assert cov.shape == (4, 4)
        weights = minimum_variance_weights(cov, 0.35)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert (weights >= -1e-9).all() and (weights <= 0.35 + 1e-6).all()
    single = estimate_covariance(returns[["A"]], "sample")
    assert abs(minimum_variance_weights(single).iloc[0] - 1.0) < 1e-9


def test_lots_and_fees() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    prices = pd.Series({"A": 1000.0, "B": 500.0})
    shares = target_shares(weights, prices, 1_000_000.0, lot=100)
    assert (shares % 100 == 0).all()
    cash_flow, fee = transaction_value(
        pd.Series({"A": 0.0, "B": 0.0}),
        shares,
        prices,
        buy_fee=0.0019,
        sell_fee=0.0029,
    )
    assert cash_flow < 0 and fee > 0


def _synthetic_setup(n_days: int = 170, seed: int = 2):
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    str_dates = [d.strftime("%Y-%m-%d") for d in dates]
    rng = np.random.default_rng(seed)
    tickers = ["A.JK", "B.JK", "C.JK", "D.JK", "E.JK", "F.JK"]
    prices = pd.DataFrame(
        100 * np.exp(rng.normal(scale=0.01, size=(n_days, 6)).cumsum(axis=0)),
        index=dates,
        columns=tickers,
    )
    return dates, str_dates, tickers, prices, rng


def test_small_backtest() -> None:
    dates, str_dates, tickers, prices, rng = _synthetic_setup()
    rows = [
        {"date": t, "ticker": c, "pred_ens": float(rng.normal())}
        for t in str_dates
        for c in tickers
    ]
    predictions = pd.DataFrame(rows)
    weights, values = run_backtest(
        predictions,
        prices,
        k=3,
        estimator="ridge_epsilon",
        lookback=60,
        max_weight=0.5,
        capital=10_000_000.0,
    )
    assert len(weights) > 0 and len(values) > 0
    assert abs(weights.groupby("date")["weight"].sum().sub(1.0).abs().max()) < 1e-6
    assert (values["equity"] > 0).all()
    assert rebalance_dates(str_dates[:42]) == str_dates[:42][::21]


def test_execution_lag_and_mixed_roles() -> None:
    dates, str_dates, tickers, prices, rng = _synthetic_setup(n_days=90, seed=3)
    rows = []
    for i, t in enumerate(str_dates):
        for c in tickers:
            rows.append(
                {
                    "date": t,
                    "ticker": c,
                    "pred_ens": float(rng.normal()),
                    "role": "validation" if i < 5 else "test",
                }
            )
    predictions = pd.DataFrame(rows)
    weights, values = run_backtest(
        predictions,
        prices,
        k=2,
        estimator="sample",
        lookback=30,
        max_weight=0.6,
        capital=10_000_000.0,
    )
    # Mixed roles are all processed: stage 4 keeps every out-of-sample row.
    assert len(weights) > 0
    assert set(weights["date"]).issubset(set(str_dates))
    # The trade executes strictly after its signal date (1-day lag).
    executed = pd.to_datetime(weights["execution_date"])
    signaled = pd.to_datetime(weights["date"])
    assert (executed > signaled).all()
    assert (values["equity"] > 0).all()


def test_optimizer_determinism() -> None:
    rng = np.random.default_rng(7)
    tickers = list("ABCDE")
    returns = pd.DataFrame(rng.normal(size=(150, 5)), index=None, columns=tickers)
    cov = estimate_covariance(returns, "ledoit_wolf")
    mu = pd.Series(rng.normal(0.001, 0.002, size=5), index=tickers)
    target = float(mu.mean())
    a = mean_variance_target_return_weights(cov, mu, target, 0.35)
    b = mean_variance_target_return_weights(cov, mu, target, 0.35)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert abs(a.sum() - 1.0) < 1e-6
    # Target outside the feasible range -> GMV fallback.
    far = mean_variance_target_return_weights(cov, mu, 10.0, 0.35)
    gmv = minimum_variance_weights(cov, 0.35)
    assert np.allclose(far.to_numpy(), gmv.to_numpy())


def main() -> int:
    test_ensemble_and_topk()
    test_covariance_and_weights()
    test_lots_and_fees()
    test_small_backtest()
    test_execution_lag_and_mixed_roles()
    test_optimizer_determinism()
    print("test_portfolio.py: 6 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
