"""Evaluation module tests; run directly: python3 tests/test_evaluation.py.

The data here is synthetic and only validates the pipeline, not results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.evaluation.dsr_pbo import (
    annualized_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    pbo_cscv,
)
from lq45.evaluation.metrics import max_drawdown_and_duration, summarize_series
from lq45.evaluation.regimes import split_regimes
from lq45.evaluation.significance import (
    romano_wolf_stepdown,
)
from lq45.evaluation.significance import (
    sharpe_difference_test as sharpe_diff_fn,
)


def test_basic_metrics() -> None:
    rng = np.random.default_rng(3)
    idx = pd.RangeIndex(252)
    returns = pd.Series(rng.normal(0.0005, 0.01, size=252), index=idx)
    summary = summarize_series(returns)
    assert summary["n_days"] == 252
    assert -1.0 < summary["max_drawdown"] <= 0.0
    assert summary["cumulative_return"] > -1.0
    max_drawdown, duration = max_drawdown_and_duration(
        pd.Series(np.cumprod(1.0 + returns), index=idx)
    )
    assert max_drawdown <= 0.0 and duration >= 0
    assert abs(annualized_sharpe(returns)) < 5.0


def test_dsr_and_pbo() -> None:
    rng = np.random.default_rng(4)
    idx = pd.RangeIndex(500)
    returns = pd.Series(rng.normal(0.001, 0.01, size=500), index=idx)
    result = deflated_sharpe(returns, n_trials=50)
    assert 0.0 <= result["dsr"] <= 1.0
    assert result["benchmark"] >= 0.0
    assert expected_max_sharpe(1, 1.0) == 0.0
    matrix = pd.DataFrame(
        rng.normal(size=(400, 6)), columns=[f"s{i}" for i in range(6)]
    )
    pbo = pbo_cscv(matrix, n_groups=4)
    assert 0.0 <= pbo["pbo"] <= 1.0
    assert pbo["n_splits"] == 6.0


def test_significance() -> None:
    rng = np.random.default_rng(5)
    idx = pd.RangeIndex(300)
    returns_a = pd.Series(rng.normal(0.001, 0.01, size=300), index=idx)
    returns_b = pd.Series(rng.normal(0.0, 0.01, size=300), index=idx)
    difference = sharpe_diff_fn(returns_a, returns_b)
    assert 0.0 <= difference["p_value"] <= 1.0
    matrix = pd.DataFrame({"s1": returns_a, "s2": returns_b})
    table = romano_wolf_stepdown(matrix, returns_b, n_bootstrap=50, seed=0)
    assert {
        "strategy",
        "sharpe_diff",
        "critical_value",
        "reject",
        "step",
    }.issubset(table.columns)
    assert 1 <= len(table) <= 2
    # Stepdown stops at the first non-rejection -> reject decreases monotonically.
    reject = table["reject"].tolist()
    for i in range(len(reject) - 1):
        if not reject[i]:
            assert not reject[i + 1]
    # Steps are sequential starting from 1.
    assert table["step"].tolist() == list(range(1, len(table) + 1))


def test_regimes() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01", "2021-06-01", "2019-06-01"]),
            "return": [0.01, 0.02, 0.03],
        }
    )
    regimes = {
        "covid": {"start": "2020-01-01", "end": "2020-12-31"},
        "recovery_rate_hike": {"start": "2021-01-01", "end": "2025-12-31"},
    }
    parts = split_regimes(frame, "date", regimes)
    assert len(parts["covid"]) == 1 and len(parts["recovery_rate_hike"]) == 1


def main() -> int:
    test_basic_metrics()
    test_dsr_and_pbo()
    test_significance()
    test_regimes()
    print("test_evaluation.py: 4 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
