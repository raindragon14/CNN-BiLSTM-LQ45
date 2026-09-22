"""Evaluation stage-5 module: metrics, DSR/PBO, significance, regimes."""

from lq45.evaluation.dsr_pbo import annualized_sharpe, deflated_sharpe, pbo_cscv
from lq45.evaluation.metrics import (
    daily_risk_free_rate,
    max_drawdown_and_duration,
    summarize_series,
)
from lq45.evaluation.regimes import split_regimes
from lq45.evaluation.significance import (
    newey_west_covariance,
    romano_wolf_stepdown,
    sharpe_difference_test,
)

__all__ = [
    "annualized_sharpe",
    "daily_risk_free_rate",
    "deflated_sharpe",
    "max_drawdown_and_duration",
    "newey_west_covariance",
    "pbo_cscv",
    "romano_wolf_stepdown",
    "sharpe_difference_test",
    "split_regimes",
    "summarize_series",
]
