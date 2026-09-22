"""Deflated Sharpe Ratio and Probability of Backtest Overfitting.

Basis (details: docs/keputusan_desain.md):
- DSR: Bailey & Lopez de Prado (2014) correct the Sharpe ratio for the
  number of trials and non-normality (skew, kurtosis).
- PBO/CSCV: Bailey et al. (2016) measure the probability that the
  selected strategy is overfit via combinatorial symmetric
  cross-validation.
- Both answer E16 (backtest overfitting control) together with
  Romano & Wolf (2005).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe with a zero risk-free rate (internal comparison)."""
    r = returns.to_numpy(dtype=float)
    vol = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    if vol <= 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / vol)


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """Expected benchmark Sharpe under the null when n_trials strategies are tested.

    Follows Bailey & Lopez de Prado (2014) Section 3: the benchmark uses
    the expected maximum of n_trials random variables.
    """
    if n_trials < 2 or variance <= 0:
        return 0.0
    gamma = 0.5772156649
    return float(
        math.sqrt(variance) * ((1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials))
        + gamma * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(
    returns: pd.Series,
    n_trials: int,
    benchmark: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """DSR: probability the true Sharpe exceeds the benchmark after correction.

    Returns the Sharpe ratio, the expected benchmark, and the DSR (0-1).
    """
    r = returns.to_numpy(dtype=float)
    n = len(r)
    sr = annualized_sharpe(returns, periods_per_year)
    if n < 3:
        return {"sharpe": sr, "benchmark": benchmark, "dsr": 0.5}
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis())  # excess over normal
    benchmark_expectation = max(
        benchmark, expected_max_sharpe(n_trials, 1.0 / n * periods_per_year)
    )
    numerator = (sr - benchmark_expectation) * math.sqrt(max(n - 1, 1))
    denominator = math.sqrt(
        max(
            1.0 - skew * sr + (kurt / 4.0) * sr * sr,
            1e-12,
        )
    )
    return {
        "sharpe": sr,
        "benchmark": float(benchmark_expectation),
        "dsr": float(norm.cdf(numerator / denominator)),
    }


def pbo_cscv(
    returns_matrix: pd.DataFrame, n_groups: int = 8, seed: int = 0
) -> dict[str, float]:
    """PBO via CSCV: probability the in-sample choice loses out-of-sample.

    The matrix has strategies as columns and dates as rows. Rows are
    split into `n_groups` equal-length segments; each fold uses half the
    segments as in-sample. The best in-sample strategy is compared with
    its out-of-sample rank; PBO = probability of a negative logit
    (Bailey et al. 2016).
    """
    rng = np.random.default_rng(seed)
    frame = returns_matrix.dropna()
    n = len(frame)
    n_strat = len(frame.columns)
    segments = np.array_split(np.arange(n), n_groups)
    half = n_groups // 2
    logit: list[float] = []
    for in_groups in itertools.combinations(range(n_groups), half):
        in_groups = list(in_groups)
        out_groups = [s for s in range(n_groups) if s not in in_groups]
        in_idx = np.concatenate([segments[s] for s in in_groups])
        out_idx = np.concatenate([segments[s] for s in out_groups])
        sr_in = frame.iloc[in_idx].apply(annualized_sharpe)
        sr_out = frame.iloc[out_idx].apply(annualized_sharpe)
        out_ranks = sr_out.rank(ascending=False)
        best = sr_in.idxmax()
        position = float(out_ranks[best])
        # Positive logit when the in-sample best ranks above the
        # out-of-sample median; PBO = probability of a negative logit.
        logit.append(math.log((n_strat + 1.0 - position) / position))
    _ = rng  # seed recorded for audit; the combinatorial split is standard
    logit_arr = np.array(logit, dtype=float)
    return {
        "pbo": float((logit_arr < 0).mean()) if len(logit_arr) else 0.5,
        "n_splits": float(len(logit_arr)),
        "mean_logit": float(logit_arr.mean()) if len(logit_arr) else 0.0,
    }
