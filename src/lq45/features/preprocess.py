"""Robust preprocessing: winsorization followed by min-max scaling.

Following Sebastian & Tantia (2024), data pre-processing section: outliers
are clipped using the boxplot rule `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`, then the
data is scaled with min-max.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RobustPreprocessor:
    """Clip outliers, then scale each column to a fixed range.

    Parameters are computed on the training data and reused for other data,
    so there is no leakage of future information.
    """

    iqr_multiplier: float = 1.5
    feature_range: tuple[float, float] = (0.0, 1.0)
    lower_: pd.Series = field(init=False, repr=False)
    upper_: pd.Series = field(init=False, repr=False)
    data_min_: pd.Series = field(init=False, repr=False)
    data_max_: pd.Series = field(init=False, repr=False)

    def fit(self, frame: pd.DataFrame) -> RobustPreprocessor:
        """Compute the clipping bounds and scaling range from training data."""
        q1 = frame.quantile(0.25)
        q3 = frame.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.iqr_multiplier * iqr
        self.upper_ = q3 + self.iqr_multiplier * iqr
        clipped = frame.clip(lower=self.lower_, upper=self.upper_, axis=1)
        self.data_min_ = clipped.min()
        self.data_max_ = clipped.max()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the bounds and scaling already computed on other data."""
        clipped = frame.clip(lower=self.lower_, upper=self.upper_, axis=1)
        span = (self.data_max_ - self.data_min_).replace(0.0, 1.0)
        low, high = self.feature_range
        return (clipped - self.data_min_) / span * (high - low) + low

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute parameters from `frame` and apply them immediately."""
        return self.fit(frame).transform(frame)

    def inverse_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Map scaled data back to the original space (inverse of `transform`).

        Values that were clipped at the winsorization bounds cannot be
        restored exactly; that limitation is accepted for this use case.
        """
        span = (self.data_max_ - self.data_min_).replace(0.0, 1.0)
        low, high = self.feature_range
        original = (frame - low) / (high - low) * span + self.data_min_
        return original
