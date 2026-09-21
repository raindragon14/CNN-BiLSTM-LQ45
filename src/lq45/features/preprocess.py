"""Praproses robust: winsorization lalu min-max.

Mengikuti Sebastian & Tantia (2024) bagian data pre-processing: outlier
dicapit memakai aturan boxplot `[Q1 - 1,5*IQR, Q3 + 1,5*IQR]`, lalu data
diskalakan dengan min-max.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RobustPreprocessor:
    """Capit outlier lalu skala tiap kolom ke rentang tetap.

    Parameter dihitung pada data latih dan dipakai ulang untuk data lain,
    sehingga tidak ada kebocoran informasi ke depan.
    """

    iqr_multiplier: float = 1.5
    feature_range: tuple[float, float] = (0.0, 1.0)
    lower_: pd.Series = field(init=False, repr=False)
    upper_: pd.Series = field(init=False, repr=False)
    data_min_: pd.Series = field(init=False, repr=False)
    data_max_: pd.Series = field(init=False, repr=False)

    def fit(self, frame: pd.DataFrame) -> RobustPreprocessor:
        """Hitung batas capit dan rentang skala dari data latih."""
        q1 = frame.quantile(0.25)
        q3 = frame.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.iqr_multiplier * iqr
        self.upper_ = q3 + self.iqr_multiplier * iqr
        dicapit = frame.clip(lower=self.lower_, upper=self.upper_, axis=1)
        self.data_min_ = dicapit.min()
        self.data_max_ = dicapit.max()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Terapkan batas dan skala yang sudah dihitung pada data lain."""
        dicapit = frame.clip(lower=self.lower_, upper=self.upper_, axis=1)
        rentang = (self.data_max_ - self.data_min_).replace(0.0, 1.0)
        rendah, tinggi = self.feature_range
        return (dicapit - self.data_min_) / rentang * (tinggi - rendah) + rendah

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Hitung parameter dari `frame` lalu langsung menerapkannya."""
        return self.fit(frame).transform(frame)

    def inverse_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Kembalikan data terskala ke ruang asal (invers `transform`).

        Nilai yang pernah dicapit pada batas winsorization tidak dapat
        dipulihkan persis; keterbatasan itu diterima untuk pemakaian ini.
        """
        rentang = (self.data_max_ - self.data_min_).replace(0.0, 1.0)
        rendah, tinggi = self.feature_range
        asli = (frame - rendah) / (tinggi - rendah) * rentang + self.data_min_
        return asli
