"""Generator lipatan walk-forward bersarang dengan purge dan embargo.

Definisi operasional mengikuti Lopez de Prado (2018) Bab 7:
- Purge: sampel latih tanggal `t` memakai label `t + tau`; label tidak
  boleh jatuh pada periode evaluasi, sehingga jendela latih dipotong
  menjadi `[0, awal_validasi - tau)`.
- Embargo: setiap periode test lampau melahirkan larangan sepanjang
  `lookback - 1` hari setelah test berakhir; bila jendela latih melebar
  menyerap daerah itu, sampel di dalamnya dibuang.

Semua batas memakai posisi (indeks) kalender bursa, bukan tanggal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 252


@dataclass
class Fold:
    """Satu lipatan walk-forward; batas berupa indeks kalender."""

    id: int
    train: tuple[int, int]  # [awal, akhir); purge sudah dipotong
    validation: tuple[int, int]
    test: tuple[int, int]
    banned: list[tuple[int, int]]  # embargo dari test lampau


@dataclass
class DesignSplit:
    """Pembagian periode desain: latih lalu validasi (tanpa test)."""

    train: tuple[int, int]
    validation: tuple[int, int]
    banned: list[tuple[int, int]] = field(default_factory=list)


def make_folds(
    n_days: int,
    split_cfg: dict[str, Any],
    purge_days: int | None = None,
    embargo_days: int | None = None,
) -> list[Fold]:
    """Bangun seluruh lipatan dari kalender sepanjang `n_days` hari.

    Bila `purge_days` dan `embargo_days` tidak diberikan, nilainya diambil
    dari `split_cfg` (baku 5 dan 59). Lipatan dibentuk selama periode test
    masih muat di dalam kalender.
    """
    train_awal = int(split_cfg["initial_train_years"]) * DAYS_PER_YEAR
    val = int(split_cfg["validation_days"])
    test = int(split_cfg["test_days"])
    step = int(split_cfg["step_days"])
    if purge_days is None:
        purge_days = int(split_cfg["purge"]["horizon_days"])
    if embargo_days is None:
        embargo_days = int(split_cfg["embargo_days"])

    lipatan: list[Fold] = []
    akhir_test_terdahulu: list[int] = []
    k = 0
    while True:
        awal_val = train_awal + k * step
        awal_test = awal_val + val
        akhir_test = awal_test + test
        if akhir_test > n_days:
            break
        train = (0, max(0, awal_val - purge_days))
        banned = [
            (akhir, min(akhir + embargo_days, n_days)) for akhir in akhir_test_terdahulu
        ]
        lipatan.append(
            Fold(
                id=k,
                train=train,
                validation=(awal_val, awal_test),
                test=(awal_test, akhir_test),
                banned=banned,
            )
        )
        akhir_test_terdahulu.append(akhir_test)
        k += 1
    return lipatan


def design_split(
    dates: pd.DatetimeIndex,
    train_range: str,
    validate_range: str,
    horizon: int,
) -> DesignSplit:
    """Pembagian periode desain dari rentang `YYYY-MM-DD/YYYY-MM-DD`.

    Purge memotong `horizon` hari terakhir latih agar label tidak jatuh
    pada periode validasi.
    """
    t0, t1 = (pd.Timestamp(batas) for batas in train_range.split("/"))
    v0, v1 = (pd.Timestamp(batas) for batas in validate_range.split("/"))

    posisi_latih = np.flatnonzero((dates >= t0) & (dates <= t1))
    posisi_validasi = np.flatnonzero((dates >= v0) & (dates <= v1))
    if len(posisi_latih) == 0 or len(posisi_validasi) == 0:
        raise ValueError(
            f"rentang desain kosong: latih {len(posisi_latih)} hari, "
            f"validasi {len(posisi_validasi)} hari"
        )
    train = (int(posisi_latih[0]), int(posisi_latih[-1]) + 1 - horizon)
    validation = (int(posisi_validasi[0]), int(posisi_validasi[-1]) + 1)
    return DesignSplit(train=train, validation=validation)


@dataclass
class PretrainSplit:
    """Pembagian untuk pre-training: semua data split train/val saja."""

    train: tuple[int, int]
    validation: tuple[int, int]


def make_pretrain_split(
    n_days: int,
    val_ratio: float = 0.1,
) -> PretrainSplit:
    """Split sederhana untuk pre-training: train + val (tidak ada test).

    Pre-training menggunakan SEMUA data (tidak ada purge/embargo karena
    tidak ada label yang bocor). Split berdasarkan ratio.

    Args:
        n_days: total panjang kalender
        val_ratio: fraksi untuk validasi (default 10%)

    Returns:
        PretrainSplit dengan train/val indices
    """
    split_idx = int(n_days * (1 - val_ratio))
    return PretrainSplit(
        train=(0, split_idx),
        validation=(split_idx, n_days),
    )
