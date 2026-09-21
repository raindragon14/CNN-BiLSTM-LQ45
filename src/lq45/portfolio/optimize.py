"""Optimasi bobot varians minimum dan mean-variance dengan kendala ritel IDX.

Dasar (rincian: docs/keputusan_desain.md):
- Kendala long-only dan jumlah bobot satu: Markowitz (1952); short
  tidak diperbolehkan pada perdagangan ritel IDX (E10).
- Batas bobot 35% keputusan mandiri; diuji 25%, 35%, 50%, 100% (E2, E17).
- Penalti turnover nol karena biaya disimulasikan eksplisit (E12).
- Formulasi target-return constrained: Chaweewanchon & Chaysiri (2022)
  Section 3.1 Persamaan (1)-(4) — min w'Σw s.t. w'μ=γ, Σw=1, 0≤w≤maxw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def bobot_varians_minimum(kov: pd.DataFrame, bobot_maks: float = 0.35) -> pd.Series:
    """Bobot varians minimum: min w'Sw dengan 0 <= w <= maks, sum w = 1.

    Bila optimasi gagal, kembalikan bobot sama rata agar pipa tidak
    berhenti; kegagalan seperti itu dicatat pemanggil.
    """
    tickers = list(kov.columns)
    n = len(tickers)
    if n == 0:
        raise ValueError("daftar ticker kosong")
    if n == 1:
        return pd.Series([1.0], index=tickers)
    matriks = kov.to_numpy(dtype=float)
    batas = [(0.0, float(bobot_maks))] * n
    samakan = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    awal = np.full(n, 1.0 / n)
    hasil = minimize(
        lambda w: float(w @ matriks @ w),
        awal,
        method="SLSQP",
        bounds=batas,
        constraints=[samakan],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not hasil.success:
        return pd.Series(awal, index=tickers)
    bobot = np.clip(hasil.x, 0.0, None)
    total = bobot.sum()
    if total <= 0:
        return pd.Series(awal, index=tickers)
    return pd.Series(bobot / total, index=tickers)


def bobot_mean_varians_target_return(
    kov: pd.DataFrame,
    expected_returns: pd.Series,
    target_return: float,
    bobot_maks: float = 0.35,
) -> pd.Series:
    """Bobot mean-variance: min w'Sw s.t. w'μ=γ, Σw=1, 0<=w<=maks.

    Formulasi Chaweewanchon & Chaysiri (2022) Section 3.1 Persamaan (1)-(4):
    Minimize σ² = ∑ᵢ∑ⱼ wᵢwⱼCᵢⱼ
    subject to ∑ᵢ wᵢEᵢ = γ, ∑ᵢ wᵢ = 1, wᵢ ≥ 0.

    Bila optimasi gagal atau target_return tidak feasible, kembalikan
    bobot GMV (varians minimum) sebagai fallback.
    """
    tickers = list(kov.columns)
    n = len(tickers)
    if n == 0:
        raise ValueError("daftar ticker kosong")
    if n == 1:
        return pd.Series([1.0], index=tickers)

    matriks = kov.to_numpy(dtype=float)
    mu = expected_returns.reindex(tickers).fillna(0.0).to_numpy(dtype=float)

    # Cek feasibility: apakah target_return achievable
    # Dengan 0<=w<=maks dan sum(w)=1, return range:
    # min_w'μ <= γ <= max_w'μ
    mu_min = mu.min() if n > 0 else 0.0
    mu_max = mu.max() if n > 0 else 0.0
    if target_return < mu_min or target_return > mu_max:
        return bobot_varians_minimum(kov, bobot_maks)

    batas = [(0.0, float(bobot_maks))] * n
    batas_sum = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    batas_target = {"type": "eq", "fun": lambda w: float(w @ mu - target_return)}

    # Coba beberapa starting point jika yang pertama gagal
    starting_points = [
        np.full(n, 1.0 / n),
        np.random.dirichlet(np.ones(n)),
    ]
    if n >= 3:
        starting_points.append(
            np.concatenate(
                [
                    [target_return / mu_max if mu_max > 0 else 1 / n],
                    np.ones(n - 1)
                    * (1 - target_return / mu_max if mu_max > 0 else 1 / n)
                    / (n - 1),
                ]
            )
        )

    for sp in starting_points:
        sp = np.clip(sp, 0, bobot_maks)
        sp = sp / sp.sum() if sp.sum() > 0 else np.full(n, 1.0 / n)
        hasil = minimize(
            lambda w: float(w @ matriks @ w),
            sp,
            method="SLSQP",
            bounds=batas,
            constraints=[batas_sum, batas_target],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if hasil.success:
            bobot = np.clip(hasil.x, 0.0, None)
            total = bobot.sum()
            if total > 0 and abs(float(bobot @ mu - target_return)) < 1e-4:
                return pd.Series(bobot / total, index=tickers)

    # Fallback ke GMV
    return bobot_varians_minimum(kov, bobot_maks)
