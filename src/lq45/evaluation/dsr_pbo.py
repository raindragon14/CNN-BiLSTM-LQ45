"""Deflated Sharpe Ratio dan Probability of Backtest Overfitting.

Dasar (rincian: docs/keputusan_desain.md):
- DSR: Bailey & Lopez de Prado (2014) mengoreksi Sharpe terhadap
  banyaknya percobaan dan ketaknormalan (skew, kurtosis).
- PBO/CSCV: Bailey et al. (2016) mengukur peluang strategi terpilih
  overfit lewat validasi silang simetris kombinatorial.
- Keduanya menjawab E16 (kontrol backtest overfitting) bersama
  Romano & Wolf (2005).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def sharpe_tahunan(imbal: pd.Series, tahun: int = 252) -> float:
    """Sharpe tahunan dengan risk-free nol (untuk pembanding internal)."""
    r = imbal.to_numpy(dtype=float)
    vol = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    if vol <= 0:
        return 0.0
    return float(np.sqrt(tahun) * r.mean() / vol)


def sharpe_benchmark_harapan(n_uji: int, varian: float) -> float:
    """Perkiraan Sharpe acuan di bawah nol bila diuji n_uji strategi.

    Mengikuti Bailey & Lopez de Prado (2014) Bagian 3: acuan memakai
    pendekatan nilai harapan maksimum dari n_uji peubah acak.
    """
    if n_uji < 2 or varian <= 0:
        return 0.0
    gamma = 0.5772156649
    return float(
        math.sqrt(varian) * ((1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_uji))
        + gamma * norm.ppf(1.0 - 1.0 / (n_uji * math.e))
    )


def deflated_sharpe(
    imbal: pd.Series, n_uji: int, acuan: float = 0.0, tahun: int = 252
) -> dict[str, float]:
    """DSR: peluang Sharpe sejati di atas acuan setelah koreksi.

    Mengembalikan rasio Sharpe, acuan harapan, dan DSR (0-1).
    """
    r = imbal.to_numpy(dtype=float)
    n = len(r)
    sr = sharpe_tahunan(imbal, tahun)
    if n < 3:
        return {"sharpe": sr, "benchmark": acuan, "dsr": 0.5}
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis())  # ekses terhadap normal
    acuan_harapan = max(acuan, sharpe_benchmark_harapan(n_uji, 1.0 / n * tahun))
    pembilang = (sr - acuan_harapan) * math.sqrt(max(n - 1, 1))
    penyebut = math.sqrt(
        max(
            1.0 - skew * sr + (kurt / 4.0) * sr * sr,
            1e-12,
        )
    )
    return {
        "sharpe": sr,
        "benchmark": float(acuan_harapan),
        "dsr": float(norm.cdf(pembilang / penyebut)),
    }


def pbo_cscv(
    matriks_imbal: pd.DataFrame, bagian: int = 8, seed: int = 0
) -> dict[str, float]:
    """PBO lewat CSCV: peluang pilihan dalam-sampel kalah out-of-sample.

    Matriks berkolom strategi dan berbaris tanggal. Baris dibagi
    menjadi `bagian` segmen sama panjang; tiap belahan memakai separuh
    segmen sebagai dalam-sampel. Strategi terbaik dalam-sampel
    dibandingkan peringkatnya di luar-sampel; PBO = peluang logit
    negatif (Bailey et al. 2016).
    """
    rng = np.random.default_rng(seed)
    frame = matriks_imbal.dropna()
    n = len(frame)
    n_strat = len(frame.columns)
    segmen = np.array_split(np.arange(n), bagian)
    separuh = bagian // 2
    logit: list[float] = []
    for dalam in itertools.combinations(range(bagian), separuh):
        dalam = list(dalam)
        luar = [s for s in range(bagian) if s not in dalam]
        idx_dalam = np.concatenate([segmen[s] for s in dalam])
        idx_luar = np.concatenate([segmen[s] for s in luar])
        sr_dalam = frame.iloc[idx_dalam].apply(sharpe_tahunan)
        sr_luar = frame.iloc[idx_luar].apply(sharpe_tahunan)
        peringkat_luar = sr_luar.rank(ascending=False)
        terbaik = sr_dalam.idxmax()
        posisi = float(peringkat_luar[terbaik])
        # Logit positif bila terbaik dalam-sampel berada di atas
        # median luar-sampel; PBO = peluang logit negatif.
        logit.append(math.log((n_strat + 1.0 - posisi) / posisi))
    _ = rng  # seed dicatat untuk audit; pembelahan kombinatorial baku
    logit_arr = np.array(logit, dtype=float)
    return {
        "pbo": float((logit_arr < 0).mean()) if len(logit_arr) else 0.5,
        "n_splits": float(len(logit_arr)),
        "mean_logit": float(logit_arr.mean()) if len(logit_arr) else 0.0,
    }
