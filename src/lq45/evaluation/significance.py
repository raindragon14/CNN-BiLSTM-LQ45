"""Uji signifikansi Sharpe dan koreksi pengujian ganda.

Dasar (rincian: docs/keputusan_desain.md):
- Uji Sharpe Ledoit & Wolf (2008): mengokohkan Jobson & Korkie (1981)
  terhadap non-normalitas dan dependensi deret waktu lewat HAC (E15).
- Romano & Wolf (2005): stepdown untuk data snooping; dipakai bersama
  PBO Bailey et al. (2016) sebagai kontrol overfitting (E16).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def kovarians_newey_west(deret: np.ndarray, jeda: int | None = None) -> float:
    """Varians jangka panjang HAC dengan bobot Bartlett.

    Jeda baku mengikuti aturan 4*(n/100)^(2/9) dari literatur deret waktu.
    """
    n = len(deret)
    if n < 2:
        return 0.0
    if jeda is None:
        jeda = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
    tengah = deret - deret.mean()
    kov = float(np.dot(tengah, tengah) / n)
    for h in range(1, min(jeda, n - 1) + 1):
        bobot = 1.0 - h / (jeda + 1.0)
        kov += 2.0 * bobot * float(np.dot(tengah[h:], tengah[:-h]) / n)
    return max(kov, 0.0)


def uji_beda_sharpe(
    imbal_a: pd.Series, imbal_b: pd.Series, tahun: int = 252
) -> dict[str, float]:
    """Uji beda Sharpe tahunan A lawan B dengan galat baku HAC.

    Mengembalikan beda Sharpe, statistik-t, dan nilai-p dua sisi.
    """
    gabung = pd.concat([imbal_a, imbal_b], axis=1, join="inner").dropna()
    if len(gabung) < 30:
        return {"delta_sharpe": 0.0, "t_stat": 0.0, "p_value": 1.0}
    selisih = gabung.iloc[:, 0].to_numpy(dtype=float) - gabung.iloc[:, 1].to_numpy(
        dtype=float
    )
    vol = float(np.std(selisih, ddof=1))
    if vol <= 0:
        return {"delta_sharpe": 0.0, "t_stat": 0.0, "p_value": 1.0}
    delta = float(np.sqrt(tahun) * selisih.mean() / vol)
    galat = float(np.sqrt(kovarians_newey_west(selisih) / len(selisih)))
    if galat <= 0:
        return {"delta_sharpe": delta, "t_stat": 0.0, "p_value": 1.0}
    t_stat = float(selisih.mean() / galat)
    nilai_p = float(2.0 * norm.sf(abs(t_stat)))
    return {"delta_sharpe": delta, "t_stat": t_stat, "p_value": nilai_p}


def romano_wolf_stepdown(
    matriks_imbal: pd.DataFrame,
    acuan: pd.Series,
    ulang: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    tahun: int = 252,
) -> pd.DataFrame:
    """Koreksi Romano-Wolf untuk Sharpe tiap strategi lawan acuan.

    Bootstrap blok sirkular (panjang blok = 21 hari) membangun sebaran
    nol bersama; hipotesis ditolak bertahap dari statistik terbesar
    (Romano & Wolf 2005, Algoritma stepdown).
    """
    rng = np.random.default_rng(seed)
    gabung = matriks_imbal.join(acuan.rename("acuan"), how="inner").dropna()
    nama = list(matriks_imbal.columns)
    blok = 21
    n = len(gabung)
    selisih = {
        c: gabung[c].to_numpy(dtype=float) - gabung["acuan"].to_numpy(dtype=float)
        for c in nama
    }
    vol = {c: float(np.std(selisih[c], ddof=1)) for c in nama}
    stat = {
        c: float(np.sqrt(tahun) * selisih[c].mean() / vol[c]) if vol[c] > 0 else 0.0
        for c in nama
    }
    # Sebaran nol: pusatkan tiap selisih lalu bootstrap maksimum.
    maks_nol: list[float] = []
    for _ in range(ulang):
        awal = rng.integers(0, n, size=int(np.ceil(n / blok)))
        idx = np.concatenate([(awal + np.arange(blok)) % n for awal in awal])[:n]
        maks = -np.inf
        for c in nama:
            contoh = selisih[c][idx] - selisih[c].mean()
            se = float(np.std(contoh, ddof=1) / np.sqrt(n))
            t = float(np.sqrt(tahun) * contoh.mean() / se) if se > 0 else 0.0
            maks = max(maks, t)
        maks_nol.append(maks)
    ambang = float(np.quantile(maks_nol, 1.0 - alpha))
    urut = sorted(nama, key=lambda c: stat[c], reverse=True)
    baris = [
        {
            "strategy": c,
            "sharpe_diff": stat[c],
            "critical_value": ambang,
            "reject": bool(stat[c] > ambang),
        }
        for c in urut
    ]
    return pd.DataFrame(baris)
