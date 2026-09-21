"""Metrik evaluasi portofolio: risiko, imbal hasil, dan biaya implisit.

Dasar (rincian: docs/keputusan_desain.md):
- Sharpe, Sortino, Calmar, MDD, turnover: Malhotra et al. (2023)
  memakai Sharpe/Sortino/Omega sebagai standar reksa dana; Wang & Liu
  (2025) mendefinisikan evaluasi risk-sensitive (Sharpe/Sortino/Calmar).
- Risk-free harian = BI-7DRRR tahunan dibagi 252 (E8, sumber resmi BI).
- Annualisasi 252 hari bursa (E9, konvensi standar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mdd_dan_durasi(ekuitas: pd.Series) -> tuple[float, int]:
    """Maximum drawdown (pecahan) dan durasi hari dari puncak ke lembah."""
    puncak = ekuitas.cummax()
    tarik = (ekuitas - puncak) / puncak.replace(0.0, np.nan)
    tarik = tarik.fillna(0.0)
    mdd = float(tarik.min())
    posisi = int(tarik.to_numpy().argmin()) if len(tarik) else 0
    puncak_posisi = int(puncak.to_numpy()[: posisi + 1].argmax()) if len(tarik) else 0
    return mdd, max(0, posisi - puncak_posisi)


def ringkas_deret(
    imbal: pd.Series,
    bebas_risiko: pd.Series | float = 0.0,
    tahun: int = 252,
) -> dict[str, float]:
    """Ringkas deret imbal hasil harian menjadi metrik tahunan.

    `bebas_risiko` deret harian atau angka tetap. Sortino memakai
    target nol pada sisi turun. Calmar memakai return tahunan dibagi
    |MDD| (nol bila MDD nol).
    """
    r = imbal.to_numpy(dtype=float)
    if isinstance(bebas_risiko, pd.Series):
        rf = bebas_risiko.reindex(imbal.index).fillna(0.0).to_numpy(dtype=float)
    else:
        rf = np.full_like(r, float(bebas_risiko))
    lebih = r - rf
    n = len(r)
    kum = float(np.prod(1.0 + r) - 1.0) if n else 0.0
    tahunan = float((1.0 + kum) ** (tahun / n) - 1.0) if n else 0.0
    vol = float(np.std(lebih, ddof=1)) if n > 1 else 0.0
    sharpe = float(np.sqrt(tahun) * lebih.mean() / vol) if vol > 0 else 0.0
    turun = lebih[lebih < 0.0]
    vol_turun = float(np.std(turun, ddof=1)) if len(turun) > 1 else 0.0
    sortino = float(np.sqrt(tahun) * lebih.mean() / vol_turun) if vol_turun > 0 else 0.0
    ekuitas = pd.Series(np.cumprod(1.0 + r), index=imbal.index)
    mdd, _ = mdd_dan_durasi(ekuitas)
    calmar = float(tahunan / abs(mdd)) if mdd < 0 else 0.0
    return {
        "cumulative_return": kum,
        "annual_return": tahunan,
        "annual_vol": float(vol * np.sqrt(tahun)) if n > 1 else 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "n_days": float(n),
    }


def bebas_risiko_harian(
    tanggal: pd.DatetimeIndex, berkas_suku: str, tahun: int = 252
) -> pd.Series:
    """Deret risk-free harian dari riwayat keputusan BI-7DRRR.

    Berkas memakai kolom `date, rate` (persen tahunan pada tanggal
    keputusan); nilai diteruskan ke depan lalu dibagi 100 dan 252.
    """
    suku = pd.read_csv(berkas_suku, parse_dates=["date"]).set_index("date")
    suku.index = pd.to_datetime(suku.index)
    harian = suku["rate"].reindex(tanggal, method="ffill").bfill()
    return (harian / 100.0 / tahun).astype(float)
