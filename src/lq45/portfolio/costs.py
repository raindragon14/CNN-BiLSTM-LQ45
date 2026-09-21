"""Simulasi biaya transaksi ritel Indonesia dengan lot 100 lembar.

Dasar (rincian: docs/keputusan_desain.md):
- Fee beli 0,19%, fee jual 0,29%, lot 100 lembar mengikuti ketentuan
  sekuritas ritel Indonesia (A8).
- Rebalancing bulanan 21 hari bursa: Espiga-Fernandez et al. (2024)
  memakai rebalancing periodik yang hemat biaya; Huang et al. (2024)
  memakai horizon bulanan (E4).
"""

from __future__ import annotations

import math

import pandas as pd


def target_lembar(
    bobot: pd.Series,
    harga: pd.Series,
    ekuitas: float,
    lot: int = 100,
) -> pd.Series:
    """Jumlah lembar target per saham dibulatkan ke bawah ke kelipatan lot.

    Dana yang tidak cukup untuk satu lot menjadi kas.
    """
    lembar: dict[str, int] = {}
    for ticker, w in bobot.items():
        p = float(harga.get(ticker, float("nan")))
        if not math.isfinite(p) or p <= 0 or w <= 0:
            lembar[ticker] = 0
            continue
        unit = int((float(w) * ekuitas) // (p * lot))
        lembar[ticker] = unit * lot
    return pd.Series(lembar, dtype=float)


def nilai_transaksi(
    lama: pd.Series,
    baru: pd.Series,
    harga: pd.Series,
    fee_beli: float = 0.0019,
    fee_jual: float = 0.0029,
) -> tuple[float, float]:
    """Nilai jual-beli dan total fee dari perubahan posisi.

    Mengembalikan (arus_kas_bersih, fee). Arus kas bersih positif
    berarti penjualan melebihi pembelian sebelum fee.
    """
    semua = set(lama.index) | set(baru.index) | set(harga.index)
    beli = 0.0
    jual = 0.0
    for ticker in semua:
        p = float(harga.get(ticker, 0.0) or 0.0)
        delta = float(baru.get(ticker, 0.0)) - float(lama.get(ticker, 0.0))
        if delta > 0:
            beli += delta * p
        elif delta < 0:
            jual += -delta * p
    fee = beli * fee_beli + jual * fee_jual
    return jual - beli, fee
