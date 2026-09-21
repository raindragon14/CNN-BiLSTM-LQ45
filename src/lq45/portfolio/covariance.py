"""Estimator kovarians untuk optimasi varians minimum.

Dasar (rincian: docs/keputusan_desain.md):
- Perbandingan empat estimator, bukan asumsi satu pemenang:
  DeMiguel et al. (2009) menunjukkan 1/N sulit dikalahkan sehingga
  klaim keunggulan estimator harus diuji; Ledoit & Wolf (2004)
  mengusulkan penyusutan untuk kovarians berdimensi besar.
- Ridge epsilon = 1e-4 menstabilkan diagonal (E11).
- Jendela L = 120 hari bursa; sensitivitas 60 dan 252 (E17).
- Lebel `gmv` memakai kovarians sampel seperti Huang et al. (2024)
  yang memakai Global Minimum Variance pada tahap kedua; hasilnya
  sama dengan `sample` pada kendala yang sama dan dipertahankan
  sebagai pembanding bernama.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def matriks_imbal_hasil(harga: pd.DataFrame, akhir: str, hari: int) -> pd.DataFrame:
    """Imbal hasil log harian untuk jendela `(akhir-hari, akhir]`.

    `harga` berindeks tanggal dengan kolom per ticker (Adj Close).
    Hanya memakai data sampai tanggal `akhir` agar tidak melihat
    masa depan. Baris ber-NaN dibuang per kolom nanti oleh pemanggil.
    """
    potong = harga[harga.index <= akhir].tail(hari + 1)
    imbal = np.log(potong / potong.shift(1)).iloc[1:]
    return imbal


def taksir_kovarians(
    imbal: pd.DataFrame, metode: str, ridge_epsilon: float = 1e-4
) -> pd.DataFrame:
    """Taksir matriks kovarians harian dari imbal hasil.

    Metode: `sample`, `ridge_epsilon`, `ledoit_wolf`, `gmv`.
    `gmv` memakai kovarians sampel (sama dengan `sample`).
    """
    bersih = imbal.dropna(axis=1)
    if bersih.shape[1] == 0:
        raise ValueError("tidak ada kolom imbal hasil yang lengkap")
    if metode in ("sample", "gmv"):
        return bersih.cov()
    if metode == "ridge_epsilon":
        kov = bersih.cov().to_numpy()
        kov = kov + ridge_epsilon * np.eye(kov.shape[0])
        return pd.DataFrame(kov, index=bersih.columns, columns=bersih.columns)
    if metode == "ledoit_wolf":
        lw = LedoitWolf().fit(bersih.to_numpy())
        return pd.DataFrame(
            lw.covariance_, index=bersih.columns, columns=bersih.columns
        )
    raise ValueError(f"metode kovarians tidak dikenal: {metode}")
