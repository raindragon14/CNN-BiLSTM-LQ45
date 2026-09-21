"""Pembagian rezim pasar untuk uji ketahanan strategi.

Dasar (rincian: docs/keputusan_desain.md):
- Uji rezim COVID: Huang et al. (2024) menguji ketahanan saat COVID.
- Rezim OOS: COVID 2020 dan pemulihan/kenaikan suku bunga 2021-2025;
  periode desain 2018-2019 bersifat tenang dan tidak dipakai menilai
  (E7, configs/experiment.yaml `regimes`).
"""

from __future__ import annotations

import pandas as pd


def bagi_rezim(
    frame: pd.DataFrame,
    kolom_tanggal: str,
    rezim: dict[str, dict[str, str]],
) -> dict[str, pd.DataFrame]:
    """Bagi bingkai berkolom tanggal ke tiap rezim.

    Batas memakai string `YYYY-MM-DD` inklusif dari konfigurasi.
    """
    tanggal = pd.to_datetime(frame[kolom_tanggal])
    keluar: dict[str, pd.DataFrame] = {}
    for nama, batas in rezim.items():
        awal = pd.Timestamp(batas["start"])
        akhir = pd.Timestamp(batas["end"])
        mask = (tanggal >= awal) & (tanggal <= akhir)
        keluar[nama] = frame[mask].reset_index(drop=True)
    return keluar
