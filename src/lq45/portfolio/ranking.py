"""Peringkat saham dari prediksi ensemble untuk praseleksi top-k.

Dasar (rincian: docs/keputusan_desain.md):
- Praseleksi lalu optimasi: Wang et al. (2020) memakai deep learning
  untuk praseleksi sebelum pembentukan portofolio; Huang et al. (2024)
  memakai dua tahap dengan presekrining sebelum Global Minimum Variance.
- Grid k = 5, 7, 10 menjawab RQ2; Chaweewanchon & Chaysiri (2022)
  menguji N = 5-10, Paiva et al. (2019) memakai 7, Wang et al. (2020)
  memakai 10.
- Rata-rata ensemble lintas seed meredam variansi pelatihan; sebaran
  dilaporkan mengikuti Reimers & Gurevych (2017) dan
  Bouthillier et al. (2021).
"""

from __future__ import annotations

import pandas as pd


def ensemble_prediksi(frame: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata pred_raw lintas seed per (tanggal, saham).

    Masukan memakai kolom `date, ticker, pred_raw` (per seed); keluaran
    satu baris per (tanggal, saham) dengan kolom `pred_ens`.
    """
    gabung = (
        frame.groupby(["date", "ticker"], as_index=False)["pred_raw"]
        .mean()
        .rename(columns={"pred_raw": "pred_ens"})
    )
    return gabung


def peringkat_topk(
    frame: pd.DataFrame, tanggal: str, k: int, kolom: str = "pred_ens"
) -> list[str]:
    """Pilih k saham berperingkat teratas pada satu tanggal.

    Saham tanpa prediksi pada tanggal itu (mis. suspensi WIKA)
       tidak ikut; bila kandidat kurang dari k, kembalikan yang ada.
    """
    potong = frame[frame["date"] == tanggal].dropna(subset=[kolom])
    potong = potong.sort_values([kolom, "ticker"], ascending=[False, True])
    return potong["ticker"].head(k).tolist()
