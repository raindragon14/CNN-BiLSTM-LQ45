"""Mesin backtest bulanan: peringkat, optimasi, biaya, penahanan.

Alur tiap tanggal rebalancing `d` (semua memakai informasi sampai `d`):
1. Peringkat ensemble pred_raw tanggal `d`, ambil top-k (saham tanpa prediksi dikecualikan karena suspensi).
2. Taksir kovarians dari L imbal hasil log harian sampai `d`.
3. Optimasi varians minimum atau mean-variance dengan kendala long-only, sum satu,
   bobot maksimum.
4. Bulatkan ke lot 100, potong fee beli 0,19% dan jual 0,29%.
5. Tahan sampai rebalancing berikut; nilai harian mengikuti harga.

Tanggal rebalancing memakai kalender tanggal prediksi `role=test`
setiap 21 hari agar selaras dengan jendela test tahap 3.
"""

from __future__ import annotations

import pandas as pd

from lq45.portfolio.costs import nilai_transaksi, target_lembar
from lq45.portfolio.covariance import (
    matriks_imbal_hasil,
    taksir_kovarians,
)
from lq45.portfolio.optimize import (
    bobot_mean_varians_target_return,
    bobot_varians_minimum,
)
from lq45.portfolio.ranking import peringkat_topk


def tanggal_rebalancing(tanggal_oos: list[str], langkah: int = 21) -> list[str]:
    """Ambil setiap tanggal ke-`langkah` dari kalender OOS terurut."""
    urut = sorted(set(tanggal_oos))
    return urut[::langkah]


def _target_return_dari_prediksi(
    prediksi_ens: pd.DataFrame,
    tanggal: str,
    pilihan: list[str],
) -> float:
    """Hitung target return rata-rata pred_ens untuk saham terpilih."""
    baris = prediksi_ens[
        (prediksi_ens["date"] == tanggal) & (prediksi_ens["ticker"].isin(pilihan))
    ]
    if baris.empty:
        return 0.0
    return float(baris["pred_ens"].mean())


def jalankan_backtest(
    prediksi_ens: pd.DataFrame,
    harga: pd.DataFrame,
    k: int,
    estimator: str,
    lihat_balik: int,
    bobot_maks: float,
    modal: float = 100_000_000.0,
    fee_beli: float = 0.0019,
    fee_jual: float = 0.0029,
    lot: int = 100,
    langkah: int = 21,
    ridge_epsilon: float = 1e-4,
    optimizer_type: str = "target_return",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest satu konfigurasi; kembalikan (bobot, nilai harian).

    `prediksi_ens` memakai kolom `date, ticker, pred_ens`; `harga`
    berindeks tanggal dengan kolom per ticker (Adj Close).
    `bobot` satu baris per tanggal rebalancing; nilai harian memakai
    kolom `date, equity, return`.
    `optimizer_type`: "target_return" (MV) atau "min_variance" (GMV).
    """
    tanggal_oos = sorted(prediksi_ens["date"].unique().tolist())
    jadwal = tanggal_rebalancing(tanggal_oos, langkah)
    indeks_harga = harga.index
    kas = float(modal)
    posisi = pd.Series(dtype=float)
    baris_bobot: list[dict] = []
    baris_nilai: list[dict] = []
    tanggal_akhir = tanggal_oos[-1]

    for i, tanggal in enumerate(jadwal):
        pilihan = peringkat_topk(prediksi_ens, tanggal, k)
        harga_d = harga.loc[harga.index <= tanggal].iloc[-1]
        ekuitas = kas + float(
            (posisi * harga_d.reindex(posisi.index).fillna(0.0)).sum()
            if len(posisi)
            else 0.0
        )
        if pilihan:
            imbal = matriks_imbal_hasil(harga[pilihan], tanggal, lihat_balik)
            imbal = imbal.dropna(axis=1)
            pilihan = imbal.columns.tolist()
        if not pilihan:
            bobot = pd.Series(dtype=float)
        else:
            kov = taksir_kovarians(imbal, estimator, ridge_epsilon)
            if optimizer_type == "target_return":
                expected_returns = prediksi_ens[
                    prediksi_ens["date"] == tanggal
                ].set_index("ticker")["pred_ens"]
                target_ret = _target_return_dari_prediksi(
                    prediksi_ens, tanggal, pilihan
                )
                bobot = bobot_mean_varians_target_return(
                    kov, expected_returns, target_ret, bobot_maks
                )
            else:
                bobot = bobot_varians_minimum(kov, bobot_maks)
            bobot = bobot.reindex(pilihan).fillna(0.0)
        target = (
            target_lembar(bobot, harga_d, ekuitas, lot)
            if len(bobot)
            else pd.Series(dtype=float)
        )
        target = target.reindex(posisi.index.union(target.index)).fillna(0.0)
        lama = posisi.reindex(target.index).fillna(0.0)
        arus, fee = nilai_transaksi(lama, target, harga_d, fee_beli, fee_jual)
        kas = kas + arus - fee
        posisi = target
        for ticker, w in bobot.items():
            baris_bobot.append(
                {
                    "date": tanggal,
                    "ticker": ticker,
                    "weight": float(w),
                    "k": k,
                    "estimator": estimator,
                    "lookback": lihat_balik,
                    "max_weight": bobot_maks,
                    "optimizer_type": optimizer_type,
                }
            )
        batas = jadwal[i + 1] if i + 1 < len(jadwal) else tanggal_akhir
        hari = indeks_harga[(indeks_harga >= tanggal) & (indeks_harga <= batas)]
        for hari_ini in hari:
            harga_h = harga.loc[hari_ini]
            nilai_saham = float(
                (posisi * harga_h.reindex(posisi.index).fillna(0.0)).sum()
                if len(posisi)
                else 0.0
            )
            baris_nilai.append({"date": hari_ini, "equity": kas + nilai_saham})

    nilai = pd.DataFrame(baris_nilai)
    nilai["date"] = pd.to_datetime(nilai["date"])
    nilai = nilai.sort_values("date").drop_duplicates("date")
    nilai["return"] = nilai["equity"].pct_change().fillna(0.0)
    bobot_frame = pd.DataFrame(
        baris_bobot,
        columns=[
            "date",
            "ticker",
            "weight",
            "k",
            "estimator",
            "lookback",
            "max_weight",
            "optimizer_type",
        ],
    )
    return bobot_frame, nilai
