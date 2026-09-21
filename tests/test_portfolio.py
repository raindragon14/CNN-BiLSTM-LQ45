"""Uji modul portofolio; jalankan langsung: python3 tests/test_portfolio.py.

Data di sini sintetis dan hanya untuk memvalidasi pipa, bukan hasil.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.portfolio.backtest import (
    jalankan_backtest,
    tanggal_rebalancing,
)
from lq45.portfolio.costs import nilai_transaksi, target_lembar
from lq45.portfolio.covariance import taksir_kovarians
from lq45.portfolio.optimize import bobot_varians_minimum
from lq45.portfolio.ranking import ensemble_prediksi, peringkat_topk


def uji_ensemble_dan_topk() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 6,
            "ticker": ["A", "B", "C"] * 2,
            "pred_raw": [0.1, 0.3, 0.2, 0.3, 0.1, 0.2],
        }
    )
    ens = ensemble_prediksi(frame)
    assert len(ens) == 3
    assert peringkat_topk(ens, "2020-01-01", 2) == ["A", "B"]
    assert peringkat_topk(ens, "2020-02-01", 2) == []


def uji_kovarians_dan_bobot() -> None:
    rng = np.random.default_rng(1)
    imbal = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("ABCD"))
    for metode in ("sample", "ridge_epsilon", "ledoit_wolf", "gmv"):
        kov = taksir_kovarians(imbal, metode)
        assert kov.shape == (4, 4)
        bobot = bobot_varians_minimum(kov, 0.35)
        assert abs(bobot.sum() - 1.0) < 1e-6
        assert (bobot >= -1e-9).all() and (bobot <= 0.35 + 1e-6).all()
    satu = taksir_kovarians(imbal[["A"]], "sample")
    assert abs(bobot_varians_minimum(satu).iloc[0] - 1.0) < 1e-9


def uji_lot_dan_fee() -> None:
    bobot = pd.Series({"A": 0.5, "B": 0.5})
    harga = pd.Series({"A": 1000.0, "B": 500.0})
    lembar = target_lembar(bobot, harga, 1_000_000.0, lot=100)
    assert (lembar % 100 == 0).all()
    arus, fee = nilai_transaksi(
        pd.Series({"A": 0.0, "B": 0.0}),
        lembar,
        harga,
        fee_beli=0.0019,
        fee_jual=0.0029,
    )
    assert arus < 0 and fee > 0


def uji_backtest_kecil() -> None:
    tanggal = pd.date_range("2020-01-01", periods=170, freq="B")
    str_tanggal = [d.strftime("%Y-%m-%d") for d in tanggal]
    rng = np.random.default_rng(2)
    tickers = ["A.JK", "B.JK", "C.JK", "D.JK", "E.JK", "F.JK"]
    harga = pd.DataFrame(
        100 * np.exp(rng.normal(scale=0.01, size=(170, 6)).cumsum(axis=0)),
        index=tanggal,
        columns=tickers,
    )
    baris = [
        {"date": t, "ticker": c, "pred_ens": float(rng.normal())}
        for t in str_tanggal
        for c in tickers
    ]
    prediksi = pd.DataFrame(baris)
    bobot, nilai = jalankan_backtest(
        prediksi,
        harga,
        k=3,
        estimator="ridge_epsilon",
        lihat_balik=60,
        bobot_maks=0.5,
        modal=10_000_000.0,
    )
    assert len(bobot) > 0 and len(nilai) > 0
    assert abs(bobot.groupby("date")["weight"].sum().sub(1.0).abs().max()) < 1e-6
    assert (nilai["equity"] > 0).all()
    assert tanggal_rebalancing(str_tanggal[:42]) == str_tanggal[:42][::21]


def main() -> int:
    uji_ensemble_dan_topk()
    uji_kovarians_dan_bobot()
    uji_lot_dan_fee()
    uji_backtest_kecil()
    print("test_portfolio.py: 4 uji lolos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
