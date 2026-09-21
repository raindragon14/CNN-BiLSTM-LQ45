"""Uji modul evaluasi; jalankan langsung: python3 tests/test_evaluation.py.

Data di sini sintetis dan hanya untuk memvalidasi pipa, bukan hasil.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.evaluation.dsr_pbo import (
    deflated_sharpe,
    pbo_cscv,
    sharpe_benchmark_harapan,
    sharpe_tahunan,
)
from lq45.evaluation.metrics import mdd_dan_durasi, ringkas_deret
from lq45.evaluation.regimes import bagi_rezim
from lq45.evaluation.significance import (
    romano_wolf_stepdown,
)
from lq45.evaluation.significance import (
    uji_beda_sharpe as beda_sharpe_fn,
)


def uji_metrik_dasar() -> None:
    rng = np.random.default_rng(3)
    idx = pd.RangeIndex(252)
    imbal = pd.Series(rng.normal(0.0005, 0.01, size=252), index=idx)
    ringkas = ringkas_deret(imbal)
    assert ringkas["n_days"] == 252
    assert -1.0 < ringkas["max_drawdown"] <= 0.0
    assert ringkas["cumulative_return"] > -1.0
    mdd, durasi = mdd_dan_durasi(pd.Series(np.cumprod(1.0 + imbal), index=idx))
    assert mdd <= 0.0 and durasi >= 0
    assert abs(sharpe_tahunan(imbal)) < 5.0


def uji_dsr_dan_pbo() -> None:
    rng = np.random.default_rng(4)
    idx = pd.RangeIndex(500)
    imbal = pd.Series(rng.normal(0.001, 0.01, size=500), index=idx)
    hasil = deflated_sharpe(imbal, n_uji=50)
    assert 0.0 <= hasil["dsr"] <= 1.0
    assert hasil["benchmark"] >= 0.0
    assert sharpe_benchmark_harapan(1, 1.0) == 0.0
    matriks = pd.DataFrame(
        rng.normal(size=(400, 6)), columns=[f"s{i}" for i in range(6)]
    )
    pbo = pbo_cscv(matriks, bagian=4)
    assert 0.0 <= pbo["pbo"] <= 1.0
    assert pbo["n_splits"] == 6.0


def uji_signifikansi() -> None:
    rng = np.random.default_rng(5)
    idx = pd.RangeIndex(300)
    imbal_a = pd.Series(rng.normal(0.001, 0.01, size=300), index=idx)
    imbal_b = pd.Series(rng.normal(0.0, 0.01, size=300), index=idx)
    uji = beda_sharpe_fn(imbal_a, imbal_b)
    assert 0.0 <= uji["p_value"] <= 1.0
    matriks = pd.DataFrame({"s1": imbal_a, "s2": imbal_b})
    tabel = romano_wolf_stepdown(matriks, imbal_b, ulang=50, seed=0)
    assert set(tabel.columns) == {
        "strategy",
        "sharpe_diff",
        "critical_value",
        "reject",
    }
    assert len(tabel) == 2


def uji_rezim() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01", "2021-06-01", "2019-06-01"]),
            "return": [0.01, 0.02, 0.03],
        }
    )
    rezim = {
        "covid": {"start": "2020-01-01", "end": "2020-12-31"},
        "recovery_rate_hike": {"start": "2021-01-01", "end": "2025-12-31"},
    }
    bagi = bagi_rezim(frame, "date", rezim)
    assert len(bagi["covid"]) == 1 and len(bagi["recovery_rate_hike"]) == 1


def main() -> int:
    uji_metrik_dasar()
    uji_dsr_dan_pbo()
    uji_signifikansi()
    uji_rezim()
    print("test_evaluation.py: 4 uji lolos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
