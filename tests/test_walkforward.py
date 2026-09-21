"""Uji lipatan walk-forward; jalankan langsung: python3 tests/test_walkforward.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.models.walkforward import design_split, make_folds

SPLIT = {
    "initial_train_years": 2,
    "validation_days": 63,
    "test_days": 21,
    "step_days": 21,
    "expanding": True,
    "purge": {"horizon_days": 5},
    "embargo_days": 59,
}


def uji_jumlah_dan_batas() -> None:
    lipatan = make_folds(1932, SPLIT)
    assert len(lipatan) == 65
    awal = lipatan[0]
    assert awal.train == (0, 499)  # 504 - purge 5
    assert awal.validation == (504, 567)
    assert awal.test == (567, 588)
    assert awal.banned == []
    akhir = lipatan[-1]
    assert akhir.test[1] == 1932  # menutup kalender persis


def uji_embargo() -> None:
    lipatan = make_folds(1932, SPLIT)
    # Test lipatan 0 berakhir di 588; larangan [588, 647) muncul di semua
    # lipatan berikutnya.
    for fold in lipatan[1:]:
        assert (588, 647) in [tuple(b) for b in fold.banned]


def uji_purge_dan_embargo_parameter() -> None:
    lipatan = make_folds(1932, SPLIT, purge_days=21)
    assert lipatan[0].train == (0, 483)  # 504 - 21
    lipatan = make_folds(1932, SPLIT, embargo_days=29)
    for fold in lipatan[1:]:
        assert (588, 617) in [tuple(b) for b in fold.banned]


def uji_test_menutup_oos() -> None:
    lipatan = make_folds(1932, SPLIT)
    tertutup = np.zeros(1932, dtype=bool)
    for fold in lipatan:
        a, b = fold.test
        assert not tertutup[a:b].any()  # test tidak tumpang tindih
        tertutup[a:b] = True
    assert tertutup[:567].sum() == 0  # 63 hari pembuka OOS tanpa test
    assert tertutup[567:].all()  # sisanya tertutup penuh


def uji_design_split() -> None:
    dates = pd.date_range("2018-01-01", "2019-12-31", freq="B")
    design = design_split(
        dates, "2018-01-01/2018-12-31", "2019-01-01/2019-12-31", horizon=5
    )
    pos_2018 = np.flatnonzero(dates.year == 2018)
    pos_2019 = np.flatnonzero(dates.year == 2019)
    assert design.train == (int(pos_2018[0]), int(pos_2018[-1]) + 1 - 5)
    assert design.validation == (int(pos_2019[0]), int(pos_2019[-1]) + 1)


def main() -> int:
    uji_jumlah_dan_batas()
    uji_embargo()
    uji_purge_dan_embargo_parameter()
    uji_test_menutup_oos()
    uji_design_split()
    print("test_walkforward.py: 5 uji lolos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
