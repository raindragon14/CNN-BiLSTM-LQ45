"""Uji dataset windowing; jalankan langsung: python3 tests/test_dataset.py.

Data di sini sintetis dan hanya untuk memvalidasi pipa, bukan hasil.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.models.dataset import build_windows, forward_log_return


def uji_forward_log_return() -> None:
    close = np.array([100.0, 110.0, 121.0, 99.0], dtype=np.float32)
    hasil = forward_log_return(close, 2)
    np.testing.assert_allclose(
        hasil[:2],
        [np.log(121.0 / 100.0), np.log(99.0 / 110.0)],
        rtol=1e-5,
        atol=1e-7,  # toleransi presisi float32
    )
    assert np.isnan(hasil[2:]).all()


def uji_bentuk_dan_isi_jendela() -> None:
    rng = np.random.default_rng(0)
    fitur = rng.normal(size=(100, 3)).astype(np.float32)
    target = rng.normal(size=(100,)).astype(np.float32)
    x, y, idx = build_windows(fitur, target, lookback=10, start=10, stop=25)
    assert x.shape == (15, 3, 10)
    assert list(idx) == list(range(10, 25))
    np.testing.assert_allclose(y, target[10:25])
    # Bandingkan dengan pembangunan manual untuk beberapa jendela.
    for i in range(5):
        t = 10 + i
        np.testing.assert_allclose(x[i], fitur[t - 9 : t + 1].T)


def uji_nan_dan_banned() -> None:
    fitur = np.ones((50, 2), dtype=np.float32)
    target = np.ones((50,), dtype=np.float32)
    fitur[0, 0] = np.nan  # masuk ke jendela yang berakhir t=9
    x, _y, idx = build_windows(
        fitur, target, lookback=10, start=9, stop=30, banned=[(20, 25)]
    )
    # 21 kandidat - 1 jendela ber-NaN - 5 tanggal dilarang = 15
    assert x.shape[0] == 15
    assert 9 not in idx
    assert not any(20 <= t < 25 for t in idx)
    assert not np.isnan(x).any()


def main() -> int:
    uji_forward_log_return()
    uji_bentuk_dan_isi_jendela()
    uji_nan_dan_banned()
    print("test_dataset.py: 3 uji lolos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
