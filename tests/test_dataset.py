"""Dataset windowing tests; run directly: python3 tests/test_dataset.py.

The data here is synthetic and only validates the pipeline, not results.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.models.dataset import build_windows, forward_log_return


def test_forward_log_return() -> None:
    close = np.array([100.0, 110.0, 121.0, 99.0], dtype=np.float32)
    result = forward_log_return(close, 2)
    np.testing.assert_allclose(
        result[:2],
        [np.log(121.0 / 100.0), np.log(99.0 / 110.0)],
        rtol=1e-5,
        atol=1e-7,  # float32 precision tolerance
    )
    assert np.isnan(result[2:]).all()


def test_window_shape_and_content() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(100, 3)).astype(np.float32)
    target = rng.normal(size=(100,)).astype(np.float32)
    x, y, idx = build_windows(features, target, lookback=10, start=10, stop=25)
    assert x.shape == (15, 3, 10)
    assert list(idx) == list(range(10, 25))
    np.testing.assert_allclose(y, target[10:25])
    # Compare against manual construction for a few windows.
    for i in range(5):
        t = 10 + i
        np.testing.assert_allclose(x[i], features[t - 9 : t + 1].T)


def test_nan_and_banned() -> None:
    features = np.ones((50, 2), dtype=np.float32)
    target = np.ones((50,), dtype=np.float32)
    features[0, 0] = np.nan  # falls within the window ending at t=9
    x, _y, idx = build_windows(
        features, target, lookback=10, start=9, stop=30, banned=[(20, 25)]
    )
    # 21 candidates - 1 NaN window - 5 banned dates = 15
    assert x.shape[0] == 15
    assert 9 not in idx
    assert not any(20 <= t < 25 for t in idx)
    assert not np.isnan(x).any()


def main() -> int:
    test_forward_log_return()
    test_window_shape_and_content()
    test_nan_and_banned()
    print("test_dataset.py: 3 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
