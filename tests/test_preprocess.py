"""Robust preprocessing tests; run directly: python3 tests/test_preprocess.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.features.preprocess import RobustPreprocessor


def test_roundtrip() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    prep = RobustPreprocessor()
    scaled = prep.fit_transform(frame)
    restored = prep.inverse_transform(scaled)
    # Values that were not clipped round-trip back close to the original.
    clipped = frame.clip(prep.lower_, prep.upper_, axis=1)
    np.testing.assert_allclose(
        restored.to_numpy(), clipped.to_numpy(), rtol=1e-5, atol=1e-6
    )


def test_range() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    prep = RobustPreprocessor()
    scaled = prep.fit_transform(frame)
    assert scaled["x"].between(0.0, 1.0).all()
    assert scaled["x"].max() == 1.0
    # The extreme point 100 is clipped to the upper bound and scaled to 1.
    assert scaled["x"].iloc[-1] == 1.0


def main() -> int:
    test_roundtrip()
    test_range()
    print("test_preprocess.py: 2 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
