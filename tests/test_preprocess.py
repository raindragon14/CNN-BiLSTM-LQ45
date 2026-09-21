"""Uji praproses robust; jalankan langsung: python3 tests/test_preprocess.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.features.preprocess import RobustPreprocessor


def uji_roundtrip() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    prep = RobustPreprocessor()
    terskala = prep.fit_transform(frame)
    balik = prep.inverse_transform(terskala)
    # Nilai yang tidak dicapit kembali mendekati aslinya.
    dicapit = frame.clip(prep.lower_, prep.upper_, axis=1)
    np.testing.assert_allclose(
        balik.to_numpy(), dicapit.to_numpy(), rtol=1e-5, atol=1e-6
    )


def uji_rentang() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    prep = RobustPreprocessor()
    terskala = prep.fit_transform(frame)
    assert terskala["x"].between(0.0, 1.0).all()
    assert terskala["x"].max() == 1.0
    # Titik ekstrem 100 dicapit ke batas atas lalu diskalakan ke 1.
    assert terskala["x"].iloc[-1] == 1.0


def main() -> int:
    uji_roundtrip()
    uji_rentang()
    print("test_preprocess.py: 2 uji lolos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
