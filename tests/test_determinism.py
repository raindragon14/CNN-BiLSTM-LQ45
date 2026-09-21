"""Uji determinisme pelatihan; jalankan: python3 tests/test_determinism.py.

Data di sini sintetis dan hanya untuk memvalidasi pipa, bukan hasil.
Dua pelatihan dengan seed sama menghasilkan loss validasi yang sama persis;
berkas ini mengujinya.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.models.training import build_model, resolve_device, train_model


def uji_determinisme() -> None:
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(256, 8, 30)).astype(np.float32)
    y_train = rng.normal(size=(256,)).astype(np.float32)
    x_val = rng.normal(size=(64, 8, 30)).astype(np.float32)
    y_val = rng.normal(size=(64,)).astype(np.float32)
    device = resolve_device("cpu")

    def satu_fit() -> list[float]:
        model = build_model(
            {
                "n_features": 8,
                "filters": (8, 16),
                "kernel_size": 3,
                "pooling": 2,
                "units": 16,
                "layers": 1,
            },
            seed=7,
        )
        hasil = train_model(
            model,
            (x_train, y_train),
            (x_val, y_val),
            batch_size=32,
            epochs=3,
            patience=8,
            lr=1e-3,
            weight_decay=0.0,
            seed=7,
            device=device,
        )
        return [baris["val_loss"] for baris in hasil.history]

    assert satu_fit() == satu_fit()


def main() -> int:
    uji_determinisme()
    print("test_determinism.py: 1 uji lolos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
