"""Training determinism tests; run: python3 tests/test_determinism.py.

The data here is synthetic and only validates the pipeline, not results.
Two trainings with the same seed must produce exactly the same validation
loss; this file checks that property.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lq45.models.training import build_model, resolve_device, train_model


def test_determinism() -> None:
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(256, 8, 30)).astype(np.float32)
    y_train = rng.normal(size=(256,)).astype(np.float32)
    x_val = rng.normal(size=(64, 8, 30)).astype(np.float32)
    y_val = rng.normal(size=(64,)).astype(np.float32)
    device = resolve_device("cpu")

    def run_single_fit() -> list[float]:
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
        result = train_model(
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
        return [row["val_loss"] for row in result.history]

    assert run_single_fit() == run_single_fit()


def main() -> int:
    test_determinism()
    print("test_determinism.py: 1 test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
