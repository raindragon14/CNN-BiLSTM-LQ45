"""Pemuatan konfigurasi dan lokasi berkas proyek."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
EXPERIMENT_DIR = ROOT / "experiments"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"


def load_config(name: str) -> dict[str, Any]:
    """Muat satu berkas YAML dari ``configs/`` berdasarkan nama tanpa ekstensi."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"konfigurasi tidak ditemukan: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(*paths: Path) -> None:
    """Buat direktori keluaran bila belum ada."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
