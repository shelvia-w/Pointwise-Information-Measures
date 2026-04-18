"""Base configuration helpers for reproducible experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


BASE_CONFIG = {
    "seed": 42,
    "batch_size": 128,
    "max_epoch": 100,
    "patience": 15,
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "validation_split": 0.15,
    "data_dir": str(DEFAULT_DATA_DIR),
    "results_dir": str(DEFAULT_RESULTS_DIR),
    "normalize": True,
    "one_hot": True,
    "cache": False,
    "shuffle": True,
    "model_weights": None,
    "freeze_backbone": False,
}


def merge_config(*configs: dict) -> dict:
    """Return a shallow merged copy of configuration dictionaries."""
    merged = deepcopy(BASE_CONFIG)
    for config in configs:
        if config:
            merged.update(config)
    return merged


def experiment_dir(config: dict, run: int | None = None) -> Path:
    """Build the canonical output directory for one model-dataset experiment."""
    name = f"{config['model']}_{config['dataset']}"
    path = Path(config.get("results_dir", DEFAULT_RESULTS_DIR)) / name
    if run is not None:
        path = path / f"run_{run}"
    return path
