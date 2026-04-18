"""Compute calibration metrics."""

from __future__ import annotations

import numpy as np

from configs.benchmark_configs import get_benchmark_config
from configs.base_config import experiment_dir
from configs.estimator_configs import EVALUATION_CONFIG
from core.datasets import prepare_datasets
from core.feature_extraction import label_array, prediction_array
from core.metrics import (
    compute_brier_score,
    compute_classification_error,
    compute_ece,
    compute_nll,
    confidence_from_probabilities,
)
from core.models import load_trained_model
from core.utils import save_json


def run(args=None, config: dict | None = None):
    """Compute ECE, Brier score, NLL, and classification error."""
    config = config or get_benchmark_config(args.benchmark)
    config.update(EVALUATION_CONFIG)
    output_dir = experiment_dir(config, run=getattr(args, "run", 1))
    bundle = prepare_datasets(config, shuffle=False)
    model = load_trained_model(output_dir / "trained_model.keras")
    probs = prediction_array(model, bundle.test)
    true_y = label_array(bundle.test)
    pred_y = np.argmax(probs, axis=1)
    confidence = confidence_from_probabilities(probs)
    metrics = {
        "ece": compute_ece(confidence, true_y, pred_y, n_bins=config["n_bins"]),
        "brier_score": compute_brier_score(probs, true_y),
        "nll": compute_nll(probs, true_y),
        "classification_error": compute_classification_error(probs, true_y),
    }
    save_json(metrics, output_dir / "calibration_metrics.json")
    return {"metrics": metrics, "output_dir": output_dir}
