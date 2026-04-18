"""Compute selective prediction curves."""

from __future__ import annotations

import numpy as np

from configs.benchmark_configs import get_benchmark_config
from configs.base_config import experiment_dir
from configs.estimator_configs import EVALUATION_CONFIG
from core.datasets import prepare_datasets
from core.feature_extraction import label_array, prediction_array
from core.metrics import confidence_from_probabilities, selective_prediction_curve
from core.models import load_trained_model
from core.plotting import plot_curve


def run(args=None, config: dict | None = None):
    """Save softmax selective prediction curve."""
    config = config or get_benchmark_config(args.benchmark)
    config.update(EVALUATION_CONFIG)
    output_dir = experiment_dir(config, run=getattr(args, "run", 1))
    bundle = prepare_datasets(config, shuffle=False)
    model = load_trained_model(output_dir / "trained_model.keras")
    probs = prediction_array(model, bundle.test)
    true_y = label_array(bundle.test)
    pred_y = np.argmax(probs, axis=1)
    confidence = confidence_from_probabilities(probs)
    curve = selective_prediction_curve(confidence, pred_y == true_y, config["coverage_points"])
    np.savez(output_dir / "softmax_selective_prediction.npz", **curve)
    plot_curve(curve["coverage"], curve["error"], "Coverage", "Selective error", output_dir / "selective_prediction.png")
    return {"curve": curve, "output_dir": output_dir}
