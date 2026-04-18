"""Compute filtering accuracy for confidence/error detection."""

from __future__ import annotations

import numpy as np

from configs.benchmark_configs import get_benchmark_config
from configs.base_config import experiment_dir
from core.datasets import prepare_datasets
from core.feature_extraction import label_array, prediction_array
from core.metrics import compute_filtering_acc, compute_opt_threshold
from core.models import load_trained_model


def run(args=None, config: dict | None = None):
    """Save the optimal softmax filtering threshold and accuracy."""
    config = config or get_benchmark_config(args.benchmark)
    output_dir = experiment_dir(config, run=getattr(args, "run", 1))
    bundle = prepare_datasets(config, shuffle=False)
    model = load_trained_model(output_dir / "trained_model.keras")
    probs = prediction_array(model, bundle.test)
    true_y = label_array(bundle.test)
    pred_y = np.argmax(probs, axis=1)
    correctness = (pred_y == true_y).astype(int)
    confidence = np.max(probs, axis=1)
    threshold = compute_opt_threshold(confidence, correctness)
    accuracy = compute_filtering_acc(confidence, correctness, threshold)
    np.savez(output_dir / "softmax_filtering_accuracy.npz", threshold=threshold, accuracy=accuracy)
    return {"threshold": threshold, "accuracy": accuracy, "output_dir": output_dir}
