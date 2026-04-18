"""Generate reliability diagram data and plots."""

from __future__ import annotations

import numpy as np

from configs.benchmark_configs import get_benchmark_config
from configs.base_config import experiment_dir
from configs.estimator_configs import EVALUATION_CONFIG
from core.datasets import prepare_datasets
from core.feature_extraction import label_array, prediction_array
from core.metrics import confidence_from_probabilities, reliability_diagram
from core.models import load_trained_model
from core.plotting import plot_reliability_diagram


def run(args=None, config: dict | None = None):
    """Compute and plot a softmax reliability diagram."""
    config = config or get_benchmark_config(args.benchmark)
    config.update(EVALUATION_CONFIG)
    output_dir = experiment_dir(config, run=getattr(args, "run", 1))
    bundle = prepare_datasets(config, shuffle=False)
    model = load_trained_model(output_dir / "trained_model.keras")
    probs = prediction_array(model, bundle.test)
    true_y = label_array(bundle.test)
    pred_y = np.argmax(probs, axis=1)
    confidence = confidence_from_probabilities(probs)
    result = reliability_diagram(confidence, true_y, pred_y, n_bins=config["n_bins"])
    np.savez(output_dir / "reliability_diagram.npz", **result)
    plot_reliability_diagram(result, "Softmax confidence", output_dir / "reliability_diagram.png")
    return {"result": result, "output_dir": output_dir}
