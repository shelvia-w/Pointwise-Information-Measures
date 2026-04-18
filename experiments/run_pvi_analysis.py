"""Run neural PVI analysis."""

from __future__ import annotations

import numpy as np

from configs.benchmark_configs import get_benchmark_config
from configs.base_config import experiment_dir
from configs.estimator_configs import PVI_CONFIG
from core.datasets import prepare_datasets
from core.models import load_trained_model
from core.utils import dataset_to_numpy, labels_to_indices, set_seed
from estimators.pvi_estimators import neural_pvi, train_pvi_null_model


def run(args=None, config: dict | None = None):
    """Train/load the null model and save test-set PVI scores."""
    config = config or get_benchmark_config(args.benchmark)
    config.update(PVI_CONFIG)
    set_seed(config["seed"])
    run_id = getattr(args, "run", 1)
    output_dir = experiment_dir(config, run=run_id)
    bundle = prepare_datasets(config, shuffle=False)
    model = load_trained_model(output_dir / "trained_model.keras")
    null_model_path = output_dir / "pvi_null_model.keras"
    null_model = train_pvi_null_model(bundle.train, config, save_path=null_model_path)

    x_test, y_test = dataset_to_numpy(bundle.test)
    y_test = labels_to_indices(y_test)
    pvi = neural_pvi(x_test, y_test, model, null_model)
    np.save(output_dir / "pvi_test.npy", pvi)
    return {"pvi": pvi, "output_dir": output_dir}
