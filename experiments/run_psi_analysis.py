"""Run PSI analysis on model features."""

from __future__ import annotations

import pickle

import numpy as np
import tensorflow as tf

from configs.benchmark_configs import get_benchmark_config
from configs.base_config import experiment_dir
from configs.estimator_configs import PSI_CONFIG
from core.datasets import class_priors, prepare_datasets
from core.feature_extraction import feature_array
from core.models import load_trained_model
from core.utils import ensure_dir, set_seed
from estimators.psi_estimators import (
    psi_bin_train,
    psi_bin_val,
    psi_gauss_train,
    psi_gauss_val,
    psi_neural_train,
    psi_neural_val,
    psi_rf_train,
    psi_rf_val,
)


def run(args=None, config: dict | None = None):
    """Train/evaluate PSI estimators and save test-set PSI scores."""
    config = config or get_benchmark_config(args.benchmark)
    config.update(PSI_CONFIG)
    if getattr(args, "psi_estimator", None):
        config["psi_estimator"] = args.psi_estimator
    set_seed(config["seed"])
    run_id = getattr(args, "run", 1)
    output_dir = ensure_dir(experiment_dir(config, run=run_id) / "psi")
    bundle = prepare_datasets(config, shuffle=False)
    model = load_trained_model(experiment_dir(config, run=run_id) / "trained_model.keras")
    x_train, y_train = feature_array(model, bundle.train, layer=config["feature_layer"])
    x_test, y_test = feature_array(model, bundle.test, layer=config["feature_layer"])

    estimator = config["psi_estimator"]
    if estimator == "histogram":
        psi_data = psi_bin_train(x_train, y_train, config["n_projs"], config["n_bins"])
        with (output_dir / "psi_histogram.pkl").open("wb") as handle:
            pickle.dump(psi_data, handle)
        psi, psi_by_proj = psi_bin_val(x_test, y_test, psi_data, config["n_projs"])
    elif estimator == "gaussian":
        psi_data = psi_gauss_train(x_train, y_train, config["n_projs"])
        with (output_dir / "psi_gaussian.pkl").open("wb") as handle:
            pickle.dump(psi_data, handle)
        psi, psi_by_proj = psi_gauss_val(x_test, y_test, psi_data, config["n_projs"])
    elif estimator == "random_forest":
        class_prob = class_priors(bundle.train, config["num_classes"])
        thetas = psi_rf_train(x_train, y_train, config["n_projs"], output_dir)
        psi, psi_by_proj = psi_rf_val(x_test, y_test, thetas, class_prob, config["n_projs"], output_dir)
    elif estimator == "neural":
        y_train_oh = tf.keras.utils.to_categorical(y_train, config["num_classes"])
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
        class_prob = class_priors(bundle.train, config["num_classes"])
        thetas = psi_neural_train(ds_train, config, config["n_projs"], output_dir)
        ds_test = tf.data.Dataset.from_tensor_slices(
            (x_test, tf.keras.utils.to_categorical(y_test, config["num_classes"]))
        )
        psi, psi_by_proj = psi_neural_val(ds_test, thetas, class_prob, config, config["n_projs"], output_dir)
    else:
        raise NotImplementedError(f"Unknown PSI estimator '{estimator}'.")

    np.save(output_dir / "psi_test.npy", psi)
    np.save(output_dir / "psi_test_by_projection.npy", psi_by_proj)
    return {"psi": psi, "output_dir": output_dir}
