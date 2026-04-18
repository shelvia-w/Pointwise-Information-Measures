"""Run neural PMI analysis on extracted model features."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from configs.benchmark_configs import get_benchmark_config
from configs.base_config import experiment_dir
from configs.estimator_configs import PMI_CONFIG
from core.datasets import prepare_datasets
from core.feature_extraction import feature_array
from core.models import load_trained_model
from core.utils import ensure_dir, set_seed
from estimators.pmi_estimators import neural_pmi, train_critic_model


def run(args=None, config: dict | None = None):
    """Train a PMI critic and save test-set PMI scores."""
    config = config or get_benchmark_config(args.benchmark)
    config.update(PMI_CONFIG)
    set_seed(config["seed"])
    output_dir = ensure_dir(experiment_dir(config, run=getattr(args, "run", 1)) / "pmi")
    bundle = prepare_datasets(config, shuffle=False)
    model = load_trained_model(experiment_dir(config, run=getattr(args, "run", 1)) / "trained_model.keras")

    x_train, y_train = feature_array(model, bundle.train, layer=config["feature_layer"])
    x_test, y_test = feature_array(model, bundle.test, layer=config["feature_layer"])
    y_train_oh = tf.keras.utils.to_categorical(y_train, config["num_classes"])
    y_test_oh = tf.keras.utils.to_categorical(y_test, config["num_classes"])

    critic_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_oh)).batch(config["batch_size"])
    critic = train_critic_model(
        critic_ds,
        critic=config["critic"],
        estimator=config["pmi_estimator"],
        epochs=config["critic_epochs"],
        save_path=output_dir / "pmi_critic.keras",
    )
    pmi_values = []
    for start in range(0, len(x_test), config["batch_size"]):
        stop = start + config["batch_size"]
        pmi_values.append(neural_pmi(x_test[start:stop], y_test_oh[start:stop], critic, config["pmi_estimator"]))
    pmi_values = np.concatenate(pmi_values)
    np.save(output_dir / "pmi_test.npy", pmi_values)
    return {"pmi": pmi_values, "output_dir": output_dir}
