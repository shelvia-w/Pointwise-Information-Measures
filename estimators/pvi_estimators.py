"""Pointwise V-information (PVI) estimators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from core.datasets import prefetch_dataset
from core.models import build_model, compile_model
from core.utils import softmax


def train_pvi_null_model(dataset, config: dict, save_path: str | Path | None = None):
    """Train a null model on zeroed inputs while preserving labels."""
    ds_null = dataset.map(lambda x, y: (tf.zeros_like(x), y))
    if len(ds_null.element_spec[0].shape) == len(config["input_shape"]):
        ds_null = prefetch_dataset(ds_null, batch_size=config["batch_size"])
    model = compile_model(build_model(config), config)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
    )
    model.fit(ds_null, epochs=config.get("null_epochs", 50), callbacks=[early_stop], verbose=1)
    if save_path:
        model.save(save_path)
    return model


def train_pvi_model_from_scratch(ds_train, ds_val, config: dict, save_path: str | Path | None = None):
    """Train a predictive model from scratch for PVI analysis."""
    model = compile_model(build_model(config), config)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.get("patience", 15),
        restore_best_weights=True,
    )
    model.fit(ds_train, epochs=config["max_epoch"], validation_data=ds_val, callbacks=[early_stop], verbose=1)
    if save_path:
        model.save(save_path)
    return model


def v_entropy(x, y, model) -> np.ndarray:
    """Compute V-entropy under a predictive model."""
    prob = model.predict(x, verbose=0)
    y = np.asarray(y).astype(int).reshape(-1)
    return -np.log2(np.clip(prob[np.arange(len(y)), y], 1e-12, None))


def v_entropy_ensemble(x1, x2, y, model1, model2) -> np.ndarray:
    """Compute ensemble V-entropy from two predictive models."""
    prob1 = model1.predict(x1, verbose=0)
    prob2 = model2.predict(x2, verbose=0)
    avg_prob = (prob1 + prob2) / 2.0
    y = np.asarray(y).astype(int).reshape(-1)
    return -np.log2(np.clip(avg_prob[np.arange(len(y)), y], 1e-12, None))


def neural_pvi(x, y, model, null_model) -> np.ndarray:
    """Compute neural PVI as null V-entropy minus conditional V-entropy."""
    null_x = np.zeros_like(x)
    return v_entropy(null_x, y, null_model) - v_entropy(x, y, model)


def neural_pvi_ensemble(x1, x2, y, model1, model2, null_model1, null_model2) -> np.ndarray:
    """Compute ensemble neural PVI."""
    null_x1 = np.zeros_like(x1)
    null_x2 = np.zeros_like(x2)
    v_cond_entropy = v_entropy_ensemble(x1, x2, y, model1, model2)
    v_null_entropy = v_entropy_ensemble(null_x1, null_x2, y, null_model1, null_model2)
    return v_null_entropy - v_cond_entropy


def v_entropy_calibrated(x, y, model, temperature: float) -> np.ndarray:
    """Compute V-entropy from temperature-scaled logits."""
    logits_layer = model.layers[-1]
    old_activation = logits_layer.activation
    logits_layer.activation = None
    logits_model = tf.keras.Model(inputs=model.input, outputs=logits_layer.output)
    logits = logits_model.predict(x, verbose=0)
    logits_layer.activation = old_activation
    prob = softmax(logits / temperature, axis=1)
    y = np.asarray(y).astype(int).reshape(-1)
    return -np.log2(np.clip(prob[np.arange(len(y)), y], 1e-12, None))


def neural_pvi_calibrated(x, y, model, null_model, model_temp: float, null_temp: float) -> np.ndarray:
    """Compute temperature-calibrated neural PVI."""
    null_x = np.zeros_like(x)
    v_cond_entropy = v_entropy_calibrated(x, y, model, model_temp)
    v_null_entropy = v_entropy_calibrated(null_x, y, null_model, null_temp)
    return v_null_entropy - v_cond_entropy
