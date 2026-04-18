"""Shared utilities for experiment setup, serialization, and calibration."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str | Path) -> None:
    """Write a dictionary to disk as formatted JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict:
    """Read a JSON file from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable NumPy softmax."""
    x = np.asarray(x)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def labels_to_indices(labels: np.ndarray) -> np.ndarray:
    """Convert one-hot or column labels to integer class indices."""
    labels = np.asarray(labels)
    if labels.ndim > 1 and labels.shape[-1] > 1:
        return np.argmax(labels, axis=-1)
    return labels.reshape(-1).astype(int)


def dataset_to_numpy(dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Materialize an unbatched or batched tf.data.Dataset as NumPy arrays."""
    xs, ys = [], []
    for x_batch, y_batch in dataset:
        xs.append(x_batch.numpy())
        ys.append(y_batch.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to logits and return probabilities."""
    return softmax(np.asarray(logits) / float(temperature), axis=-1)


def fit_temperature(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    iters: int = 300,
    learning_rate: float = 0.01,
) -> float:
    """Fit a scalar temperature on a validation dataset."""
    logits_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-1].output)
    logits = logits_model.predict(dataset, verbose=0)
    _, labels = dataset_to_numpy(dataset)
    labels = labels_to_indices(labels)
    y_true = tf.keras.utils.to_categorical(labels, logits.shape[-1])

    logits_tensor = tf.convert_to_tensor(logits, dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)
    temperature = tf.Variable(1.0, trainable=True, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for _ in range(iters):
        with tf.GradientTape() as tape:
            scaled_logits = logits_tensor / temperature
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels_tensor,
                    logits=scaled_logits,
                )
            )
        grads = tape.gradient(loss, [temperature])
        optimizer.apply_gradients(zip(grads, [temperature]))
        temperature.assign(tf.maximum(temperature, 1e-3))

    return float(temperature.numpy())
