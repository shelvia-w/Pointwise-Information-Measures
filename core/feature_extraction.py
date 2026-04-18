"""Feature, logit, and prediction extraction helpers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from core.utils import dataset_to_numpy, labels_to_indices


def prediction_array(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """Return class probabilities for a dataset."""
    return model.predict(dataset, verbose=0)


def label_array(dataset: tf.data.Dataset) -> np.ndarray:
    """Return integer labels for a dataset."""
    _, labels = dataset_to_numpy(dataset)
    return labels_to_indices(labels)


def logit_array(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """Return pre-softmax logits by cloning the final dense layer activation."""
    logits_layer = model.layers[-1]
    old_activation = logits_layer.activation
    logits_layer.activation = None
    logits_model = tf.keras.Model(inputs=model.input, outputs=logits_layer.output)
    logits = logits_model.predict(dataset, verbose=0)
    logits_layer.activation = old_activation
    return logits


def feature_array(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    layer: int | str = -2,
    flatten: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract intermediate features and integer labels from a dataset."""
    target_layer = model.layers[layer] if isinstance(layer, int) else model.get_layer(layer)
    feature_model = tf.keras.Model(inputs=model.input, outputs=target_layer.output)
    features = feature_model.predict(dataset, verbose=0)
    if flatten:
        features = features.reshape(features.shape[0], -1)
    return features, label_array(dataset)
