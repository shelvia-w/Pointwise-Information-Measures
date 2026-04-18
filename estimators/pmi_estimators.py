"""Pointwise mutual information (PMI) estimators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from core.utils import ensure_dir


def neural_pmi(x, y, pmi_model, estimator: str = "probabilistic_classifier") -> np.ndarray:
    """Estimate PMI from a trained neural critic."""
    scores = pmi_model(x, y)
    batch_size = scores.shape[0]
    if estimator == "probabilistic_classifier":
        joint_logits = tf.sigmoid(tf.reshape(scores, [-1])[:: batch_size + 1])
        pmi = tf.math.log((batch_size - 1.0) * joint_logits / (1.0 - joint_logits))
    elif estimator == "density_ratio_fitting":
        pmi = tf.linalg.diag_part(tf.math.log(tf.maximum(scores, 1e-4)))
    elif estimator == "variational_f_js":
        pmi = tf.linalg.diag_part(scores)
    else:
        raise NotImplementedError(f"PMI estimator '{estimator}' is not supported.")
    return np.asarray(pmi) / np.log(2.0)


def train_critic_model(
    dataset: tf.data.Dataset,
    critic: str = "separable",
    estimator: str = "probabilistic_classifier",
    epochs: int = 50,
    save_path: str | Path | None = None,
) -> keras.Model:
    """Train a neural critic for PMI estimation."""
    if critic == "concat":
        model = ConcatCritic(dataset)
    elif critic == "separable":
        model = SeparableCritic(dataset)
    else:
        raise NotImplementedError(f"Critic model '{critic}' is not supported.")

    objectives = {
        "probabilistic_classifier": probabilistic_classifier_obj,
        "density_ratio_fitting": density_ratio_fitting_obj,
        "variational_f_js": js_fgan_lower_bound_obj,
    }
    loss_fn = objectives[estimator]
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            scores = model(x_batch, y_batch)
            loss_value = -loss_fn(scores)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

    for _ in range(epochs):
        for x_batch, y_batch in dataset:
            train_step(x_batch, y_batch)

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        model.save(save_path)
    return model


def mlp_critic(input_dim: int, output_dim: int) -> keras.Model:
    """Small MLP used in neural critic architectures."""
    return keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation="relu"),
            layers.Dense(output_dim),
        ]
    )


class SeparableCritic(keras.Model):
    """Separable critic f(x,y)=g(x)^T h(y)."""

    def __init__(self, dataset: tf.data.Dataset, output_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        dim_x = int(dataset.element_spec[0].shape[-1])
        dim_y = int(dataset.element_spec[1].shape[-1])
        self.output_dim = output_dim
        self._g = mlp_critic(dim_x, output_dim)
        self._h = mlp_critic(dim_y, output_dim)

    def call(self, x, y):
        g_output = self._g(x)
        h_output = self._h(y)
        return tf.matmul(h_output, tf.transpose(g_output))

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config


class ConcatCritic(keras.Model):
    """Concatenation critic that evaluates all batch pairs."""

    def __init__(self, dataset: tf.data.Dataset, **kwargs):
        super().__init__(**kwargs)
        dim_x = int(dataset.element_spec[0].shape[-1])
        dim_y = int(dataset.element_spec[1].shape[-1])
        self._f = mlp_critic(dim_x + dim_y, 1)

    def call(self, x, y):
        batch_size = tf.shape(x)[0]
        x_tiled = tf.tile(tf.expand_dims(x, axis=1), [1, batch_size, 1])
        y_tiled = tf.tile(tf.expand_dims(y, axis=0), [batch_size, 1, 1])
        xy_pairs = tf.concat([x_tiled, y_tiled], axis=-1)
        scores = self._f(tf.reshape(xy_pairs, [batch_size * batch_size, -1]))
        return tf.reshape(scores, [batch_size, batch_size])


@tf.function
def probabilistic_classifier_obj(score):
    """Binary classification objective for joint vs marginal pairs."""
    criterion = keras.losses.BinaryCrossentropy(from_logits=True)
    batch_size = score.shape[0]
    labels = [0.0] * (batch_size * batch_size)
    labels[:: batch_size + 1] = [1.0] * batch_size
    labels = tf.reshape(tf.convert_to_tensor(labels, dtype=score.dtype), (-1, 1))
    logits = tf.reshape(score, (-1, 1))
    return -criterion(labels, logits)


@tf.function
def density_ratio_fitting_obj(score):
    """Density-ratio fitting objective."""
    score_square = tf.square(score)
    batch_size = score.shape[0]
    joint_term = tf.reduce_mean(tf.linalg.diag_part(score))
    marg_term = (
        tf.reduce_sum(score_square) - tf.reduce_sum(tf.linalg.diag_part(score_square))
    ) / (batch_size * (batch_size - 1.0))
    return joint_term - 0.5 * marg_term


@tf.function
def js_fgan_lower_bound_obj(score):
    """Jensen-Shannon f-GAN lower-bound objective."""
    score_diag = tf.linalg.diag_part(score)
    first_term = -tf.reduce_mean(tf.nn.softplus(-score_diag))
    batch_size = score.shape[0]
    second_term = (
        tf.reduce_sum(tf.nn.softplus(score)) - tf.reduce_sum(tf.nn.softplus(score_diag))
    ) / (batch_size * (batch_size - 1.0))
    return first_term - second_term
