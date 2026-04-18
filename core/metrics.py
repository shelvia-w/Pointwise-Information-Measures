"""Calibration, filtering, and selective prediction metrics."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from core.utils import labels_to_indices


def reliability_diagram(confidence, true_y, pred_y, n_bins: int = 15) -> dict:
    """Compute reliability-diagram bins and expected calibration error."""
    confidence = np.asarray(confidence)
    true_y = labels_to_indices(true_y)
    pred_y = labels_to_indices(pred_y)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(confidence, bins, right=True)
    indices = np.clip(indices, 1, n_bins)

    bin_acc = np.zeros(n_bins, dtype=np.float32)
    bin_conf = np.zeros(n_bins, dtype=np.float32)
    bin_counts = np.zeros(n_bins, dtype=np.int32)

    for b in range(n_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_acc[b] = np.mean(true_y[selected] == pred_y[selected])
            bin_conf[b] = np.mean(confidence[selected])
            bin_counts[b] = len(selected)

    total = max(np.sum(bin_counts), 1)
    gaps = np.abs(bin_acc - bin_conf)
    return {
        "bins": bins,
        "bin_size": 1.0 / n_bins,
        "bin_counts": bin_counts,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "avg_acc": float(np.sum(bin_acc * bin_counts) / total),
        "avg_conf": float(np.sum(bin_conf * bin_counts) / total),
        "gaps": gaps,
        "ece": float(np.sum(gaps * bin_counts) / total * 100.0),
    }


def compute_ece(confidence, true_y, pred_y, n_bins: int = 15) -> float:
    """Compute expected calibration error in percent."""
    return reliability_diagram(confidence, true_y, pred_y, n_bins)["ece"]


def compute_nll(probabilities, true_y) -> float:
    """Compute categorical negative log-likelihood."""
    probabilities = np.asarray(probabilities)
    true_y = labels_to_indices(true_y)
    y_true = tf.one_hot(true_y, depth=probabilities.shape[-1])
    return float(tf.keras.losses.CategoricalCrossentropy()(y_true, probabilities).numpy())


def compute_brier_score(probabilities, true_y) -> float:
    """Compute multiclass Brier score."""
    probabilities = np.asarray(probabilities)
    true_y = labels_to_indices(true_y)
    y_true = tf.one_hot(true_y, depth=probabilities.shape[-1]).numpy()
    return float(np.mean(np.sum((probabilities - y_true) ** 2, axis=1)))


def compute_classification_error(probabilities, true_y) -> float:
    """Compute classification error from class probabilities."""
    pred_y = np.argmax(probabilities, axis=1)
    true_y = labels_to_indices(true_y)
    return float(1.0 - np.mean(pred_y == true_y))


def compute_opt_threshold(metric, true_label) -> float:
    """Find the threshold that maximizes binary filtering accuracy."""
    metric = np.asarray(metric)
    true_label = np.asarray(true_label)
    best_threshold = float(metric[0])
    best_acc = -np.inf
    for threshold in np.sort(metric):
        pred_label = (metric >= threshold).astype(int)
        acc = np.mean(pred_label == true_label)
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold


def compute_filtering_acc(metric, true_label, threshold: float) -> float:
    """Compute binary filtering accuracy in percent."""
    pred_label = (np.asarray(metric) >= threshold).astype(int)
    return float(np.mean(pred_label == np.asarray(true_label)) * 100.0)


def selective_prediction_curve(scores, correctness, coverage_points: int = 100) -> dict:
    """Compute selective prediction error as lower-confidence samples are filtered."""
    scores = np.asarray(scores)
    correctness = np.asarray(correctness).astype(bool)
    order = np.argsort(scores)[::-1]
    sorted_correct = correctness[order]
    coverages = np.linspace(1.0 / coverage_points, 1.0, coverage_points)
    errors = []
    for coverage in coverages:
        keep = max(1, int(np.ceil(len(scores) * coverage)))
        errors.append(float(1.0 - np.mean(sorted_correct[:keep])))
    return {"coverage": coverages, "error": np.asarray(errors)}


def confidence_from_probabilities(probabilities) -> np.ndarray:
    """Return maximum softmax confidence."""
    return np.max(np.asarray(probabilities), axis=1)
