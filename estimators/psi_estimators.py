"""Pointwise sliced information (PSI) estimators."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

from core.datasets import prefetch_dataset
from core.models import get_optimizer, mlp
from core.utils import ensure_dir, labels_to_indices


def sample_from_sphere(d: int) -> np.ndarray:
    """Sample a random unit vector in R^d."""
    vec = np.random.randn(d, 1)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def psi_bin_train(x, y, n_projs: int, n_bins: int) -> dict:
    """Fit histogram PSI projection densities."""
    y = labels_to_indices(y)
    data = {"hist": [], "bin_edges": [], "thetas": []}
    n_classes = int(np.max(y) + 1)
    class_counts = np.array([np.sum(y == k) for k in range(n_classes)])

    for _ in range(n_projs):
        theta = sample_from_sphere(x.shape[1])
        data["thetas"].append(theta.squeeze())
        projected = np.dot(x, theta).squeeze()
        hist_list, edges_list = [], []
        for k in range(n_classes):
            hist, bin_edges = np.histogram(projected[y == k], bins=n_bins, density=False)
            hist_list.append(hist)
            edges_list.append(bin_edges)
        data["hist"].append(hist_list)
        data["bin_edges"].append(edges_list)

    data["hist"] = np.asarray(data["hist"])
    data["bin_edges"] = np.asarray(data["bin_edges"])
    data["thetas"] = np.asarray(data["thetas"])
    data["n_classes"] = n_classes
    data["n_train"] = len(x)
    data["class_prob"] = class_counts / len(x)
    return data


def psi_bin_val(x, y, psi_bin_data: dict, n_projs: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate histogram PSI projections."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.eye(psi_bin_data["n_classes"])[y.astype(int)]
    pmi_list = []
    n_bins = psi_bin_data["hist"].shape[2]
    n_class_list = np.asarray(psi_bin_data["class_prob"] * psi_bin_data["n_train"]).astype(int)
    projected = np.dot(x, psi_bin_data["thetas"][:n_projs].T)

    for m in range(n_projs):
        p_theta_given_y = []
        for k in range(psi_bin_data["n_classes"]):
            bin_idx = np.clip(np.digitize(projected[:, m], psi_bin_data["bin_edges"][m][k]), 1, n_bins)
            p = psi_bin_data["hist"][m][k][bin_idx - 1] / max(n_class_list[k], 1)
            p_theta_given_y.append(p)
        p_theta_given_y = np.asarray(p_theta_given_y)
        p_theta = np.sum(psi_bin_data["class_prob"][:, np.newaxis] * p_theta_given_y, axis=0)
        numerator = np.sum(y * p_theta_given_y.T, axis=1)
        pmi_list.append(np.log2(np.clip(numerator, 1e-5, None)) - np.log2(np.clip(p_theta, 1e-5, None)))

    pmi_arr = np.asarray(pmi_list).T
    return np.mean(pmi_arr, axis=1), pmi_arr


def psi_gauss_train(x, y, n_projs: int) -> dict:
    """Fit Gaussian PSI projection densities."""
    y = labels_to_indices(y)
    n_classes = int(np.max(y) + 1)
    data = {"mu": [], "std": [], "thetas": []}
    class_counts = np.array([np.sum(y == k) for k in range(n_classes)])

    for _ in range(n_projs):
        theta = sample_from_sphere(x.shape[1])
        data["thetas"].append(theta.squeeze())
        projected = np.dot(x, theta).squeeze()
        mu_list, std_list = [], []
        for k in range(n_classes):
            mu, std = stats.norm.fit(projected[y == k])
            mu_list.append(mu)
            std_list.append(max(std, 1e-8))
        data["mu"].append(mu_list)
        data["std"].append(std_list)

    data["mu"] = np.asarray(data["mu"])
    data["std"] = np.asarray(data["std"])
    data["thetas"] = np.asarray(data["thetas"])
    data["n_classes"] = n_classes
    data["class_prob"] = class_counts / len(x)
    return data


def psi_gauss_val(x, y, psi_gauss_data: dict, n_projs: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Gaussian PSI projections."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.eye(psi_gauss_data["n_classes"])[y.astype(int)]
    projected = np.dot(x, psi_gauss_data["thetas"][:n_projs].T)
    pmi_list = []
    for m in range(n_projs):
        p_theta_given_y = []
        for k in range(psi_gauss_data["n_classes"]):
            p = stats.norm.pdf(projected[:, m], psi_gauss_data["mu"][m][k], psi_gauss_data["std"][m][k])
            p_theta_given_y.append(p)
        p_theta_given_y = np.asarray(p_theta_given_y)
        p_theta = np.sum(psi_gauss_data["class_prob"][:, np.newaxis] * p_theta_given_y, axis=0)
        numerator = np.sum(y * p_theta_given_y.T, axis=1)
        pmi_list.append(np.log2(np.clip(numerator, 1e-5, None)) - np.log2(np.clip(p_theta, 1e-5, None)))
    pmi_arr = np.asarray(pmi_list).T
    return np.mean(pmi_arr, axis=1), pmi_arr


def psi_rf_train(x, y, n_projs: int, save_path: str | Path) -> np.ndarray:
    """Train random-forest PSI estimators on random projections."""
    y = labels_to_indices(y)
    save_path = ensure_dir(save_path)
    model_dir = ensure_dir(save_path / "psi_models")
    all_thetas = []
    for proj in range(n_projs):
        theta = sample_from_sphere(x.shape[1])
        all_thetas.append(theta)
        projected = np.dot(x, theta)
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(projected, y)
        with (model_dir / f"psi_model_{proj + 1}.pkl").open("wb") as handle:
            pickle.dump(clf, handle)
    thetas = np.asarray(all_thetas)
    np.save(save_path / "all_thetas.npy", thetas)
    return thetas


def psi_rf_val(x, labels, thetas, class_prob, n_projs: int, save_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate random-forest PSI estimators."""
    labels = labels_to_indices(labels)
    model_dir = Path(save_path) / "psi_models"
    pmi_list = []
    for proj in range(n_projs):
        projected = np.dot(x, thetas[proj])
        with (model_dir / f"psi_model_{proj + 1}.pkl").open("rb") as handle:
            clf = pickle.load(handle)
        pred_prob = clf.predict_proba(projected)
        p_y_given_theta = np.clip(pred_prob[np.arange(len(labels)), labels], 1e-5, None)
        p_y = np.clip(class_prob[labels], 1e-5, None)
        pmi_list.append(np.log2(p_y_given_theta / p_y))
    pmi_arr = np.asarray(pmi_list).T
    return np.mean(pmi_arr, axis=1), pmi_arr


def psi_neural_train(ds, config: dict, n_projs: int, save_path: str | Path) -> np.ndarray:
    """Train neural PSI estimators on random projections."""
    save_path = ensure_dir(save_path)
    model_dir = ensure_dir(save_path / "psi_models")
    all_thetas = []
    feature_dim = int(ds.element_spec[0].shape[-1])
    for proj in range(n_projs):
        theta = np.float32(sample_from_sphere(feature_dim))
        all_thetas.append(theta)
        ds_theta = ds.map(lambda x, y: (tf.tensordot(x, theta, axes=1), y))
        ds_theta = prefetch_dataset(ds_theta, batch_size=config["batch_size"])
        model = mlp((1,), config["num_classes"], n_layers=1, n_hidden=128)
        optimizer = get_optimizer(config["optimizer"], learning_rate=0.01)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(ds_theta, epochs=100, callbacks=[early_stop], verbose=1)
        model.save(model_dir / f"psi_model_{proj + 1}.keras")
    thetas = np.asarray(all_thetas)
    np.save(save_path / "all_thetas.npy", thetas)
    return thetas


def psi_neural_val(ds, thetas, class_prob, config: dict, n_projs: int, save_path: str | Path):
    """Evaluate neural PSI estimators."""
    labels = labels_to_indices(np.concatenate([y.numpy() for _, y in ds], axis=0))
    p_y = np.clip(class_prob[labels], 1e-5, None)
    pmi_list = []
    for proj in range(n_projs):
        model = tf.keras.models.load_model(Path(save_path) / "psi_models" / f"psi_model_{proj + 1}.keras")
        ds_theta = ds.map(lambda x, y: (tf.tensordot(x, thetas[proj], axes=1), y)).batch(config["batch_size"])
        class_conditional_prob = np.max(model.predict(ds_theta, verbose=0), axis=1)
        pmi_list.append(np.log2(np.clip(class_conditional_prob, 1e-5, None) / p_y))
    pmi_arr = np.asarray(pmi_list).T
    return np.mean(pmi_arr, axis=1), pmi_arr
