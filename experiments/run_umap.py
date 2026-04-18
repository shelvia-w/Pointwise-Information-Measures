"""Generate UMAP embeddings for model features."""

from __future__ import annotations

import numpy as np
import umap

from configs.benchmark_configs import get_benchmark_config
from configs.base_config import experiment_dir
from configs.estimator_configs import PSI_CONFIG
from core.datasets import prepare_datasets
from core.feature_extraction import feature_array
from core.models import load_trained_model
from core.plotting import plot_umap_embedding


def run(args=None, config: dict | None = None):
    """Fit UMAP on test features and save embedding data/plot."""
    config = config or get_benchmark_config(args.benchmark)
    config.update(PSI_CONFIG)
    output_dir = experiment_dir(config, run=getattr(args, "run", 1))
    bundle = prepare_datasets(config, shuffle=False)
    model = load_trained_model(output_dir / "trained_model.keras")
    features, labels = feature_array(model, bundle.test, layer=config["feature_layer"])
    reducer = umap.UMAP(random_state=config["seed"])
    embedding = reducer.fit_transform(features)
    np.savez(output_dir / "umap_embedding.npz", embedding=embedding, labels=labels)
    plot_umap_embedding(embedding, labels, save_path=output_dir / "umap_embedding.png")
    return {"embedding": embedding, "output_dir": output_dir}
