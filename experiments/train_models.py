"""Train benchmark models."""

from __future__ import annotations

from configs.benchmark_configs import get_benchmark_config
from configs.base_config import experiment_dir
from core.datasets import prepare_datasets
from core.models import train_model
from core.utils import save_json, set_seed


def run(args=None, config: dict | None = None):
    """Train the selected benchmark model."""
    config = config or get_benchmark_config(args.benchmark)
    set_seed(config["seed"])
    output_dir = experiment_dir(config, run=getattr(args, "run", 1))
    bundle = prepare_datasets(config)
    model, history = train_model(bundle.train, bundle.val, config, save_dir=output_dir)
    save_json(config, output_dir / "config.json")
    return {"model": model, "history": history.history, "output_dir": output_dir}
