"""Benchmark model-dataset pairs used in the paper experiments."""

from __future__ import annotations

from configs.base_config import merge_config


BENCHMARKS = {
    "mlp_mnist": {
        "model": "mlp",
        "dataset": "mnist",
        "num_classes": 10,
        "input_shape": (28, 28, 1),
        "image_size": (28, 28),
        "batch_size": 128,
        "max_epoch": 100,
    },
    "cnn_fashion_mnist": {
        "model": "cnn",
        "dataset": "fashion_mnist",
        "num_classes": 10,
        "input_shape": (28, 28, 1),
        "image_size": (28, 28),
        "batch_size": 128,
        "max_epoch": 100,
    },
    "vgg16_stl10": {
        "model": "vgg16",
        "dataset": "stl10",
        "num_classes": 10,
        "input_shape": (96, 96, 3),
        "image_size": (96, 96),
        "batch_size": 64,
        "max_epoch": 100,
    },
    "resnet50_cifar10": {
        "model": "resnet50",
        "dataset": "cifar10",
        "num_classes": 10,
        "input_shape": (32, 32, 3),
        "image_size": (32, 32),
        "batch_size": 128,
        "max_epoch": 100,
    },
    "resnet101_cifar100": {
        "model": "resnet101",
        "dataset": "cifar100",
        "num_classes": 100,
        "input_shape": (32, 32, 3),
        "image_size": (32, 32),
        "batch_size": 128,
        "max_epoch": 100,
    },
    "inceptionv3_stanford_dogs": {
        "model": "inceptionv3",
        "dataset": "stanford_dogs",
        "num_classes": 120,
        "input_shape": (224, 224, 3),
        "image_size": (224, 224),
        "batch_size": 32,
        "max_epoch": 100,
    },
    "densenet121_tinyimagenet": {
        "model": "densenet121",
        "dataset": "tiny_imagenet",
        "num_classes": 200,
        "input_shape": (64, 64, 3),
        "image_size": (64, 64),
        "batch_size": 64,
        "max_epoch": 100,
    },
}


DEFAULT_BENCHMARK = "mlp_mnist"


def get_benchmark_config(name: str = DEFAULT_BENCHMARK, **overrides) -> dict:
    """Return a merged benchmark configuration by name."""
    if name not in BENCHMARKS:
        names = ", ".join(sorted(BENCHMARKS))
        raise KeyError(f"Unknown benchmark '{name}'. Available benchmarks: {names}")
    return merge_config(BENCHMARKS[name], overrides)
