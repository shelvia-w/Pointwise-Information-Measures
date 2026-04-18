"""Dataset loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclass
class DatasetBundle:
    """Container for train, validation, test datasets and metadata."""

    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    num_classes: int
    input_shape: tuple[int, int, int]
    info: object | None = None


def load_dataset(config: dict, shuffle: bool | None = None) -> DatasetBundle:
    """Load a benchmark dataset and return unbatched TensorFlow datasets."""
    dataset = config["dataset"].lower()
    shuffle = config.get("shuffle", True) if shuffle is None else shuffle

    if dataset in {"mnist", "fashion_mnist", "cifar10", "cifar100"}:
        return _load_keras_dataset(dataset, config)
    if dataset in {"stl10", "stanford_dogs", "tiny_imagenet"}:
        return _load_tfds_or_local(dataset, config, shuffle=shuffle)

    return _load_tfds_or_local(dataset, config, shuffle=shuffle)


def prepare_datasets(config: dict, shuffle: bool | None = None) -> DatasetBundle:
    """Load, preprocess, batch, and prefetch a dataset."""
    bundle = load_dataset(config, shuffle=shuffle)
    train = preprocess_dataset(bundle.train, config, bundle.num_classes, training=True)
    val = preprocess_dataset(bundle.val, config, bundle.num_classes, training=False)
    test = preprocess_dataset(bundle.test, config, bundle.num_classes, training=False)
    return DatasetBundle(
        train=prefetch_dataset(train, config["batch_size"], cache=config.get("cache", False)),
        val=prefetch_dataset(val, config["batch_size"], cache=config.get("cache", False)),
        test=prefetch_dataset(test, config["batch_size"], cache=False),
        num_classes=bundle.num_classes,
        input_shape=bundle.input_shape,
        info=bundle.info,
    )


def preprocess_dataset(
    dataset: tf.data.Dataset,
    config: dict,
    n_classes: int | None = None,
    training: bool = False,
) -> tf.data.Dataset:
    """Apply resizing, normalization, channel conversion, and one-hot encoding."""
    image_size = tuple(config.get("image_size", config["input_shape"][:2]))
    n_classes = n_classes or config["num_classes"]
    normalize = config.get("normalize", True)
    one_hot = config.get("one_hot", True)

    def _map(image, label):
        image = tf.cast(image, tf.float32)
        if image.shape.rank == 2:
            image = tf.expand_dims(image, axis=-1)
        if image.shape[-1] == 1 and config["input_shape"][-1] == 3:
            image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, image_size)
        if normalize:
            image = image / 255.0
        label = tf.cast(tf.reshape(label, [-1])[0], tf.int32)
        if one_hot:
            label = tf.one_hot(label, depth=n_classes)
        return image, label

    if training and config.get("shuffle", True):
        dataset = dataset.shuffle(config.get("shuffle_buffer", 10000), seed=config.get("seed", 42))
    return dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def prefetch_dataset(
    dataset: tf.data.Dataset,
    batch_size: int,
    cache: bool = False,
) -> tf.data.Dataset:
    """Batch and prefetch a dataset."""
    dataset = dataset.batch(batch_size)
    if cache:
        dataset = dataset.cache()
    return dataset.prefetch(tf.data.AUTOTUNE)


def _load_keras_dataset(name: str, config: dict) -> DatasetBundle:
    loaders = {
        "mnist": tf.keras.datasets.mnist.load_data,
        "fashion_mnist": tf.keras.datasets.fashion_mnist.load_data,
        "cifar10": tf.keras.datasets.cifar10.load_data,
        "cifar100": lambda: tf.keras.datasets.cifar100.load_data(label_mode="fine"),
    }
    (x_train, y_train), (x_test, y_test) = loaders[name]()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    val_size = int(len(x_train) * config.get("validation_split", 0.15))
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return DatasetBundle(
        train=train,
        val=val,
        test=test,
        num_classes=config["num_classes"],
        input_shape=tuple(config["input_shape"]),
    )


def _load_tfds_or_local(name: str, config: dict, shuffle: bool) -> DatasetBundle:
    data_dir = config.get("data_dir")
    local_path = config.get("local_data_path")
    if local_path:
        return _load_local_image_folders(config)

    tfds_name = {
        "stl10": "stl10",
        "stanford_dogs": "stanford_dogs",
        "tiny_imagenet": "tiny_imagenet",
    }.get(name, name)

    try:
        (train, val, test), info = tfds.load(
            tfds_name,
            split=["train[:85%]", "train[85%:]", "test"],
            data_dir=data_dir,
            shuffle_files=shuffle,
            as_supervised=True,
            with_info=True,
        )
    except Exception as exc:
        if name in {"stanford_dogs", "tiny_imagenet"}:
            raise RuntimeError(
                f"Could not load {name} through TensorFlow Datasets. "
                "Set config['local_data_path'] to a directory with train/val/test class folders."
            ) from exc
        raise

    return DatasetBundle(
        train=train,
        val=val,
        test=test,
        num_classes=config["num_classes"],
        input_shape=tuple(config["input_shape"]),
        info=info,
    )


def _load_local_image_folders(config: dict) -> DatasetBundle:
    root = Path(config["local_data_path"])
    image_size = tuple(config["image_size"])
    batch_size = None

    def from_directory(split: str) -> tf.data.Dataset:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Expected local dataset split at {split_dir}")
        return tf.keras.utils.image_dataset_from_directory(
            split_dir,
            labels="inferred",
            label_mode="int",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=(split == "train"),
            seed=config.get("seed", 42),
        )

    return DatasetBundle(
        train=from_directory("train"),
        val=from_directory("val"),
        test=from_directory("test"),
        num_classes=config["num_classes"],
        input_shape=tuple(config["input_shape"]),
    )


def class_priors(dataset: tf.data.Dataset, num_classes: int) -> np.ndarray:
    """Estimate class priors from an unbatched or batched dataset."""
    counts = np.zeros(num_classes, dtype=np.float64)
    total = 0
    for _, labels in dataset:
        labels_np = labels.numpy()
        if labels_np.ndim > 1:
            labels_np = np.argmax(labels_np, axis=-1)
        labels_np = labels_np.reshape(-1)
        for label in labels_np:
            counts[int(label)] += 1
            total += 1
    return counts / max(total, 1)
