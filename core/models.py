"""TensorFlow/Keras model creation and training."""

from __future__ import annotations

import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import densenet, inception_v3, resnet, vgg16

from core.utils import ensure_dir


def get_optimizer(optimizer_name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    """Instantiate a Keras optimizer by name."""
    optimizer_class = getattr(tf.keras.optimizers, optimizer_name)
    return optimizer_class(learning_rate=learning_rate)


def build_model(config: dict) -> tf.keras.Model:
    """Build a Keras model from a benchmark configuration."""
    model_name = config["model"].lower()
    input_shape = tuple(config["input_shape"])
    num_classes = int(config["num_classes"])

    if model_name == "mlp":
        return mlp(input_shape, num_classes)
    if model_name == "cnn":
        return cnn(input_shape, num_classes)
    if model_name == "vgg16":
        return application_model(vgg16.VGG16, input_shape, num_classes, config)
    if model_name == "resnet50":
        return application_model(resnet.ResNet50, input_shape, num_classes, config)
    if model_name == "resnet101":
        return application_model(resnet.ResNet101, input_shape, num_classes, config)
    if model_name == "inceptionv3":
        return application_model(inception_v3.InceptionV3, input_shape, num_classes, config)
    if model_name == "densenet121":
        return application_model(densenet.DenseNet121, input_shape, num_classes, config)

    raise NotImplementedError(f"Model '{model_name}' is not implemented.")


def mlp(input_shape: tuple[int, ...], num_classes: int, n_layers: int = 3, n_hidden: int = 512) -> tf.keras.Model:
    """Build the MLP used for MNIST."""
    model = tf.keras.Sequential(name="mlp")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    for _ in range(n_layers):
        model.add(layers.Dense(n_hidden, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


def cnn(input_shape: tuple[int, ...], num_classes: int) -> tf.keras.Model:
    """Build the CNN used for Fashion-MNIST."""
    model = tf.keras.Sequential(name="cnn")
    model.add(layers.Input(shape=input_shape))
    for filters in (32, 64, 128):
        model.add(layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


def application_model(
    constructor,
    input_shape: tuple[int, int, int],
    num_classes: int,
    config: dict,
) -> tf.keras.Model:
    """Build an ImageNet-style backbone with the project classification head."""
    base_model = constructor(
        include_top=False,
        input_shape=input_shape,
        weights=config.get("model_weights"),
    )
    base_model.trainable = not config.get("freeze_backbone", False)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=base_model.input, outputs=outputs, name=config["model"])


def compile_model(model: tf.keras.Model, config: dict) -> tf.keras.Model:
    """Compile a classifier with categorical cross-entropy."""
    optimizer = get_optimizer(config["optimizer"], config["learning_rate"])
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def train_model(
    ds_train: tf.data.Dataset,
    ds_val: tf.data.Dataset,
    config: dict,
    save_dir: str | Path | None = None,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Build, train, and optionally save a benchmark model."""
    model = compile_model(build_model(config), config)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config["patience"],
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        ds_train,
        epochs=config["max_epoch"],
        validation_data=ds_val,
        callbacks=callbacks,
        verbose=1,
    )

    if save_dir is not None:
        save_dir = ensure_dir(save_dir)
        model.save(save_dir / "trained_model.keras")
        with (save_dir / "history.pickle").open("wb") as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model, history


def load_trained_model(path: str | Path) -> tf.keras.Model:
    """Load a Keras model from disk."""
    return tf.keras.models.load_model(path)
