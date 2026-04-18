# Pointwise Information Measures as Confidence Estimators in Deep Neural Networks

This is the code repository for the paper "Pointwise Information Measures as Confidence Estimators in Deep Neural Networks: A Comparative Study" published in ICML 2025.

Read the blog here:

The current codebase focuses on reproducible Python experiments for:

- computing PMI, PSI, and PVI as confidence estimators,
- evaluating failure prediction and confidence calibration,
- benchmarking across standard vision datasets and architectures,
- comparing against standard post-hoc confidence baselines.


## Research Questions

This project is organized around three research questions:

- Can pointwise information measures provide better confidence estimates than standard post-hoc baselines?
- How do PMI, PSI, and PVI compare for failure prediction tasks such as misclassification detection and selective prediction?
- How well do these measures perform for confidence calibration?

## Repository Structure

```text
project_root/
  README.md
  requirements.txt

  configs/
    base_config.py
    benchmark_configs.py
    estimator_configs.py

  core/
    datasets.py
    models.py
    utils.py
    metrics.py
    plotting.py
    feature_extraction.py

  estimators/
    pmi_estimators.py
    pvi_estimators.py
    psi_estimators.py

  experiments/
    train_models.py
    run_pmi_analysis.py
    run_pvi_analysis.py
    run_psi_analysis.py
    run_filtering_error.py
    run_reliability_diagram.py
    run_calibration_error.py
    run_selective_prediction.py
    run_umap.py

  scripts/
    main.py
```

## Method Overview
This project studies three pointwise information measures:
1. PMI — measures the pointwise association between features and labels.
2. PSI — a sliced variant of PMI designed to scale better in high dimensions.
3. PVI — measures usable predictive information under a restricted predictive family.

The repository includes implementations for estimating these measures and evaluating them as confidence scores on trained image classifiers. The experiments cover both:
- Failure prediction
-- misclassification detection
- selective prediction
-- Confidence calibration

## Dataset Notes

MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 are loaded automatically through `tf.keras.datasets`. STL-10 and Stanford Dogs are loaded through TensorFlow Datasets where available. TinyImageNet support is provided through TensorFlow Datasets when available, with a local-folder fallback.

For local datasets such as Stanford Dogs or TinyImageNet, set `local_data_path` in the selected configuration. The expected directory layout is:

```text
dataset_root/
  train/
    class_001/
    class_002/
  val/
    class_001/
    class_002/
  test/
    class_001/
    class_002/
```

## Benchmark Pairs

The default benchmark configurations are:

| Benchmark key | Model | Dataset |
| --- | --- | --- |
| `mlp_mnist` | MLP | MNIST |
| `cnn_fashion_mnist` | CNN | Fashion-MNIST |
| `vgg16_stl10` | VGG16 | STL-10 |
| `resnet50_cifar10` | ResNet50 | CIFAR-10 |
| `resnet101_cifar100` | ResNet101 | CIFAR-100 |
| `inceptionv3_stanford_dogs` | InceptionV3 | Stanford Dogs |
| `densenet121_tinyimagenet` | DenseNet121 | TinyImageNet |

## Experiments

Train a benchmark model:

```bash
python scripts/main.py train_models --benchmark mlp_mnist
```

Run estimator analyses:

```bash
python scripts/main.py pmi_analysis --benchmark mlp_mnist
python scripts/main.py pvi_analysis --benchmark mlp_mnist
python scripts/main.py psi_analysis --benchmark mlp_mnist --psi-estimator histogram
```

Run uncertainty and calibration experiments:

```bash
python scripts/main.py filtering_error --benchmark mlp_mnist
python scripts/main.py reliability_diagram --benchmark mlp_mnist
python scripts/main.py calibration_error --benchmark mlp_mnist
python scripts/main.py selective_prediction --benchmark mlp_mnist
python scripts/main.py umap --benchmark mlp_mnist
```

## Reproducibility Notes

- Results in the paper are averaged over multiple runs.
- All confidence methods are evaluated in a post-hoc setting.
- Temperature scaling is used where appropriate for fair comparison across methods.
- PMI is omitted for some very large-class benchmarks due to computational cost.

## Citation

```bibtex
@article{wongso2025pointwise,
  title   = {Pointwise Information Measures as Confidence Estimators in Deep Neural Networks: A Comparative Study},
  author  = {Wongso, Shelvia and Ghosh, Rohan and Motani, Mehul},
  journal = {Proceedings of the 42nd International Conference on Machine Learning},
  year    = {2025},
  url = {https://proceedings.mlr.press/v206/wongso23a.html}
}
```
