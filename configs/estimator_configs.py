"""Default settings for PMI, PVI, PSI, and evaluation experiments."""

PMI_CONFIG = {
    "critic": "separable",
    "pmi_estimator": "probabilistic_classifier",
    "critic_epochs": 50,
    "feature_layer": -2,
}

PVI_CONFIG = {
    "null_epochs": 50,
    "calibrate": False,
    "temperature_iters": 300,
}

PSI_CONFIG = {
    "psi_estimator": "histogram",
    "n_projs": 100,
    "n_bins": 30,
    "feature_layer": -2,
}

EVALUATION_CONFIG = {
    "n_bins": 15,
    "coverage_points": 100,
}
