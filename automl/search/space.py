"""Default search spaces for interchangeable anomaly detectors."""

from __future__ import annotations


def build_default_search_space(*, random_state: int | None = None) -> dict[str, dict[str, list[object]]]:
    """Return a small search space for the built-in detectors."""

    return {
        "isolation_forest": {
            "contamination": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "max_samples": ["auto", 0.5, 0.8],
            "max_features": [0.5, 1.0],
            "bootstrap": [False, True],
            "random_state": [random_state],
        },
        "local_outlier_factor": {
            "contamination": [0.01, 0.03, 0.05, 0.1],
            "n_neighbors": [10, 20, 30, 50],
            "leaf_size": [20, 30, 40],
            "metric": ["minkowski", "euclidean"],
            "algorithm": ["auto", "ball_tree"],
        },
        "one_class_svm": {
            "contamination": [0.01, 0.03, 0.05, 0.1],
            "nu": [0.01, 0.03, 0.05, 0.1],
            "kernel": ["rbf", "sigmoid", "linear"],
            "gamma": ["scale", "auto"],
            "degree": [2, 3, 4],
            "coef0": [0.0, 0.1, 0.5],
        },
    }