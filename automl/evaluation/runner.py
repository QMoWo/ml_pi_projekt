"""End-to-end evaluation of a detector on a dataset."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..data.tep import TEPDataset
from ..detectors.base import BaseDetector
from .metrics import f1_from_scores, pr_auc, roc_auc


@dataclass(slots=True)
class EvaluationResult:
    """Metrics collected for a single detector run."""

    metrics: dict[str, float]
    train_time_seconds: float
    threshold: float | None


def evaluate_detector(
    detector: BaseDetector,
    train_dataset: TEPDataset,
    test_dataset: TEPDataset,
    *,
    contamination: float = 0.05,
    threshold: float | None = None,
) -> EvaluationResult:
    start_time = perf_counter()
    detector.fit(train_dataset.features, train_dataset.labels)
    train_time_seconds = perf_counter() - start_time

    test_scores = np.asarray(detector.score_samples(test_dataset.features), dtype=float)
    metrics: dict[str, float] = {"train_time_seconds": train_time_seconds}
    resolved_threshold = threshold

    if resolved_threshold is None:
        train_scores = np.asarray(detector.score_samples(train_dataset.features), dtype=float)
        resolved_threshold = float(np.quantile(train_scores, 1.0 - contamination))

    if test_dataset.labels is not None:
        labels = np.asarray(test_dataset.labels)
        metrics["pr_auc"] = pr_auc(labels, test_scores)
        metrics["roc_auc"] = roc_auc(labels, test_scores)
        if resolved_threshold is not None:
            metrics["f1"] = f1_from_scores(labels, test_scores, resolved_threshold)

    return EvaluationResult(metrics=metrics, train_time_seconds=train_time_seconds, threshold=resolved_threshold)