"""High-level orchestration for AutoML anomaly detection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AutoMLConfig
from .data.tep import TEPSplits, load_tep_splits
from .registry import DetectorRegistry, build_default_registry
from .evaluation.runner import EvaluationResult, evaluate_detector


@dataclass(slots=True)
class AutoMLResult:
    """Result of a completed AutoML run."""

    detector_name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]


class AutoMLPipeline:
    """Wire together registry, search, and evaluation components."""

    def __init__(self, config: AutoMLConfig, registry: DetectorRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or build_default_registry()

    def run(self, dataset: TEPSplits, detector_names: list[str] | None = None) -> AutoMLResult:
        selected_detectors = detector_names or self.registry.names()
        best_result: AutoMLResult | None = None
        best_score = float("-inf")

        for detector_name in selected_detectors:
            detector = self.registry.create(
                detector_name,
                contamination=self.config.contamination,
                random_state=self.config.random_state,
            )
            evaluation: EvaluationResult = evaluate_detector(
                detector=detector,
                train_dataset=dataset.training_dataset(),
                test_dataset=dataset.evaluation_dataset(),
                contamination=self.config.contamination,
            )

            score = evaluation.metrics.get(self.config.metric)
            if score is None:
                continue

            candidate = AutoMLResult(
                detector_name=detector_name,
                parameters={
                    "contamination": self.config.contamination,
                    "random_state": self.config.random_state,
                },
                metrics=evaluation.metrics,
            )

            if score > best_score:
                best_score = score
                best_result = candidate

        if best_result is None:
            raise ValueError(f"No detector produced the requested metric: {self.config.metric}")

        return best_result


def run_minimal_workflow(data_dir: str | Path, config: AutoMLConfig | None = None) -> AutoMLResult:
    """Load the TEP splits, train the baseline detector, and return metrics."""

    resolved_config = config or AutoMLConfig()
    pipeline = AutoMLPipeline(resolved_config)
    splits = load_tep_splits(data_dir)
    return pipeline.run(splits, detector_names=["isolation_forest"])


def run_comparison_workflow(
    data_dir: str | Path,
    detector_names: list[str] | None = None,
    config: AutoMLConfig | None = None,
) -> AutoMLResult:
    """Run the pipeline on an explicit list of detectors, or all registered ones."""

    resolved_config = config or AutoMLConfig()
    pipeline = AutoMLPipeline(resolved_config)
    splits = load_tep_splits(data_dir)
    return pipeline.run(splits, detector_names=detector_names)