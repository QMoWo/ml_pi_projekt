"""High-level orchestration for AutoML anomaly detection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AutoMLConfig
from .data.tep import TEPSplits, load_tep_splits
from .registry import DetectorRegistry, build_default_registry
from .evaluation.runner import EvaluationResult, evaluate_detector
from .search.random_search import RandomSearch
from .search.space import build_default_search_space


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

    def run_random_search(self, dataset: TEPSplits, detector_names: list[str] | None = None) -> AutoMLResult:
        """Search detector/parameter combinations with Random Search and return the best result."""

        selected_detectors = detector_names or self.registry.names()
        search_space = build_default_search_space(random_state=self.config.random_state)
        search = RandomSearch(parameter_space=search_space, random_state=self.config.random_state)
        candidates = search.suggest(selected_detectors, self.config.max_trials)

        best_result: AutoMLResult | None = None
        best_score = float("-inf")

        for candidate in candidates:
            detector_name = str(candidate["detector"])
            parameters = dict(candidate["parameters"])
            detector = self.registry.create(detector_name, **parameters)
            evaluation: EvaluationResult = evaluate_detector(
                detector=detector,
                train_dataset=dataset.training_dataset(),
                test_dataset=dataset.evaluation_dataset(),
                contamination=self.config.contamination,
            )

            score = evaluation.metrics.get(self.config.metric)
            if score is None:
                continue

            result = AutoMLResult(
                detector_name=detector_name,
                parameters=parameters,
                metrics=evaluation.metrics,
            )

            if score > best_score:
                best_score = score
                best_result = result

        if best_result is None:
            raise ValueError(f"No detector produced the requested metric: {self.config.metric}")

        return best_result


def run_minimal_workflow(
    data_dir: str | Path,
    config: AutoMLConfig | None = None,
    detector_name: str = "isolation_forest",
) -> AutoMLResult:
    """Load the TEP splits, train the baseline detector, and return metrics."""

    resolved_config = config or AutoMLConfig()
    pipeline = AutoMLPipeline(resolved_config)
    splits = load_tep_splits(data_dir)
    return pipeline.run(splits, detector_names=[detector_name])


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


def run_random_search_workflow(
    data_dir: str | Path,
    detector_names: list[str] | None = None,
    config: AutoMLConfig | None = None,
) -> AutoMLResult:
    """Run the pipeline with Random Search over detector parameters."""

    resolved_config = config or AutoMLConfig()
    pipeline = AutoMLPipeline(resolved_config)
    splits = load_tep_splits(data_dir)
    return pipeline.run_random_search(splits, detector_names=detector_names)


def run_search_workflow(
    data_dir: str | Path,
    detector_names: list[str] | None = None,
    config: AutoMLConfig | None = None,
) -> AutoMLResult:
    """Run the default random-search search mode over all or selected detectors."""

    return run_random_search_workflow(data_dir, detector_names=detector_names, config=config)