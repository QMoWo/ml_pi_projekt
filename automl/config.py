"""Configuration objects for the AutoML pipeline."""

from dataclasses import dataclass


@dataclass(slots=True)
class AutoMLConfig:
    """Top-level configuration for search, evaluation, and orchestration."""

    metric: str = "pr_auc"
    max_trials: int = 25
    random_state: int | None = None
    contamination: float = 0.05