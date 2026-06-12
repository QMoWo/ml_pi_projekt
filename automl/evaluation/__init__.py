"""Evaluation utilities for anomaly detection."""

from .metrics import f1_from_scores, pr_auc, roc_auc
from .runner import EvaluationResult, evaluate_detector

__all__ = ["EvaluationResult", "evaluate_detector", "f1_from_scores", "pr_auc", "roc_auc"]