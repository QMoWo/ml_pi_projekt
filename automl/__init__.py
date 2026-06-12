"""Core package for modular anomaly-detection AutoML."""

from .config import AutoMLConfig
from .pipeline import AutoMLPipeline, AutoMLResult

__all__ = ["AutoMLConfig", "AutoMLPipeline", "AutoMLResult"]