"""Search strategies for AutoML."""

from .base import SearchStrategy
from .random_search import RandomSearch

__all__ = ["RandomSearch", "SearchStrategy"]