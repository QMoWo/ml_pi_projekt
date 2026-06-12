"""Registry for interchangeable anomaly detectors."""

from collections.abc import Callable

from .detectors import IsolationForestDetector, LocalOutlierFactorDetector


class DetectorRegistry:
    """Map detector names to factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., object]] = {}

    def register(self, name: str, factory: Callable[..., object]) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs: object) -> object:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"Unknown detector: {name}") from exc
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)


def build_default_registry() -> DetectorRegistry:
    """Create a registry with the initial interchangeable detectors."""

    registry = DetectorRegistry()
    registry.register("isolation_forest", IsolationForestDetector)
    registry.register(
        "local_outlier_factor",
        lambda contamination=0.05, random_state=None: LocalOutlierFactorDetector(
            contamination=contamination,
            n_neighbors=20,
        ),
    )
    return registry