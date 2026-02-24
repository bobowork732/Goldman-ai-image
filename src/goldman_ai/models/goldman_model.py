"""High-level model wrapper used by inference pipelines."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from goldman_ai.config import InferenceConfig


class GoldmanAIModel:
    """Thin wrapper around a model backend configuration.

    This class is intentionally lightweight so it can be swapped with a
    framework-specific implementation later.
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()

    def load(self) -> None:
        """Placeholder for lazy-loading model weights/resources."""

    def get_generation_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Merge defaults with request-level overrides."""
        kwargs = asdict(self.config)
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        return kwargs
