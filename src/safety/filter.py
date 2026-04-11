"""Safety filter integration points for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import torch


@dataclass
class SafetyDecision:
    is_safe: bool
    reason: str = ""


class BasicSafetyFilter:
    """Keyword + optional image-model safety filter.

    This is intentionally minimal and designed as an extension point:
    - prompt keyword deny-list checks
    - optional external classifier hook on generated tensors
    """

    def __init__(
        self,
        enabled: bool = True,
        blocked_keywords: Iterable[str] | None = None,
        image_classifier: Callable[[torch.Tensor], List[bool]] | None = None,
    ) -> None:
        self.enabled = enabled
        self.blocked_keywords = set((blocked_keywords or ["nsfw", "explicit", "gore", "nudity"]))
        self.image_classifier = image_classifier

    def check_prompt(self, prompt: str) -> SafetyDecision:
        if not self.enabled:
            return SafetyDecision(is_safe=True)

        p = prompt.lower()
        for kw in self.blocked_keywords:
            if kw in p:
                return SafetyDecision(is_safe=False, reason=f"blocked_keyword:{kw}")
        return SafetyDecision(is_safe=True)

    def filter_prompts(self, prompts: List[str]) -> List[SafetyDecision]:
        return [self.check_prompt(p) for p in prompts]

    def mask_unsafe_images(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        """Replace unsafe outputs with black images."""
        if not self.enabled:
            return images

        decisions = self.filter_prompts(prompts)

        if self.image_classifier is not None:
            clf_safe = self.image_classifier(images)
            for i, safe in enumerate(clf_safe):
                if not safe:
                    decisions[i] = SafetyDecision(is_safe=False, reason="image_classifier")

        output = images.clone()
        for i, d in enumerate(decisions):
            if not d.is_safe:
                output[i] = 0.0
        return output
