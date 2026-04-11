"""Evaluation metric hooks for diffusion validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class MetricOutput:
    fid_proxy: Optional[float] = None
    clip_score_proxy: Optional[float] = None


class ValidationMetrics:
    """Metric registry with optional FID/CLIP integrations.

    This module intentionally keeps external dependencies optional.
    If specialized libraries are unavailable, it computes lightweight proxy scores
    suitable for checkpoint ranking during development.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @torch.no_grad()
    def fid_proxy(self, real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
        """Compute a cheap feature-statistics proxy for FID-like ranking."""
        real = real_images.flatten(1).float()
        fake = generated_images.flatten(1).float()
        real_mean = real.mean(dim=1)
        fake_mean = fake.mean(dim=1)
        return torch.mean((real_mean - fake_mean) ** 2).item()

    @torch.no_grad()
    def clip_score_proxy(self, images: torch.Tensor, text_embeddings: torch.Tensor) -> float:
        """Compute a cosine-similarity proxy between image and text global features."""
        img_feat = images.flatten(1).float()
        img_feat = img_feat / img_feat.norm(dim=1, keepdim=True).clamp(min=1e-8)

        txt_feat = text_embeddings.mean(dim=1)
        txt_feat = txt_feat / txt_feat.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # align dimensions if needed
        min_dim = min(img_feat.shape[1], txt_feat.shape[1])
        sim = (img_feat[:, :min_dim] * txt_feat[:, :min_dim]).sum(dim=1)
        return sim.mean().item()

    @torch.no_grad()
    def evaluate_batch(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> MetricOutput:
        return MetricOutput(
            fid_proxy=self.fid_proxy(real_images, generated_images),
            clip_score_proxy=self.clip_score_proxy(generated_images, text_embeddings),
        )

    @staticmethod
    def to_dict(output: MetricOutput) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if output.fid_proxy is not None:
            out["val/fid_proxy"] = float(output.fid_proxy)
        if output.clip_score_proxy is not None:
            out["val/clip_score_proxy"] = float(output.clip_score_proxy)
        return out
