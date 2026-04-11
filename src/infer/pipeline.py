"""Inference pipeline for text-to-image generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from src.diffusion.scheduler import DiffusionScheduler
from src.infer.samplers import build_sampler
from src.models.text_encoder import TextEncoder
from src.safety.filter import BasicSafetyFilter
from src.models.unet import UNetDenoiser
from src.models.vae import VAE


@dataclass
class GenerationConfig:
    steps: int = 50
    guidance_scale: float = 7.5
    height: int = 256
    width: int = 256
    sampler: str = "ddim"
    eta: float = 0.0


class LatentDiffusionPipeline:
    def __init__(
        self,
        vae: VAE,
        unet: UNetDenoiser,
        text_encoder: TextEncoder,
        scheduler: DiffusionScheduler,
        device: torch.device,
    ) -> None:
        self.vae = vae.eval().to(device)
        self.unet = unet.eval().to(device)
        self.text_encoder = text_encoder.eval().to(device)
        self.scheduler = scheduler.to(device)
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        text_model_name: str | None = None,
    ) -> "LatentDiffusionPipeline":
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        vae = VAE()
        vae.load_state_dict(ckpt["vae"])

        text_encoder = TextEncoder(model_name=text_model_name)
        text_encoder.load_state_dict(ckpt["text_encoder"])

        unet = UNetDenoiser(text_embed_dim=text_encoder.embed_dim)
        if "ema_unet" in ckpt:
            unet.load_state_dict(ckpt["ema_unet"])
        else:
            unet.load_state_dict(ckpt["unet"])

        scheduler = DiffusionScheduler()
        return cls(vae=vae, unet=unet, text_encoder=text_encoder, scheduler=scheduler, device=torch.device(device))

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        seed: int,
        cfg: GenerationConfig,
        safety_filter: BasicSafetyFilter | None = None,
    ) -> torch.Tensor:
        self.set_seed(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        bsz = len(prompts)
        latent_h = cfg.height // 8
        latent_w = cfg.width // 8

        latents = torch.randn((bsz, 4, latent_h, latent_w), generator=generator, device=self.device)

        cond_ids = self.text_encoder.tokenize_to_ids(list(prompts)).to(self.device)
        uncond_ids = self.text_encoder.tokenize_to_ids(list(negative_prompts)).to(self.device)
        cond_emb = self.text_encoder.encode_tokens(cond_ids)
        uncond_emb = self.text_encoder.encode_tokens(uncond_ids)

        sampler = build_sampler(cfg.sampler, self.scheduler, steps=cfg.steps, eta=cfg.eta)

        for i, t in enumerate(sampler.timesteps):
            t_prev = sampler.timesteps[i + 1] if i + 1 < len(sampler.timesteps) else 0
            t_batch = torch.full((bsz,), t, device=self.device, dtype=torch.long)

            eps_uncond = self.unet(latents, t_batch, uncond_emb)
            eps_cond = self.unet(latents, t_batch, cond_emb)
            eps = eps_uncond + cfg.guidance_scale * (eps_cond - eps_uncond)

            latents = sampler.step(
                model_output=eps,
                sample=latents,
                t=t,
                t_prev=t_prev,
                generator=generator,
            )

        decoded = self.vae.decode(latents).clamp(-1, 1)
        images = (decoded + 1.0) / 2.0
        images = images.clamp(0, 1)
        if safety_filter is not None:
            images = safety_filter.mask_unsafe_images(images, list(prompts))
        return images
