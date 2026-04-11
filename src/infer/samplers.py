"""Sampling algorithms for latent diffusion inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from src.diffusion.scheduler import DiffusionScheduler


@dataclass
class SamplerConfig:
    name: str = "ddim"
    steps: int = 50
    eta: float = 0.0


class BaseSampler:
    def __init__(self, scheduler: DiffusionScheduler, steps: int) -> None:
        self.scheduler = scheduler
        self.steps = steps
        self.timesteps = self._make_timesteps(steps)

    def _make_timesteps(self, steps: int) -> List[int]:
        vals = torch.linspace(self.scheduler.num_train_steps - 1, 0, steps=steps)
        return [int(v.item()) for v in vals]

    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        t: int,
        t_prev: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        raise NotImplementedError


class DDPMSampler(BaseSampler):
    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        t: int,
        t_prev: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        del t_prev, generator
        return self.scheduler.step(model_output, t, sample)


class DDIMSampler(BaseSampler):
    def __init__(self, scheduler: DiffusionScheduler, steps: int, eta: float = 0.0) -> None:
        super().__init__(scheduler, steps)
        self.eta = eta

    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        t: int,
        t_prev: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        b = sample.shape[0]
        t_batch = torch.full((b,), t, device=sample.device, dtype=torch.long)
        prev_batch = torch.full((b,), max(t_prev, 0), device=sample.device, dtype=torch.long)

        alpha_t = self.scheduler._extract(self.scheduler.tensors.alphas_cumprod, t_batch, sample.shape)
        alpha_prev = self.scheduler._extract(self.scheduler.tensors.alphas_cumprod, prev_batch, sample.shape)
        pred_x0 = self.scheduler.predict_x0(sample, model_output, t_batch).clamp(-1, 1)

        sigma = (
            self.eta
            * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
            * torch.sqrt((1 - alpha_t / alpha_prev).clamp(min=0.0))
        )
        noise = torch.randn_like(sample, generator=generator) if self.eta > 0 else torch.zeros_like(sample)

        dir_xt = torch.sqrt((1 - alpha_prev - sigma**2).clamp(min=0.0)) * model_output
        prev_sample = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma * noise
        return prev_sample


class EulerSampler(BaseSampler):
    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        t: int,
        t_prev: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        del generator
        b = sample.shape[0]
        t_batch = torch.full((b,), t, device=sample.device, dtype=torch.long)
        prev_batch = torch.full((b,), max(t_prev, 0), device=sample.device, dtype=torch.long)

        alpha_t = self.scheduler._extract(self.scheduler.tensors.alphas_cumprod, t_batch, sample.shape)
        alpha_prev = self.scheduler._extract(self.scheduler.tensors.alphas_cumprod, prev_batch, sample.shape)

        sigma_t = torch.sqrt((1 - alpha_t) / alpha_t)
        sigma_prev = torch.sqrt((1 - alpha_prev) / alpha_prev)

        pred_x0 = self.scheduler.predict_x0(sample, model_output, t_batch)
        d = (sample - pred_x0) / sigma_t.clamp(min=1e-8)
        dt = sigma_prev - sigma_t
        return sample + d * dt


def build_sampler(name: str, scheduler: DiffusionScheduler, steps: int, eta: float = 0.0) -> BaseSampler:
    name = name.lower()
    if name == "ddpm":
        return DDPMSampler(scheduler=scheduler, steps=steps)
    if name == "ddim":
        return DDIMSampler(scheduler=scheduler, steps=steps, eta=eta)
    if name == "euler":
        return EulerSampler(scheduler=scheduler, steps=steps)
    raise ValueError(f"Unknown sampler: {name}. Expected one of: ddpm, ddim, euler.")
