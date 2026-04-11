"""Diffusion scheduler utilities for forward noising and reverse denoising."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ScheduleTensors:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor


class DiffusionScheduler:
    """DDPM-style scheduler with helpers for q(x_t|x_0) and p(x_{t-1}|x_t)."""

    def __init__(
        self,
        num_train_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_train_steps = num_train_steps
        self.device = torch.device(device)
        betas = torch.linspace(beta_start, beta_end, num_train_steps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.tensors = ScheduleTensors(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
            posterior_variance=posterior_variance,
        )

    def to(self, device: str | torch.device) -> "DiffusionScheduler":
        self.device = torch.device(device)
        for key, value in self.tensors.__dict__.items():
            setattr(self.tensors, key, value.to(self.device))
        return self

    def _extract(self, a: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        batch_size = timesteps.shape[0]
        out = a.gather(-1, timesteps).float()
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.tensors.sqrt_alphas_cumprod, timesteps, x0.shape)
        sqrt_one_minus = self._extract(self.tensors.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_x0(self, xt: torch.Tensor, noise_pred: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.tensors.sqrt_alphas_cumprod, timesteps, xt.shape)
        sqrt_one_minus = self._extract(self.tensors.sqrt_one_minus_alphas_cumprod, timesteps, xt.shape)
        return (xt - sqrt_one_minus * noise_pred) / sqrt_alpha.clamp(min=1e-8)

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        t = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
        betas_t = self._extract(self.tensors.betas, t, sample.shape)
        alpha_t = self._extract(self.tensors.alphas, t, sample.shape)
        alpha_bar_t = self._extract(self.tensors.alphas_cumprod, t, sample.shape)
        sqrt_one_minus = self._extract(self.tensors.sqrt_one_minus_alphas_cumprod, t, sample.shape)

        mean = (1.0 / torch.sqrt(alpha_t)) * (sample - (betas_t / sqrt_one_minus) * model_output)

        if timestep == 0:
            return mean

        posterior_var = self._extract(self.tensors.posterior_variance, t, sample.shape)
        noise = torch.randn_like(sample)
        return mean + torch.sqrt(posterior_var.clamp(min=1e-20)) * noise
