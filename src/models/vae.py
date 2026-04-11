"""Variational Autoencoder used to map images to latent space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Simple Conv -> GroupNorm -> SiLU block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """Convolutional encoder that outputs latent mean/log-variance maps."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ConvBlock(base_channels * 2, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4, stride=2),
            ConvBlock(base_channels * 4, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 8, stride=2),
            ConvBlock(base_channels * 8, base_channels * 8),
        )
        self.to_mu = nn.Conv2d(base_channels * 8, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(base_channels * 8, latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.down(x)
        return self.to_mu(h), self.to_logvar(h)


class Decoder(nn.Module):
    """Convolutional decoder from latent space back to pixel space."""

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        self.in_proj = ConvBlock(latent_channels, base_channels * 8)
        self.up = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 8),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(base_channels * 8, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(base_channels * 4, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(base_channels * 2, base_channels),
            ConvBlock(base_channels, base_channels),
        )
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z)
        h = self.up(h)
        return self.out(h)


@dataclass
class VAELoss:
    """Container for VAE loss terms."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


class VAE(nn.Module):
    """VAE with helper methods for latent diffusion pretraining."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 4,
        latent_scale: float = 0.18215,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, latent_channels)
        self.decoder = Decoder(in_channels, base_channels, latent_channels)
        self.latent_scale = latent_scale

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return z * self.latent_scale, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z / self.latent_scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x, sample=True)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1e-6,
    ) -> VAELoss:
        rec_loss = F.mse_loss(recon, x)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl = kl.mean()
        total = rec_loss + kl_weight * kl
        return VAELoss(total=total, reconstruction=rec_loss, kl=kl)
