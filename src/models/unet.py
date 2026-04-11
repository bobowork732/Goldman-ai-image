"""UNet noise predictor with timestep/text conditioning for latent diffusion."""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        g1 = min(8, in_channels)
        while in_channels % g1 != 0:
            g1 -= 1
        g2 = min(8, out_channels)
        while out_channels % g2 != 0:
            g2 -= 1
        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.res = ResBlock(in_channels, out_channels, cond_dim)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res(x, cond)
        return self.down(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock(out_channels + skip_channels, out_channels, cond_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        return self.res(h, cond)


class UNetDenoiser(nn.Module):
    """Compact UNet suitable for latent diffusion epsilon prediction."""

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mults: Optional[List[int]] = None,
        text_embed_dim: int = 768,
        time_embed_dim: int = 512,
    ) -> None:
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 4]

        self.in_conv = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.cond_proj = nn.Linear(text_embed_dim, time_embed_dim)

        dims = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        in_dim = base_channels
        for d in dims:
            self.down_blocks.append(DownBlock(in_dim, d, time_embed_dim))
            in_dim = d

        self.mid = ResBlock(dims[-1], dims[-1], time_embed_dim)

        self.up_blocks = nn.ModuleList()
        rev = list(reversed(dims))
        in_dim = rev[0]
        for i, skip_dim in enumerate(rev):
            out_dim = rev[i + 1] if i + 1 < len(rev) else base_channels
            self.up_blocks.append(UpBlock(in_dim, skip_dim, out_dim, time_embed_dim))
            in_dim = out_dim

        g = min(8, base_channels)
        while base_channels % g != 0:
            g -= 1
        self.out = nn.Sequential(
            nn.GroupNorm(g, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1),
        )
        self.time_embed_dim = time_embed_dim

    def build_condition(self, timesteps: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(timesteps, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        pooled_text = text_embeddings.mean(dim=1)
        return t_emb + self.cond_proj(pooled_text)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.build_condition(timesteps, text_embeddings)
        h = self.in_conv(latents)
        skips = []
        for block in self.down_blocks:
            h, skip = block(h, cond)
            skips.append(skip)

        h = self.mid(h, cond)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = block(h, skip, cond)

        return self.out(h)
