"""Training entrypoint for VAE pretraining and latent diffusion."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from src.data.collate import FixedLengthTokenizer, build_collate_fn
from src.data.dataset import ImageCaptionDataset
from src.data.transforms import build_image_transforms
from src.diffusion.scheduler import DiffusionScheduler
from src.eval.metrics import ValidationMetrics
from src.infer.samplers import build_sampler
from src.models.text_encoder import TextEncoder
from src.models.unet import UNetDenoiser
from src.models.vae import VAE
from src.train.logging import build_logger


@dataclass
class EMA:
    model: nn.Module
    decay: float = 0.9999

    def __post_init__(self) -> None:
        self.shadow = copy.deepcopy(self.model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self) -> None:
        model_params = dict(self.model.named_parameters())
        for name, shadow_param in self.shadow.named_parameters():
            shadow_param.data.mul_(self.decay).add_(model_params[name].data, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow.load_state_dict(state_dict)


def save_checkpoint(
    ckpt_path: Path,
    vae: VAE,
    unet: UNetDenoiser,
    text_encoder: TextEncoder,
    ema: EMA,
    optimizers: Dict[str, torch.optim.Optimizer],
    step: int,
    metric_value: float | None = None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "metric_value": metric_value,
            "vae": vae.state_dict(),
            "unet": unet.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "ema_unet": ema.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        },
        ckpt_path,
    )


def load_checkpoint(
    ckpt_path: Path,
    vae: VAE,
    unet: UNetDenoiser,
    text_encoder: TextEncoder,
    ema: EMA,
    optimizers: Dict[str, torch.optim.Optimizer],
) -> int:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    vae.load_state_dict(checkpoint["vae"])
    unet.load_state_dict(checkpoint["unet"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    ema.load_state_dict(checkpoint["ema_unet"])
    for name, opt in optimizers.items():
        if name in checkpoint["optimizers"]:
            opt.load_state_dict(checkpoint["optimizers"][name])
    return int(checkpoint.get("step", 0))


def vae_pretrain_step(
    batch: Dict[str, torch.Tensor | List[str]],
    vae: VAE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float,
) -> Dict[str, float]:
    vae.train()
    images = batch["pixel_values"].to(device)

    recon, mu, logvar = vae(images)
    losses = vae.loss(images, recon, mu, logvar, kl_weight=kl_weight)

    optimizer.zero_grad(set_to_none=True)
    losses.total.backward()
    optimizer.step()

    return {
        "train/vae_total": losses.total.item(),
        "train/vae_reconstruction": losses.reconstruction.item(),
        "train/vae_kl": losses.kl.item(),
    }


def diffusion_train_step(
    batch: Dict[str, torch.Tensor | List[str]],
    vae: VAE,
    unet: UNetDenoiser,
    text_encoder: TextEncoder,
    scheduler: DiffusionScheduler,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    device: torch.device,
    conditioning_dropout: float,
) -> Dict[str, float]:
    unet.train()
    text_encoder.train()
    images = batch["pixel_values"].to(device)

    with torch.no_grad():
        latents, _, _ = vae.encode(images, sample=True)

    input_ids = batch["input_ids"].to(device)
    if conditioning_dropout > 0:
        drop_mask = (torch.rand(input_ids.shape[0], device=device) < conditioning_dropout).unsqueeze(1)
        input_ids = torch.where(drop_mask, torch.zeros_like(input_ids), input_ids)

    text_emb = text_encoder.encode_tokens(input_ids)

    timesteps = torch.randint(0, scheduler.num_train_steps, (latents.shape[0],), device=device, dtype=torch.long)
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    noise_pred = unet(noisy_latents, timesteps, text_emb)
    loss = F.mse_loss(noise_pred, noise)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    ema.update()

    return {"train/epsilon_loss": loss.item()}


@torch.no_grad()
def generate_preview_samples(
    vae: VAE,
    ema_unet: nn.Module,
    text_encoder: TextEncoder,
    scheduler: DiffusionScheduler,
    prompts: List[str],
    out_dir: Path,
    step: int,
    device: torch.device,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = build_sampler("ddim", scheduler, steps=steps, eta=0.0)
    bsz = len(prompts)
    latents = torch.randn(bsz, 4, height // 8, width // 8, device=device)

    cond_ids = text_encoder.tokenize_to_ids(prompts).to(device)
    uncond_ids = text_encoder.tokenize_to_ids([""] * bsz).to(device)
    cond_emb = text_encoder.encode_tokens(cond_ids)
    uncond_emb = text_encoder.encode_tokens(uncond_ids)

    for i, t in enumerate(sampler.timesteps):
        t_prev = sampler.timesteps[i + 1] if i + 1 < len(sampler.timesteps) else 0
        t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)
        eps_u = ema_unet(latents, t_batch, uncond_emb)
        eps_c = ema_unet(latents, t_batch, cond_emb)
        eps = eps_u + guidance_scale * (eps_c - eps_u)
        latents = sampler.step(eps, latents, t, t_prev, torch.Generator(device=device))

    images = ((vae.decode(latents).clamp(-1, 1) + 1) / 2).clamp(0, 1)

    grid = images[0].permute(1, 2, 0).cpu().numpy()
    img = (grid * 255).round().astype("uint8")
    out_path = out_dir / f"sample_step_{step}.png"
    Image.fromarray(img).save(out_path)
    return out_path


@torch.no_grad()
def evaluate_validation(
    loader: DataLoader,
    vae: VAE,
    ema_unet: nn.Module,
    text_encoder: TextEncoder,
    scheduler: DiffusionScheduler,
    metrics: ValidationMetrics,
    device: torch.device,
    max_batches: int = 4,
) -> Dict[str, float]:
    vae.eval()
    ema_unet.eval()
    text_encoder.eval()

    val_eps = []
    val_fid = []
    val_clip = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        images = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        latents, _, _ = vae.encode(images, sample=False)
        text_emb = text_encoder.encode_tokens(input_ids)

        timesteps = torch.randint(0, scheduler.num_train_steps, (latents.shape[0],), device=device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy = scheduler.add_noise(latents, noise, timesteps)
        pred = ema_unet(noisy, timesteps, text_emb)
        val_eps.append(F.mse_loss(pred, noise).item())

        rec = ((vae.decode(latents).clamp(-1, 1) + 1) / 2).clamp(0, 1)
        m = metrics.evaluate_batch((images + 1) / 2, rec, text_emb)
        if m.fid_proxy is not None:
            val_fid.append(m.fid_proxy)
        if m.clip_score_proxy is not None:
            val_clip.append(m.clip_score_proxy)

    out: Dict[str, float] = {"val/epsilon_loss": float(sum(val_eps) / max(len(val_eps), 1))}
    if val_fid:
        out["val/fid_proxy"] = float(sum(val_fid) / len(val_fid))
    if val_clip:
        out["val/clip_score_proxy"] = float(sum(val_clip) / len(val_clip))
    return out


def update_top_checkpoints(ranked: List[Tuple[float, Path]], metric: float, path: Path, keep_top_k: int) -> None:
    ranked.append((metric, path))
    ranked.sort(key=lambda x: x[0])
    while len(ranked) > keep_top_k:
        _, p = ranked.pop()
        if p.exists():
            p.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAE + latent diffusion")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_image_transforms(
        resolution=cfg["train"]["resolution"],
        center_crop=cfg["train"].get("center_crop", False),
        random_flip=cfg["train"].get("random_flip", True),
        color_jitter=cfg["train"].get("color_jitter", 0.0),
    )

    dataset = ImageCaptionDataset(
        cfg["dataset"]["path"],
        transform=transform,
        strict=cfg["dataset"].get("strict_checks", False),
        allow_empty_caption=cfg["dataset"].get("allow_empty_caption", False),
        source_type=cfg["dataset"].get("source_type", "folder"),
    )

    val_fraction = cfg["train"].get("val_split", 0.1)
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    vae = VAE(in_channels=3, base_channels=cfg["vae"]["base_channels"], latent_channels=cfg["vae"]["latent_channels"]).to(device)
    text_encoder = TextEncoder(
        model_name=cfg["text_encoder"].get("model_name"),
        max_length=cfg["text_encoder"].get("max_length", 77),
        embed_dim=cfg["text_encoder"].get("embed_dim", 768),
    ).to(device)

    tokenizer = FixedLengthTokenizer(text_encoder.tokenize_to_ids, max_length=text_encoder.max_length)
    collate = build_collate_fn(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"].get("num_workers", 4), drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"].get("num_workers", 2), drop_last=False, collate_fn=collate)

    unet = UNetDenoiser(
        latent_channels=cfg["vae"]["latent_channels"],
        base_channels=cfg["unet"]["base_channels"],
        channel_mults=cfg["unet"].get("channel_mults", [1, 2, 4]),
        text_embed_dim=text_encoder.embed_dim,
        time_embed_dim=cfg["unet"].get("time_embed_dim", 512),
    ).to(device)

    scheduler = DiffusionScheduler(
        num_train_steps=cfg["diffusion"]["num_train_steps"],
        beta_start=cfg["diffusion"].get("beta_start", 1e-4),
        beta_end=cfg["diffusion"].get("beta_end", 0.02),
        device=device,
    )

    vae_optim = torch.optim.AdamW(vae.parameters(), lr=cfg["train"]["vae_lr"])
    diffusion_optim = torch.optim.AdamW(list(unet.parameters()) + list(text_encoder.parameters()), lr=cfg["train"]["diffusion_lr"])
    ema = EMA(unet, decay=cfg["train"].get("ema_decay", 0.9999))

    logger = build_logger(
        cfg["train"].get("logger", "none"),
        log_dir=cfg["train"].get("log_dir", "runs/default"),
        project=cfg["train"].get("wandb_project", "latent-diffusion"),
        run_name=cfg["train"].get("run_name"),
        config=cfg,
    )

    metrics = ValidationMetrics(device=device)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), vae, unet, text_encoder, ema, {"vae": vae_optim, "diffusion": diffusion_optim})

    vae_steps = cfg["train"]["vae_pretrain_steps"]
    total_steps = cfg["train"]["total_steps"]
    save_every = cfg["train"]["checkpoint_interval"]
    eval_every = cfg["train"].get("eval_interval", 500)
    sample_every = cfg["train"].get("sample_interval", 500)
    keep_top_k = cfg["train"].get("keep_top_k", 3)
    global_step = start_step
    ranked_checkpoints: List[Tuple[float, Path]] = []

    while global_step < total_steps:
        for batch in train_loader:
            if global_step >= total_steps:
                break

            if global_step < vae_steps:
                train_metrics = vae_pretrain_step(batch, vae, vae_optim, device, kl_weight=cfg["train"].get("kl_weight", 1e-6))
            else:
                train_metrics = diffusion_train_step(
                    batch,
                    vae,
                    unet,
                    text_encoder,
                    scheduler,
                    diffusion_optim,
                    ema,
                    device,
                    conditioning_dropout=cfg["train"].get("cond_dropout", 0.1),
                )

            global_step += 1
            logger.log_metrics(train_metrics, global_step)

            if global_step % cfg["train"].get("log_interval", 50) == 0:
                print(f"step={global_step} train={train_metrics}")

            if global_step >= vae_steps and global_step % sample_every == 0:
                sample_path = generate_preview_samples(
                    vae=vae,
                    ema_unet=ema.shadow,
                    text_encoder=text_encoder,
                    scheduler=scheduler,
                    prompts=cfg["train"].get("sample_prompts", ["a cinematic mountain landscape"]),
                    out_dir=Path(cfg["train"]["checkpoint_dir"]) / "samples",
                    step=global_step,
                    device=device,
                    steps=cfg["train"].get("sample_steps", 20),
                    guidance_scale=cfg["train"].get("sample_guidance_scale", 7.5),
                    height=cfg["train"]["resolution"],
                    width=cfg["train"]["resolution"],
                )
                logger.log_text("samples/path", str(sample_path), global_step)

            if global_step >= vae_steps and global_step % eval_every == 0:
                val_metrics = evaluate_validation(val_loader, vae, ema.shadow, text_encoder, scheduler, metrics, device)
                logger.log_metrics(val_metrics, global_step)
                print(f"step={global_step} val={val_metrics}")

                score = val_metrics.get("val/fid_proxy", val_metrics.get("val/epsilon_loss", 1e9))
                best_ckpt = Path(cfg["train"]["checkpoint_dir"]) / f"ranked_step_{global_step}.pt"
                save_checkpoint(best_ckpt, vae, unet, text_encoder, ema, {"vae": vae_optim, "diffusion": diffusion_optim}, global_step, metric_value=score)
                update_top_checkpoints(ranked_checkpoints, score, best_ckpt, keep_top_k)

            if global_step % save_every == 0:
                save_checkpoint(
                    Path(cfg["train"]["checkpoint_dir"]) / f"step_{global_step}.pt",
                    vae,
                    unet,
                    text_encoder,
                    ema,
                    {"vae": vae_optim, "diffusion": diffusion_optim},
                    global_step,
                )

    save_checkpoint(
        Path(cfg["train"]["checkpoint_dir"]) / "final.pt",
        vae,
        unet,
        text_encoder,
        ema,
        {"vae": vae_optim, "diffusion": diffusion_optim},
        global_step,
    )
    logger.close()


if __name__ == "__main__":
    main()
