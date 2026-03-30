"""VAE helper utilities for Goldman AI."""

from __future__ import annotations

from typing import Optional

DEFAULT_VAE = "madebyollin/sdxl-vae-fp16-fix"


def load_vae(*, vae_id_or_path: Optional[str], torch_dtype):
    """Load a custom/default VAE if requested. Returns None when disabled."""
    if vae_id_or_path is not None and vae_id_or_path.strip().lower() in {"none", "off", "disable"}:
        return None

    chosen_vae = vae_id_or_path or DEFAULT_VAE

    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(chosen_vae, torch_dtype=torch_dtype)
