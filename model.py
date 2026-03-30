"""Model loading and image generation helpers for Goldman AI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "SG161222/RealVisXL_V5.0"


def generate_with_model(
    *,
    prompt: str,
    negative_prompt: str,
    model_id_or_path: str,
    vae_id_or_path: Optional[str],
    steps: int,
    guidance: float,
    height: int,
    width: int,
    seed: Optional[int],
    output_dir: str,
) -> Path:
    """Generate an image and return the output path."""
    import torch
    from diffusers import AutoPipelineForText2Image

    from vae import load_vae

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae = load_vae(vae_id_or_path=vae_id_or_path, torch_dtype=dtype)

    print(f"Loading model '{model_id_or_path}' on {device}...")
    if vae is not None:
        print(f"Using VAE '{vae_id_or_path}'")

    pipeline_kwargs = {"torch_dtype": dtype}
    if vae is not None:
        pipeline_kwargs["vae"] = vae

    pipe = AutoPipelineForText2Image.from_pretrained(model_id_or_path, **pipeline_kwargs)
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    realistic_prompt = (
        "ultra realistic, RAW photo, high detail skin texture, professional photography, natural lighting, "
        f"{prompt}"
    )

    result = pipe(
        prompt=realistic_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=generator,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_path = out / f"goldman-ai-{timestamp}.png"
    result.images[0].save(output_path)
    return output_path
