"""Text-to-image CLI for latent diffusion checkpoints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from PIL import Image, PngImagePlugin

from src.infer.pipeline import GenerationConfig, LatentDiffusionPipeline
from src.safety.filter import BasicSafetyFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from prompts")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt")
    parser.add_argument("--prompt-file", type=str, default=None, help="Optional text file with one prompt per line")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim", "euler"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="outputs/txt2img")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--text-model-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size used for prompt-list generation")
    parser.add_argument("--safety-filter", action="store_true", help="Enable basic safety filtering")
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompt_file:
        file_prompts = [line.strip() for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()]
        prompts.extend([p for p in file_prompts if p])
    if not prompts:
        raise ValueError("Provide --prompt and/or --prompt-file.")
    return prompts


def save_image_with_metadata(image: Image.Image, output_path: Path, metadata: dict) -> None:
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("generation", json.dumps(metadata))
    image.save(output_path, format="PNG", pnginfo=pnginfo)

    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = LatentDiffusionPipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        text_model_name=args.text_model_name,
    )

    cfg = GenerationConfig(
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        sampler=args.sampler,
        eta=args.eta,
    )

    safety_filter = BasicSafetyFilter(enabled=args.safety_filter)

    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        batch_negative = [args.negative_prompt] * len(batch_prompts)
        batch_seed = args.seed + start

        images = pipeline.generate(
            prompts=batch_prompts,
            negative_prompts=batch_negative,
            seed=batch_seed,
            cfg=cfg,
            safety_filter=safety_filter,
        )

        for i, (img_tensor, prompt) in enumerate(zip(images, batch_prompts)):
            img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype("uint8")
            pil = Image.fromarray(img)

            image_index = start + i
            out_path = output_dir / f"sample_{image_index:05d}.png"
            metadata = {
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
                "seed": batch_seed + i,
                "guidance_scale": args.guidance_scale,
                "steps": args.steps,
                "height": args.height,
                "width": args.width,
                "sampler": args.sampler,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            save_image_with_metadata(pil, out_path, metadata)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
