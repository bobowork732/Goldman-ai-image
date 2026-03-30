#!/usr/bin/env python3
"""Goldman AI - CLI entrypoint for realistic image generation."""

from __future__ import annotations

import argparse
import sys

from model import DEFAULT_MODEL, generate_with_model
from vae import DEFAULT_VAE

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goldman-ai",
        description="Generate realistic images with Goldman AI.",
    )
    parser.add_argument("prompt", help="Text prompt describing the image to generate.")
    parser.add_argument(
        "-n",
        "--negative-prompt",
        default="cartoon, anime, painting, blurry, low quality, distorted",
        help="Things to avoid in the generated image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs",
        help="Output directory for generated images (default: outputs).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face model id or local model path "
            f"(default: {DEFAULT_MODEL})."
        ),
    )
    parser.add_argument(
        "--vae",
        default=DEFAULT_VAE,
        help=(
            "VAE model id/local path for higher fidelity decoding. "
            "Use 'none' to disable custom VAE."
        ),
    )
    parser.add_argument("--steps", type=int, default=40, help="Inference steps (default: 40).")
    parser.add_argument("--guidance", type=float, default=6.0, help="Guidance scale (default: 6.0).")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels.")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible generation.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        output = generate_with_model(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            model_id_or_path=args.model,
            vae_id_or_path=args.vae,
            steps=args.steps,
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed,
            output_dir=args.output,
        )
    except KeyboardInterrupt:
        print("Generation cancelled.")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to generate image: {exc}", file=sys.stderr)
        return 1

    print(f"Saved image to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
