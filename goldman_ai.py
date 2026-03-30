#!/usr/bin/env python3
"""Goldman AI: a lightweight prompt-to-image generator in pure Python.

No external dependencies are required.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldmanAIConfig:
    width: int = 1024
    height: int = 1024
    seed: int | None = None


def _stable_seed(prompt: str, seed: int | None = None) -> int:
    if seed is not None:
        return seed
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _palette_from_prompt(prompt: str, rng: random.Random) -> list[tuple[int, int, int]]:
    digest = hashlib.md5(prompt.encode("utf-8")).digest()
    colors: list[tuple[int, int, int]] = []
    for i in range(0, 15, 3):
        base = digest[i]
        r = (base + rng.randint(0, 120)) % 256
        g = (digest[(i + 1) % len(digest)] + rng.randint(0, 120)) % 256
        b = (digest[(i + 2) % len(digest)] + rng.randint(0, 120)) % 256
        colors.append((r, g, b))
    return colors


def _blend(base: tuple[int, int, int], top: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    return (
        int(base[0] * (1 - alpha) + top[0] * alpha),
        int(base[1] * (1 - alpha) + top[1] * alpha),
        int(base[2] * (1 - alpha) + top[2] * alpha),
    )


def _draw_circle(canvas: list[list[tuple[int, int, int]]], cx: int, cy: int, radius: int, color: tuple[int, int, int], alpha: float) -> None:
    h = len(canvas)
    w = len(canvas[0])
    x0 = max(0, cx - radius)
    x1 = min(w - 1, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h - 1, cy + radius)
    r2 = radius * radius
    for y in range(y0, y1 + 1):
        dy = y - cy
        for x in range(x0, x1 + 1):
            dx = x - cx
            if dx * dx + dy * dy <= r2:
                canvas[y][x] = _blend(canvas[y][x], color, alpha)


def _write_bmp(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    row_padding = (4 - (width * 3) % 4) % 4
    pixel_data_size = (width * 3 + row_padding) * height
    file_size = 54 + pixel_data_size

    with path.open("wb") as f:
        # BMP Header
        f.write(b"BM")
        f.write(struct.pack("<I", file_size))
        f.write(b"\x00\x00\x00\x00")
        f.write(struct.pack("<I", 54))

        # DIB Header (BITMAPINFOHEADER)
        f.write(struct.pack("<I", 40))
        f.write(struct.pack("<i", width))
        f.write(struct.pack("<i", height))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 24))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", pixel_data_size))
        f.write(struct.pack("<i", 2835))
        f.write(struct.pack("<i", 2835))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))

        # Pixel data (BGR, bottom-up)
        for row in reversed(pixels):
            for r, g, b in row:
                f.write(bytes((b, g, r)))
            if row_padding:
                f.write(b"\x00" * row_padding)


def generate_image(prompt: str, output_path: str | Path, config: GoldmanAIConfig | None = None) -> Path:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    cfg = config or GoldmanAIConfig()
    seed = _stable_seed(prompt, cfg.seed)
    rng = random.Random(seed)
    palette = _palette_from_prompt(prompt, rng)

    # Initialize gradient background
    pixels: list[list[tuple[int, int, int]]] = []
    for y in range(cfg.height):
        t = y / max(1, cfg.height - 1)
        c1 = palette[int(t * (len(palette) - 1))]
        c2 = palette[min(len(palette) - 1, int(t * (len(palette) - 1)) + 1)]
        local_t = (t * (len(palette) - 1)) % 1
        row_color = _blend(c1, c2, local_t)
        pixels.append([row_color for _ in range(cfg.width)])

    # Add circular elements
    for _ in range(80):
        cx = rng.randint(0, cfg.width - 1)
        cy = rng.randint(0, cfg.height - 1)
        radius = rng.randint(max(4, cfg.width // 50), max(8, cfg.width // 8))
        color = rng.choice(palette)
        alpha = rng.uniform(0.15, 0.5)
        _draw_circle(pixels, cx, cy, radius, color, alpha)

    # Prompt-reactive wave overlay
    wave_amp = (sum(ord(c) for c in prompt) % 80) + 30
    wave_freq = (len(prompt) % 7) + 2
    wave_color = rng.choice(palette)
    for x in range(cfg.width):
        y = cfg.height // 2 + int(math.sin(x / cfg.width * math.pi * wave_freq) * wave_amp)
        if 0 <= y < cfg.height:
            for yy in range(y, min(cfg.height, y + 8)):
                pixels[yy][x] = _blend(pixels[yy][x], wave_color, 0.35)

    out = Path(output_path)
    if out.suffix.lower() != ".bmp":
        out = out.with_suffix(".bmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_bmp(out, pixels)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Goldman AI image generator")
    parser.add_argument("prompt", help="Text prompt for the generated artwork")
    parser.add_argument("-o", "--output", default="outputs/goldman_ai.bmp", help="Output BMP path")
    parser.add_argument("--width", type=int, default=768, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=768, help="Image height in pixels")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible variation")
    args = parser.parse_args()

    cfg = GoldmanAIConfig(width=args.width, height=args.height, seed=args.seed)
    output = generate_image(args.prompt, args.output, cfg)
    print(f"Generated with Goldman AI: {output}")


if __name__ == "__main__":
    main()
