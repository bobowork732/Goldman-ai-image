# Goldman AI Image Generator

`Goldman AI` is a lightweight Python image generator that creates abstract art from a text prompt.

## Setup

No third-party dependencies are required.

## Usage

```bash
python3 goldman_ai.py "futuristic skyline at dusk" -o outputs/skyline.bmp
```

Optional arguments:

- `--width` image width (default `768`)
- `--height` image height (default `768`)
- `--seed` deterministic seed for controlled variation

## Example

```bash
python3 goldman_ai.py "liquid chrome lotus, cinematic lighting" --seed 42
```

This writes a BMP file to `outputs/goldman_ai.bmp` by default.
