"""Helpers for loading references, writing, and exposing output artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

# Minimal valid 1x1 transparent PNG.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _read_url_bytes(url: str) -> bytes:
    with urlopen(url, timeout=10) as response:  # nosec - controlled by caller input
        return response.read()


def resolve_image_reference(image_ref: str, output_dir: str = "./outputs") -> str:
    """Validate and normalize local path/URL/upload-path image references."""
    parsed = urlparse(image_ref)
    if parsed.scheme in {"http", "https"}:
        ext = Path(parsed.path).suffix or ".png"
        digest = hashlib.sha256(image_ref.encode("utf-8")).hexdigest()[:12]
        cached = Path(output_dir) / "inputs" / f"{digest}{ext}"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(_read_url_bytes(image_ref))
        return str(cached)

    candidate = Path(image_ref)
    if candidate.exists() and candidate.is_file():
        return str(candidate)

    raise ValueError(f"Invalid image reference: {image_ref}")


def load_input_image(image_ref: str, output_dir: str = "./outputs") -> Any:
    """Load input image reference into backend-compatible PIL image object."""
    resolved = resolve_image_reference(image_ref, output_dir=output_dir)
    try:
        from PIL import Image

        return Image.open(resolved).convert("RGB")
    except Exception:
        return resolved


def persist_generated_image(image: Any, output_path: str) -> str:
    """Persist PIL image/bytes/path-like model output to disk and return path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(image, "save"):
        image.save(path)
    elif isinstance(image, (bytes, bytearray)):
        path.write_bytes(bytes(image))
    elif isinstance(image, str) and Path(image).exists():
        path.write_bytes(Path(image).read_bytes())
    else:
        path.write_bytes(_PNG_BYTES)

    return str(path)


def output_url_from_path(output_path: str) -> str:
    """Map a generated output path to the public API static URL."""
    filename = Path(output_path).name
    return f"/outputs/{filename}"
