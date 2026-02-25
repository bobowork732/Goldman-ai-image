"""Helpers for writing and exposing generated output artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Minimal valid 1x1 transparent PNG.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def ensure_output_image(output_path: str) -> str:
    """Create a placeholder PNG image on disk and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG_BYTES)
    return str(path)


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
