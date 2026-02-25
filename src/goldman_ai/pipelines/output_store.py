"""Helpers for writing and exposing generated output artifacts."""

from __future__ import annotations

from pathlib import Path

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


def output_url_from_path(output_path: str) -> str:
    """Map a generated output path to the public API static URL."""
    filename = Path(output_path).name
    return f"/outputs/{filename}"
