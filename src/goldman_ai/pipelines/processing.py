"""Shared image post-processing primitives and Goldman sampler."""

from __future__ import annotations

from goldman_ai.config import DEFAULT_SAMPLER_ITERATIONS


def apply_blur(image: str, amount: float) -> dict[str, str | float]:
    """Return metadata for a blur operation placeholder."""
    return {"operation": "blur", "image": image, "amount": amount}


def apply_noise(image: str, sigma: float) -> dict[str, str | float]:
    """Return metadata for a noise operation placeholder."""
    return {"operation": "noise", "image": image, "sigma": sigma}


def apply_sharpen(image: str, amount: float) -> dict[str, str | float]:
    """Return metadata for a sharpen operation placeholder."""
    return {"operation": "sharpen", "image": image, "amount": amount}


def goldman_iteration(image: str, params: dict[str, float] | None = None) -> dict[str, object]:
    """Run a single blur -> noise -> sharpen sequence and return trace metadata."""
    p = params or {}
    blur_op = apply_blur(image, p.get("blur", 0.2))
    noise_op = apply_noise(image, p.get("noise", 0.05))
    sharpen_op = apply_sharpen(image, p.get("sharpen", 0.2))
    return {
        "input_image": image,
        "operations": [blur_op, noise_op, sharpen_op],
    }


def run_goldman_sampling(
    image: str,
    iterations: int = DEFAULT_SAMPLER_ITERATIONS,
    params: dict[str, float] | None = None,
) -> dict[str, object]:
    """Repeat Goldman iteration and return final image plus sampler trace."""
    applied_iterations = max(iterations, 1)
    trace = [goldman_iteration(image, params) for _ in range(applied_iterations)]
    return {
        "final_image": image,
        "iterations": applied_iterations,
        "trace": trace,
    }
