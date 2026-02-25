"""Image-to-image pipeline orchestration."""

from __future__ import annotations

from goldman_ai.models import GoldmanAIModel
from goldman_ai.pipelines.output_store import ensure_output_image, output_url_from_path
from goldman_ai.pipelines.processing import run_goldman_sampling


def run_image_to_image(
    model: GoldmanAIModel,
    prompt: str,
    image: str,
    strength: float = 0.8,
    samples: int | None = None,
    seed: int | None = None,
    sampler_iterations: int | None = None,
    processing_params: dict[str, float] | None = None,
) -> dict:
    """Run image-to-image generation and return output placeholders."""
    settings = model.get_generation_kwargs(
        prompt=prompt,
        image=image,
        strength=strength,
        sample_count=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=processing_params,
    )
    output_images = []
    sampler_debug = []
    for idx in range(settings["sample_count"]):
        output_path = f"{settings['output_dir']}/image_to_image_{idx + 1}.png"
        ensure_output_image(output_path)
        sampled = run_goldman_sampling(
            output_path,
            iterations=settings["sampler_iterations"],
            params=processing_params,
        )
        output_images.append(output_url_from_path(sampled["final_image"]))
        sampler_debug.append(sampled)
    return {
        "task": "image_to_image",
        "input_image": image,
        "prompt": prompt,
        "parameters": settings,
        "output_images": output_images,
        "sampler_debug": sampler_debug,
    }
