"""Text-to-image pipeline orchestration."""

from __future__ import annotations

from goldman_ai.models import GoldmanAIModel
from goldman_ai.pipelines.output_store import output_url_from_path, persist_generated_image
from goldman_ai.pipelines.processing import run_goldman_sampling


def run_text_to_image(
    model: GoldmanAIModel,
    prompt: str,
    samples: int | None = None,
    seed: int | None = None,
    size: str | None = None,
    sampler_iterations: int | None = None,
    processing_params: dict[str, float] | None = None,
) -> dict:
    """Run text-to-image generation and return output placeholders."""
    settings = model.get_generation_kwargs(
        prompt=prompt,
        sample_count=samples,
        seed=seed,
        size=size,
        sampler_iterations=sampler_iterations,
        processing_params=processing_params,
    )
    generated_images = model.generate_text_to_image(prompt=prompt, sample_count=settings["sample_count"])

    output_images = []
    sampler_debug = []
    for idx, generated in enumerate(generated_images, start=1):
        output_path = f"{settings['output_dir']}/text_to_image_{idx}.png"
        persisted = persist_generated_image(generated, output_path)
        sampled = run_goldman_sampling(
            persisted,
            iterations=settings["sampler_iterations"],
            params=processing_params,
        )
        output_images.append(output_url_from_path(sampled["final_image"]))
        sampler_debug.append(sampled)
    return {
        "task": "text_to_image",
        "prompt": prompt,
        "parameters": settings,
        "output_images": output_images,
        "sampler_debug": sampler_debug,
    }
