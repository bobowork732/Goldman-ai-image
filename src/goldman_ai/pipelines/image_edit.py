"""Image edit pipeline orchestration."""

from __future__ import annotations

from goldman_ai.models import GoldmanAIModel
from goldman_ai.pipelines.processing import run_goldman_sampling


def add_object(
    model: GoldmanAIModel,
    image: str,
    object_prompt: str,
    mask_or_box: str,
    samples: int | None = None,
    seed: int | None = None,
    sampler_iterations: int | None = None,
    processing_params: dict[str, float] | None = None,
) -> dict:
    """Add object(s) to an input image."""
    settings = model.get_generation_kwargs(
        image=image,
        object_prompt=object_prompt,
        mask_or_box=mask_or_box,
        sample_count=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=processing_params,
    )
    output_images = []
    sampler_debug = []
    for idx in range(settings["sample_count"]):
        output_path = f"{settings['output_dir']}/add_object_{idx + 1}.png"
        sampled = run_goldman_sampling(
            output_path,
            iterations=settings["sampler_iterations"],
            params=processing_params,
        )
        output_images.append(sampled["final_image"])
        sampler_debug.append(sampled)
    return {
        "task": "add_object",
        "input_image": image,
        "prompt": object_prompt,
        "mask_or_box": mask_or_box,
        "parameters": settings,
        "output_images": output_images,
        "sampler_debug": sampler_debug,
    }


def remove_object(
    model: GoldmanAIModel,
    image: str,
    mask_or_box: str,
    samples: int | None = None,
    seed: int | None = None,
    sampler_iterations: int | None = None,
    processing_params: dict[str, float] | None = None,
) -> dict:
    """Remove object(s) from an input image."""
    settings = model.get_generation_kwargs(
        image=image,
        mask_or_box=mask_or_box,
        sample_count=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=processing_params,
    )
    output_images = []
    sampler_debug = []
    for idx in range(settings["sample_count"]):
        output_path = f"{settings['output_dir']}/remove_object_{idx + 1}.png"
        sampled = run_goldman_sampling(
            output_path,
            iterations=settings["sampler_iterations"],
            params=processing_params,
        )
        output_images.append(sampled["final_image"])
        sampler_debug.append(sampled)
    return {
        "task": "remove_object",
        "input_image": image,
        "mask_or_box": mask_or_box,
        "parameters": settings,
        "output_images": output_images,
        "sampler_debug": sampler_debug,
    }


def restyle_image(
    model: GoldmanAIModel,
    image: str,
    style_prompt: str,
    samples: int | None = None,
    seed: int | None = None,
    sampler_iterations: int | None = None,
    processing_params: dict[str, float] | None = None,
) -> dict:
    """Restyle an input image with a text style prompt."""
    settings = model.get_generation_kwargs(
        image=image,
        style_prompt=style_prompt,
        sample_count=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=processing_params,
    )
    output_images = []
    sampler_debug = []
    for idx in range(settings["sample_count"]):
        output_path = f"{settings['output_dir']}/restyle_image_{idx + 1}.png"
        sampled = run_goldman_sampling(
            output_path,
            iterations=settings["sampler_iterations"],
            params=processing_params,
        )
        output_images.append(sampled["final_image"])
        sampler_debug.append(sampled)
    return {
        "task": "restyle_image",
        "input_image": image,
        "prompt": style_prompt,
        "parameters": settings,
        "output_images": output_images,
        "sampler_debug": sampler_debug,
    }
