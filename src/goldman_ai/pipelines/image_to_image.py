"""Image-to-image pipeline orchestration."""

from __future__ import annotations

import hashlib
import time

from goldman_ai.models import GoldmanAIModel
from goldman_ai.pipelines.output_store import load_input_image, output_url_from_path, persist_generated_image
from goldman_ai.pipelines.processing import run_goldman_sampling


def _name_for(prompt: str, image_ref: str, idx: int) -> str:
    digest = hashlib.sha1(f"{prompt}|{image_ref}".encode("utf-8")).hexdigest()[:10]
    return f"image_to_image_{digest}_{idx}.png"


def run_image_to_image(
    model: GoldmanAIModel,
    prompt: str,
    image: str,
    strength: float = 0.8,
    samples: int | None = None,
    seed: int | None = None,
    sampler_iterations: int | None = None,
    processing_params: dict[str, float] | None = None,
    include_sampler_debug: bool = True,
) -> dict:
    """Run image-to-image generation and return output payload."""
    start = time.perf_counter()
    settings = model.get_generation_kwargs(
        prompt=prompt,
        image=image,
        strength=strength,
        sample_count=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=processing_params,
    )

    t0 = time.perf_counter()
    model.load()
    t1 = time.perf_counter()

    input_image = load_input_image(image, output_dir=settings["output_dir"])
    t2 = time.perf_counter()

    generated_images = model.generate_image_to_image(
        prompt=prompt,
        image=input_image,
        strength=strength,
        sample_count=settings["sample_count"],
    )
    t3 = time.perf_counter()

    output_images: list[str] = []
    sampler_debug: list[dict] = []
    for idx, generated in enumerate(generated_images, start=1):
        output_path = f"{settings['output_dir']}/{_name_for(prompt, image, idx)}"
        persisted = persist_generated_image(generated, output_path)
        sampled = run_goldman_sampling(persisted, iterations=settings["sampler_iterations"], params=processing_params)
        output_images.append(output_url_from_path(sampled["final_image"]))
        if include_sampler_debug:
            sampler_debug.append(sampled)

    t4 = time.perf_counter()
    run_info = {
        "seed": settings.get("seed"),
        "sampler_iterations": settings.get("sampler_iterations"),
        "model_identifier": model.model_identifier,
        "backend": model.backend,
        "elapsed_ms": round((t4 - start) * 1000, 2),
        "stage_timings_ms": {
            "load": round((t1 - t0) * 1000, 2),
            "preprocess": round((t2 - t1) * 1000, 2),
            "infer": round((t3 - t2) * 1000, 2),
            "postprocess": 0.0,
            "save": round((t4 - t3) * 1000, 2),
        },
    }

    return {
        "task": "image_to_image",
        "input_image": image,
        "prompt": prompt,
        "parameters": settings,
        "run_info": run_info,
        "output_images": output_images,
        "sampler_debug": sampler_debug if include_sampler_debug else None,
    }
