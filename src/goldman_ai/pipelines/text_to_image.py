"""Text-to-image pipeline orchestration."""

from __future__ import annotations

import hashlib
import time

from goldman_ai.models import GoldmanAIModel
from goldman_ai.pipelines.output_store import output_url_from_path, persist_generated_image
from goldman_ai.pipelines.processing import run_goldman_sampling


def _name_for(prompt: str, idx: int) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return f"text_to_image_{digest}_{idx}.png"


def run_text_to_image(
    model: GoldmanAIModel,
    prompt: str,
    samples: int | None = None,
    seed: int | None = None,
    size: str | None = None,
    sampler_iterations: int | None = None,
    processing_params: dict[str, float] | None = None,
    include_sampler_debug: bool = True,
) -> dict:
    """Run text-to-image generation and return output payload."""
    start = time.perf_counter()
    settings = model.get_generation_kwargs(
        prompt=prompt,
        sample_count=samples,
        seed=seed,
        size=size,
        sampler_iterations=sampler_iterations,
        processing_params=processing_params,
    )

    t0 = time.perf_counter()
    model.load()
    t1 = time.perf_counter()

    generated_images = model.generate_text_to_image(prompt=prompt, sample_count=settings["sample_count"])
    t2 = time.perf_counter()

    output_images: list[str] = []
    sampler_debug: list[dict] = []
    for idx, generated in enumerate(generated_images, start=1):
        output_path = f"{settings['output_dir']}/{_name_for(prompt, idx)}"
        persisted = persist_generated_image(generated, output_path)
        sampled = run_goldman_sampling(persisted, iterations=settings["sampler_iterations"], params=processing_params)
        output_images.append(output_url_from_path(sampled["final_image"]))
        if include_sampler_debug:
            sampler_debug.append(sampled)

    t3 = time.perf_counter()
    run_info = {
        "seed": settings.get("seed"),
        "sampler_iterations": settings.get("sampler_iterations"),
        "model_identifier": model.model_identifier,
        "backend": model.backend,
        "elapsed_ms": round((t3 - start) * 1000, 2),
        "stage_timings_ms": {
            "load": round((t1 - t0) * 1000, 2),
            "preprocess": 0.0,
            "infer": round((t2 - t1) * 1000, 2),
            "postprocess": 0.0,
            "save": round((t3 - t2) * 1000, 2),
        },
    }

    return {
        "task": "text_to_image",
        "prompt": prompt,
        "parameters": settings,
        "run_info": run_info,
        "output_images": output_images,
        "sampler_debug": sampler_debug if include_sampler_debug else None,
    }
