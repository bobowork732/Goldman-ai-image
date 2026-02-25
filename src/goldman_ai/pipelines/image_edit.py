"""Image edit pipeline orchestration."""

from __future__ import annotations

import hashlib
import time

from goldman_ai.models import GoldmanAIModel
from goldman_ai.pipelines.output_store import load_input_image, output_url_from_path, persist_generated_image
from goldman_ai.pipelines.processing import run_goldman_sampling


def _name_for(task: str, key: str, idx: int) -> str:
    digest = hashlib.sha1(f"{task}|{key}".encode("utf-8")).hexdigest()[:10]
    return f"{task}_{digest}_{idx}.png"


def _finalize(
    *,
    task: str,
    model: GoldmanAIModel,
    settings: dict,
    source_key: str,
    generated_images: list,
    processing_params: dict[str, float] | None,
    include_sampler_debug: bool,
    start: float,
    load_ms: float,
    preprocess_ms: float,
    infer_ms: float,
) -> tuple[list[str], list[dict] | None, dict]:
    output_images: list[str] = []
    sampler_debug: list[dict] = []
    save_start = time.perf_counter()
    for idx, generated in enumerate(generated_images, start=1):
        output_path = f"{settings['output_dir']}/{_name_for(task, source_key, idx)}"
        persisted = persist_generated_image(generated, output_path)
        sampled = run_goldman_sampling(persisted, iterations=settings["sampler_iterations"], params=processing_params)
        output_images.append(output_url_from_path(sampled["final_image"]))
        if include_sampler_debug:
            sampler_debug.append(sampled)
    save_end = time.perf_counter()
    run_info = {
        "seed": settings.get("seed"),
        "sampler_iterations": settings.get("sampler_iterations"),
        "model_identifier": model.model_identifier,
        "backend": model.backend,
        "elapsed_ms": round((save_end - start) * 1000, 2),
        "stage_timings_ms": {
            "load": round(load_ms, 2),
            "preprocess": round(preprocess_ms, 2),
            "infer": round(infer_ms, 2),
            "postprocess": 0.0,
            "save": round((save_end - save_start) * 1000, 2),
        },
    }
    return output_images, (sampler_debug if include_sampler_debug else None), run_info


def add_object(
    model: GoldmanAIModel,
    image: str,
    object_prompt: str,
    mask_or_box: str,
    samples: int | None = None,
    seed: int | None = None,
    sampler_iterations: int | None = None,
    processing_params: dict[str, float] | None = None,
    include_sampler_debug: bool = True,
) -> dict:
    """Add object(s) to an input image."""
    start = time.perf_counter()
    settings = model.get_generation_kwargs(
        image=image,
        object_prompt=object_prompt,
        mask_or_box=mask_or_box,
        sample_count=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=processing_params,
    )
    t0 = time.perf_counter(); model.load(); t1 = time.perf_counter()
    init_image = load_input_image(image, output_dir=settings["output_dir"])
    mask_image = load_input_image(mask_or_box, output_dir=settings["output_dir"])
    t2 = time.perf_counter()
    generated_images = model.edit_add_object(
        image=init_image,
        object_prompt=object_prompt,
        mask_or_box=mask_image,
        sample_count=settings["sample_count"],
    )
    t3 = time.perf_counter()

    output_images, sampler_debug, run_info = _finalize(
        task="add_object",
        model=model,
        settings=settings,
        source_key=object_prompt + image,
        generated_images=generated_images,
        processing_params=processing_params,
        include_sampler_debug=include_sampler_debug,
        start=start,
        load_ms=(t1 - t0) * 1000,
        preprocess_ms=(t2 - t1) * 1000,
        infer_ms=(t3 - t2) * 1000,
    )
    return {
        "task": "add_object",
        "input_image": image,
        "prompt": object_prompt,
        "mask_or_box": mask_or_box,
        "parameters": settings,
        "run_info": run_info,
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
    include_sampler_debug: bool = True,
) -> dict:
    """Remove object(s) from an input image."""
    start = time.perf_counter()
    settings = model.get_generation_kwargs(
        image=image,
        mask_or_box=mask_or_box,
        sample_count=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=processing_params,
    )
    t0 = time.perf_counter(); model.load(); t1 = time.perf_counter()
    init_image = load_input_image(image, output_dir=settings["output_dir"])
    mask_image = load_input_image(mask_or_box, output_dir=settings["output_dir"])
    t2 = time.perf_counter()
    generated_images = model.edit_remove_object(
        image=init_image,
        mask_or_box=mask_image,
        sample_count=settings["sample_count"],
    )
    t3 = time.perf_counter()

    output_images, sampler_debug, run_info = _finalize(
        task="remove_object",
        model=model,
        settings=settings,
        source_key=image + mask_or_box,
        generated_images=generated_images,
        processing_params=processing_params,
        include_sampler_debug=include_sampler_debug,
        start=start,
        load_ms=(t1 - t0) * 1000,
        preprocess_ms=(t2 - t1) * 1000,
        infer_ms=(t3 - t2) * 1000,
    )
    return {
        "task": "remove_object",
        "input_image": image,
        "mask_or_box": mask_or_box,
        "parameters": settings,
        "run_info": run_info,
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
    include_sampler_debug: bool = True,
) -> dict:
    """Restyle an input image with a text style prompt."""
    start = time.perf_counter()
    settings = model.get_generation_kwargs(
        image=image,
        style_prompt=style_prompt,
        sample_count=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=processing_params,
    )
    t0 = time.perf_counter(); model.load(); t1 = time.perf_counter()
    init_image = load_input_image(image, output_dir=settings["output_dir"])
    t2 = time.perf_counter()
    generated_images = model.edit_restyle(
        image=init_image,
        style_prompt=style_prompt,
        sample_count=settings["sample_count"],
    )
    t3 = time.perf_counter()

    output_images, sampler_debug, run_info = _finalize(
        task="restyle_image",
        model=model,
        settings=settings,
        source_key=style_prompt + image,
        generated_images=generated_images,
        processing_params=processing_params,
        include_sampler_debug=include_sampler_debug,
        start=start,
        load_ms=(t1 - t0) * 1000,
        preprocess_ms=(t2 - t1) * 1000,
        infer_ms=(t3 - t2) * 1000,
    )
    return {
        "task": "restyle_image",
        "input_image": image,
        "prompt": style_prompt,
        "parameters": settings,
        "run_info": run_info,
        "output_images": output_images,
        "sampler_debug": sampler_debug,
    }
