"""FastAPI app exposing inference endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from goldman_ai.models import GoldmanAIModel
from goldman_ai.pipelines import add_object, remove_object, restyle_image, run_image_to_image, run_text_to_image

app = FastAPI(title="Goldman AI Inference API", version="0.3.0")
model = GoldmanAIModel()
model.load()


class ProcessingParams(BaseModel):
    blur: float = Field(default=0.2, ge=0.0, le=10.0)
    noise: float = Field(default=0.05, ge=0.0, le=10.0)
    sharpen: float = Field(default=0.2, ge=0.0, le=10.0)


class GenerationResponse(BaseModel):
    task: str
    input_image: str | None = None
    prompt: str | None = None
    mask_or_box: str | None = None
    parameters: dict
    output_images: list[str]
    sampler_debug: list[dict] | None = None


class TextToImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    samples: int | None = Field(default=None, ge=1)
    sampler_iterations: int | None = Field(default=None, ge=1)
    seed: int | None = None
    size: str | None = Field(default=None, description="Output size, e.g. 512x512")
    processing_params: ProcessingParams | None = None


class ImageToImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    image: str = Field(..., description="Path/URI to source image")
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    samples: int | None = Field(default=None, ge=1)
    sampler_iterations: int | None = Field(default=None, ge=1)
    seed: int | None = None
    processing_params: ProcessingParams | None = None


class AddObjectRequest(BaseModel):
    image: str = Field(..., description="Path/URI to source image")
    object_prompt: str = Field(..., min_length=1)
    mask_or_box: str = Field(..., description="Mask path/URI or box coordinates")
    samples: int | None = Field(default=None, ge=1)
    sampler_iterations: int | None = Field(default=None, ge=1)
    seed: int | None = None
    processing_params: ProcessingParams | None = None


class RemoveObjectRequest(BaseModel):
    image: str = Field(..., description="Path/URI to source image")
    mask_or_box: str = Field(..., description="Mask path/URI or box coordinates")
    samples: int | None = Field(default=None, ge=1)
    sampler_iterations: int | None = Field(default=None, ge=1)
    seed: int | None = None
    processing_params: ProcessingParams | None = None


class RestyleRequest(BaseModel):
    image: str = Field(..., description="Path/URI to source image")
    style_prompt: str = Field(..., min_length=1)
    samples: int | None = Field(default=None, ge=1)
    sampler_iterations: int | None = Field(default=None, ge=1)
    seed: int | None = None
    processing_params: ProcessingParams | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate/text-to-image", response_model=GenerationResponse)
def generate_text_to_image(req: TextToImageRequest) -> dict:
    return run_text_to_image(
        model,
        req.prompt,
        samples=req.samples,
        seed=req.seed,
        size=req.size,
        sampler_iterations=req.sampler_iterations,
        processing_params=req.processing_params.model_dump() if req.processing_params else None,
    )


@app.post("/generate/image-to-image", response_model=GenerationResponse)
def generate_image_to_image(req: ImageToImageRequest) -> dict:
    return run_image_to_image(
        model,
        req.prompt,
        req.image,
        strength=req.strength,
        samples=req.samples,
        seed=req.seed,
        sampler_iterations=req.sampler_iterations,
        processing_params=req.processing_params.model_dump() if req.processing_params else None,
    )


@app.post("/edit/add-object", response_model=GenerationResponse)
def edit_add_object(req: AddObjectRequest) -> dict:
    return add_object(
        model,
        req.image,
        req.object_prompt,
        req.mask_or_box,
        samples=req.samples,
        seed=req.seed,
        sampler_iterations=req.sampler_iterations,
        processing_params=req.processing_params.model_dump() if req.processing_params else None,
    )


@app.post("/edit/remove-object", response_model=GenerationResponse)
def edit_remove_object(req: RemoveObjectRequest) -> dict:
    return remove_object(
        model,
        req.image,
        req.mask_or_box,
        samples=req.samples,
        seed=req.seed,
        sampler_iterations=req.sampler_iterations,
        processing_params=req.processing_params.model_dump() if req.processing_params else None,
    )


@app.post("/edit/restyle", response_model=GenerationResponse)
def edit_restyle(req: RestyleRequest) -> dict:
    return restyle_image(
        model,
        req.image,
        req.style_prompt,
        samples=req.samples,
        seed=req.seed,
        sampler_iterations=req.sampler_iterations,
        processing_params=req.processing_params.model_dump() if req.processing_params else None,
    )
