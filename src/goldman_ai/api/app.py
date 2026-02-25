"""FastAPI app exposing inference endpoints."""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from goldman_ai.models import GoldmanAIModel
from goldman_ai.pipelines import add_object, remove_object, restyle_image, run_image_to_image, run_text_to_image

app = FastAPI(title="Goldman AI Inference API", version="0.6.0")
model = GoldmanAIModel()
model.load()
logger = logging.getLogger("goldman_ai.api")

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_DIR = Path("./outputs/uploads")


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
    run_info: dict | None = None
    output_images: list[str]
    sampler_debug: list[dict] | None = None


class TextToImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    samples: int | None = Field(default=None, ge=1)
    sampler_iterations: int | None = Field(default=None, ge=1)
    seed: int | None = None
    size: str | None = Field(default=None, description="Output size, e.g. 512x512")
    processing_params: ProcessingParams | None = None
    include_sampler_debug: bool = False


class ImageToImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    image: str = Field(..., description="Path/URI to source image")
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    samples: int | None = Field(default=None, ge=1)
    sampler_iterations: int | None = Field(default=None, ge=1)
    seed: int | None = None
    processing_params: ProcessingParams | None = None
    include_sampler_debug: bool = False


class AddObjectRequest(BaseModel):
    image: str = Field(..., description="Path/URI to source image")
    object_prompt: str = Field(..., min_length=1)
    mask_or_box: str = Field(..., description="Mask path/URI or box coordinates")
    samples: int | None = Field(default=None, ge=1)
    sampler_iterations: int | None = Field(default=None, ge=1)
    seed: int | None = None
    processing_params: ProcessingParams | None = None
    include_sampler_debug: bool = False


class RemoveObjectRequest(BaseModel):
    image: str = Field(..., description="Path/URI to source image")
    mask_or_box: str = Field(..., description="Mask path/URI or box coordinates")
    samples: int | None = Field(default=None, ge=1)
    sampler_iterations: int | None = Field(default=None, ge=1)
    seed: int | None = None
    processing_params: ProcessingParams | None = None
    include_sampler_debug: bool = False


class RestyleRequest(BaseModel):
    image: str = Field(..., description="Path/URI to source image")
    style_prompt: str = Field(..., min_length=1)
    samples: int | None = Field(default=None, ge=1)
    sampler_iterations: int | None = Field(default=None, ge=1)
    seed: int | None = None
    processing_params: ProcessingParams | None = None
    include_sampler_debug: bool = False


app.mount("/outputs", StaticFiles(directory="./outputs"), name="outputs")


def _log_response(request_id: str, endpoint: str, response: dict) -> None:
    logger.info(
        "inference_response %s",
        json.dumps(
            {
                "request_id": request_id,
                "endpoint": endpoint,
                "task": response.get("task"),
                "output_images": response.get("output_images", []),
            }
        ),
    )


def _validate_local_ref(image_ref: str) -> str:
    if "\x00" in image_ref:
        raise HTTPException(status_code=400, detail="Invalid image reference")
    parsed = urlparse(image_ref)
    if parsed.scheme in {"http", "https"}:
        return image_ref
    ref_path = Path(image_ref)
    if any(part == ".." for part in ref_path.parts):
        raise HTTPException(status_code=400, detail="Path traversal is not allowed")
    return str(ref_path)


async def _save_upload(upload: UploadFile) -> str:
    content_type = (upload.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")

    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload too large (max {MAX_UPLOAD_BYTES} bytes)")

    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(upload.filename or "upload.png").name)
    if not safe_name:
        safe_name = "upload.png"

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    out_path.write_bytes(data)
    return str(out_path)


async def _normalize_image_input(image_ref: str | None, image_file: UploadFile | None, field_name: str) -> str:
    """Normalize image source with explicit precedence: upload file > JSON/path reference."""
    if image_file is not None:
        return await _save_upload(image_file)
    if image_ref:
        return _validate_local_ref(image_ref)
    raise HTTPException(status_code=422, detail=f"Provide `{field_name}` as upload file or image reference")


def _parse_processing_params_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        parsed = ProcessingParams.model_validate(json.loads(raw))
        return parsed.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid processing_params_json: {exc}") from exc


@app.get("/", response_class=HTMLResponse)
def web_app() -> str:
    return """
    <html><body><h1>Goldman AI Demo</h1></body></html>
    """


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/download/{filename}")
def download_result(filename: str) -> FileResponse:
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = Path("./outputs") / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(path=path, filename=path.name, media_type="application/octet-stream")


@app.post("/generate/text-to-image", response_model=GenerationResponse)
def generate_text_to_image(req: TextToImageRequest) -> dict:
    request_id = uuid.uuid4().hex
    response = run_text_to_image(
        model,
        req.prompt,
        samples=req.samples,
        seed=req.seed,
        size=req.size,
        sampler_iterations=req.sampler_iterations,
        processing_params=req.processing_params.model_dump() if req.processing_params else None,
        include_sampler_debug=req.include_sampler_debug,
    )
    _log_response(request_id, "/generate/text-to-image", response)
    return response


@app.post("/generate/image-to-image", response_model=GenerationResponse)
def generate_image_to_image(req: ImageToImageRequest) -> dict:
    request_id = uuid.uuid4().hex
    response = run_image_to_image(
        model,
        req.prompt,
        req.image,
        strength=req.strength,
        samples=req.samples,
        seed=req.seed,
        sampler_iterations=req.sampler_iterations,
        processing_params=req.processing_params.model_dump() if req.processing_params else None,
        include_sampler_debug=req.include_sampler_debug,
    )
    _log_response(request_id, "/generate/image-to-image", response)
    return response


@app.post("/generate/image-to-image/upload", response_model=GenerationResponse)
async def generate_image_to_image_upload(
    prompt: str = Form(...),
    image: str | None = Form(default=None),
    image_file: UploadFile | None = File(default=None),
    strength: float = Form(default=0.8),
    samples: int | None = Form(default=None),
    sampler_iterations: int | None = Form(default=None),
    seed: int | None = Form(default=None),
    include_sampler_debug: bool = Form(default=False),
    processing_params_json: str | None = Form(default=None),
) -> dict:
    request_id = uuid.uuid4().hex
    image_ref = await _normalize_image_input(image, image_file, field_name="image")
    response = run_image_to_image(
        model,
        prompt,
        image_ref,
        strength=strength,
        samples=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=_parse_processing_params_json(processing_params_json),
        include_sampler_debug=include_sampler_debug,
    )
    _log_response(request_id, "/generate/image-to-image/upload", response)
    return response


@app.post("/edit/add-object", response_model=GenerationResponse)
def edit_add_object(req: AddObjectRequest) -> dict:
    request_id = uuid.uuid4().hex
    response = add_object(
        model,
        req.image,
        req.object_prompt,
        req.mask_or_box,
        samples=req.samples,
        seed=req.seed,
        sampler_iterations=req.sampler_iterations,
        processing_params=req.processing_params.model_dump() if req.processing_params else None,
        include_sampler_debug=req.include_sampler_debug,
    )
    _log_response(request_id, "/edit/add-object", response)
    return response


@app.post("/edit/add-object/upload", response_model=GenerationResponse)
async def edit_add_object_upload(
    object_prompt: str = Form(...),
    image: str | None = Form(default=None),
    image_file: UploadFile | None = File(default=None),
    mask_or_box: str | None = Form(default=None),
    mask_file: UploadFile | None = File(default=None),
    samples: int | None = Form(default=None),
    sampler_iterations: int | None = Form(default=None),
    seed: int | None = Form(default=None),
    include_sampler_debug: bool = Form(default=False),
    processing_params_json: str | None = Form(default=None),
) -> dict:
    request_id = uuid.uuid4().hex
    image_ref = await _normalize_image_input(image, image_file, field_name="image")
    mask_ref = await _normalize_image_input(mask_or_box, mask_file, field_name="mask_or_box")
    response = add_object(
        model,
        image_ref,
        object_prompt,
        mask_ref,
        samples=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=_parse_processing_params_json(processing_params_json),
        include_sampler_debug=include_sampler_debug,
    )
    _log_response(request_id, "/edit/add-object/upload", response)
    return response


@app.post("/edit/remove-object", response_model=GenerationResponse)
def edit_remove_object(req: RemoveObjectRequest) -> dict:
    request_id = uuid.uuid4().hex
    response = remove_object(
        model,
        req.image,
        req.mask_or_box,
        samples=req.samples,
        seed=req.seed,
        sampler_iterations=req.sampler_iterations,
        processing_params=req.processing_params.model_dump() if req.processing_params else None,
        include_sampler_debug=req.include_sampler_debug,
    )
    _log_response(request_id, "/edit/remove-object", response)
    return response


@app.post("/edit/remove-object/upload", response_model=GenerationResponse)
async def edit_remove_object_upload(
    image: str | None = Form(default=None),
    image_file: UploadFile | None = File(default=None),
    mask_or_box: str | None = Form(default=None),
    mask_file: UploadFile | None = File(default=None),
    samples: int | None = Form(default=None),
    sampler_iterations: int | None = Form(default=None),
    seed: int | None = Form(default=None),
    include_sampler_debug: bool = Form(default=False),
    processing_params_json: str | None = Form(default=None),
) -> dict:
    request_id = uuid.uuid4().hex
    image_ref = await _normalize_image_input(image, image_file, field_name="image")
    mask_ref = await _normalize_image_input(mask_or_box, mask_file, field_name="mask_or_box")
    response = remove_object(
        model,
        image_ref,
        mask_ref,
        samples=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=_parse_processing_params_json(processing_params_json),
        include_sampler_debug=include_sampler_debug,
    )
    _log_response(request_id, "/edit/remove-object/upload", response)
    return response


@app.post("/edit/restyle", response_model=GenerationResponse)
def edit_restyle(req: RestyleRequest) -> dict:
    request_id = uuid.uuid4().hex
    response = restyle_image(
        model,
        req.image,
        req.style_prompt,
        samples=req.samples,
        seed=req.seed,
        sampler_iterations=req.sampler_iterations,
        processing_params=req.processing_params.model_dump() if req.processing_params else None,
        include_sampler_debug=req.include_sampler_debug,
    )
    _log_response(request_id, "/edit/restyle", response)
    return response


@app.post("/edit/restyle/upload", response_model=GenerationResponse)
async def edit_restyle_upload(
    style_prompt: str = Form(...),
    image: str | None = Form(default=None),
    image_file: UploadFile | None = File(default=None),
    samples: int | None = Form(default=None),
    sampler_iterations: int | None = Form(default=None),
    seed: int | None = Form(default=None),
    include_sampler_debug: bool = Form(default=False),
    processing_params_json: str | None = Form(default=None),
) -> dict:
    request_id = uuid.uuid4().hex
    image_ref = await _normalize_image_input(image, image_file, field_name="image")
    response = restyle_image(
        model,
        image_ref,
        style_prompt,
        samples=samples,
        seed=seed,
        sampler_iterations=sampler_iterations,
        processing_params=_parse_processing_params_json(processing_params_json),
        include_sampler_debug=include_sampler_debug,
    )
    _log_response(request_id, "/edit/restyle/upload", response)
    return response
