# Goldman-ai-image

Local Python project scaffold for Goldman AI image inference workflows.

## Project layout

```text
src/goldman_ai/
├── api/
│   └── app.py
├── models/
│   └── goldman_model.py
├── pipelines/
│   ├── text_to_image.py
│   ├── image_to_image.py
│   ├── image_edit.py
│   ├── processing.py
│   └── output_store.py
└── config.py
```

## Setup

1. Create and activate a virtualenv:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install runtime dependencies:

   ```bash
   pip install fastapi uvicorn pydantic pillow
   # Optional real backend:
   pip install diffusers torch transformers accelerate
   ```

3. Expose the package from `src` during local development:

   ```bash
   export PYTHONPATH=src
   ```

## Defaults and tuning knobs

- `samples` controls **how many output images** are generated (default from `InferenceConfig.sample_count`, backed by `DEFAULT_SAMPLES = 20`).
- `sampler_iterations` controls **how many blur→noise→sharpen passes** the shared Goldman sampler runs per output (default from `InferenceConfig.sampler_iterations`, backed by `DEFAULT_SAMPLER_ITERATIONS = 20`).
- `processing_params` controls sampler operation strengths (`blur`, `noise`, `sharpen`).

These two knobs are independent, so you can request many outputs with few sampler iterations (or vice versa).

## Run local inference API + web app

Start the FastAPI application with Uvicorn:

```bash
uvicorn goldman_ai.api.app:app --reload --host 0.0.0.0 --port 8000
```

Open the web app:

```text
http://localhost:8000/
```

Health check:

```bash
curl http://localhost:8000/health
```

## Generate output result

```bash
curl -X POST http://localhost:8000/generate/text-to-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"A futuristic skyline at sunset",
    "samples":1,
    "sampler_iterations":3,
    "processing_params":{"blur":0.3,"noise":0.04,"sharpen":0.25}
  }'
```

Example response snippet:

```json
{
  "task": "text_to_image",
  "output_images": ["/outputs/text_to_image_1.png"],
  "sampler_debug": [{"iterations": 3, "trace": [...]}]
}
```

## Download result

After generation, download the file by name:

```bash
curl -OJ http://localhost:8000/download/text_to_image_1.png
```

You can also access static output directly:

```text
http://localhost:8000/outputs/text_to_image_1.png
```

## Input image references

- Pipelines accept image references as a local path, URL, or uploaded temporary file path.
- URL references are validated and cached under `./outputs/inputs/` before inference.
- Image references are converted to backend-compatible PIL objects before model calls.


## Image ingestion modes (JSON vs multipart)

For image-required capabilities (`/generate/image-to-image`, `/edit/*`) you can use either:

1. **JSON reference mode** (existing endpoints)
   - Send `image` / `mask_or_box` as local path or URL string.
2. **Multipart upload mode** (new `/upload` variants)
   - `/generate/image-to-image/upload`
   - `/edit/add-object/upload`
   - `/edit/remove-object/upload`
   - `/edit/restyle/upload`

### Precedence rules

In upload endpoints, source selection is:
1. uploaded file field (`image_file`, `mask_file`)
2. fallback reference field (`image`, `mask_or_box`)

### Validation rules

- Allowed MIME types: `image/png`, `image/jpeg`, `image/jpg`, `image/webp`.
- Maximum upload size: `10 MiB`.
- Path traversal is blocked for local path references and download filenames.

### Multipart examples

```bash
curl -X POST http://localhost:8000/generate/image-to-image/upload \
  -F 'prompt=Turn this into watercolor' \
  -F 'image_file=@./assets/input.png' \
  -F 'strength=0.7' \
  -F 'samples=1'
```

```bash
curl -X POST http://localhost:8000/edit/add-object/upload \
  -F 'object_prompt=a red balloon' \
  -F 'image_file=@./assets/scene.png' \
  -F 'mask_file=@./assets/object-mask.png'
```

## Notes

- `output_images` always contains consumable output URLs.
- The app writes placeholder PNG artifacts into `./outputs/` so web and download flows are testable end-to-end.
- Processing-step details are returned separately under `sampler_debug`.
- Replace `GoldmanAIModel.load()` and pipeline internals with concrete model backend code (e.g., Diffusers) for production inference.


- `GoldmanAIModel` now attempts to load Diffusers pipelines when dependencies are available; otherwise it falls back to generated placeholder images.


## Telemetry and debug modes

Each generation/edit response includes `run_info` with:
- `seed`
- `sampler_iterations`
- `model_identifier`
- `backend`
- `elapsed_ms`
- `stage_timings_ms` (`load`, `preprocess`, `infer`, `postprocess`, `save`)

By default, API requests run in **production mode** (`include_sampler_debug=false`) to keep payloads compact.
Set `include_sampler_debug=true` for **debug mode** to include full `sampler_debug` traces.

Example production-mode JSON request:

```json
{
  "prompt": "A futuristic skyline",
  "samples": 1,
  "include_sampler_debug": false
}
```

Example debug-mode JSON request:

```json
{
  "prompt": "A futuristic skyline",
  "samples": 1,
  "include_sampler_debug": true,
  "processing_params": {"blur": 0.2, "noise": 0.05, "sharpen": 0.2}
}
```

The API also emits structured logs with `request_id`, `endpoint`, and `output_images` for each inference call.
