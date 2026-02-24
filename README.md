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
│   └── processing.py
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
   pip install fastapi uvicorn pydantic
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

## Run local inference API

Start the FastAPI application with Uvicorn:

```bash
uvicorn goldman_ai.api.app:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Generate and edit examples

```bash
curl -X POST http://localhost:8000/generate/text-to-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"A futuristic skyline at sunset",
    "samples":2,
    "sampler_iterations":6,
    "seed":42,
    "size":"512x512",
    "processing_params":{"blur":0.3,"noise":0.04,"sharpen":0.25}
  }'
```

```bash
curl -X POST http://localhost:8000/generate/image-to-image \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"Turn this into watercolor",
    "image":"./assets/input.png",
    "strength":0.7,
    "samples":2,
    "sampler_iterations":4,
    "processing_params":{"blur":0.2,"noise":0.03,"sharpen":0.1}
  }'
```

```bash
curl -X POST http://localhost:8000/edit/add-object \
  -H "Content-Type: application/json" \
  -d '{
    "image":"./assets/scene.png",
    "object_prompt":"a red balloon",
    "mask_or_box":"x=100,y=120,w=80,h=100",
    "samples":1,
    "sampler_iterations":3,
    "processing_params":{"blur":0.15,"noise":0.05,"sharpen":0.3}
  }'
```

```bash
curl -X POST http://localhost:8000/edit/remove-object \
  -H "Content-Type: application/json" \
  -d '{
    "image":"./assets/scene.png",
    "mask_or_box":"./assets/object-mask.png",
    "samples":1,
    "sampler_iterations":3,
    "processing_params":{"blur":0.12,"noise":0.03,"sharpen":0.2}
  }'
```

```bash
curl -X POST http://localhost:8000/edit/restyle \
  -H "Content-Type: application/json" \
  -d '{
    "image":"./assets/scene.png",
    "style_prompt":"cyberpunk neon",
    "samples":2,
    "sampler_iterations":5,
    "processing_params":{"blur":0.25,"noise":0.02,"sharpen":0.35}
  }'
```

## Notes

- `output_images` always contains consumable output image paths/URLs.
- Processing-step details are returned separately under `sampler_debug`.
- Replace `GoldmanAIModel.load()` and pipeline internals with concrete model backend code (e.g., Diffusers) for production inference.
