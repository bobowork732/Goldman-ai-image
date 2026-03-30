# Goldman AI Image Generator

A Python image generator called **Goldman AI** that creates realistic images from text prompts using a stronger photorealistic model.

## Features
- Generates photorealistic images from prompts.
- Uses `SG161222/RealVisXL_V5.0` by default for better photorealism.
- Uses `madebyollin/sdxl-vae-fp16-fix` as default VAE for cleaner details.
- Supports negative prompts, seeds, resolution settings, and guidance controls.
- Works on CUDA (faster) or CPU (slower).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python app.py "a luxury penthouse interior in New York at sunset"
```

The generated image is saved to `outputs/`.

### More examples
```bash
python app.py "a photorealistic portrait of a business leader in a modern office"
python app.py "a red sports car parked outside a glass skyscraper, rainy night" --seed 42
python app.py "a cinematic skyline of lower Manhattan" --width 1024 --height 576 --steps 45
python app.py "a close-up portrait with detailed eyes and skin texture" --vae madebyollin/sdxl-vae-fp16-fix
```

## Optional: use a different model
```bash
python app.py "a realistic golden retriever in a studio" --model stabilityai/stable-diffusion-xl-base-1.0
```

## Notes
- First run downloads model weights from Hugging Face.
- For some models, you may need to authenticate with `huggingface-cli login`.
- GPU with at least 8GB VRAM is strongly recommended for good speed.
- You can point `--model` to your own local model directory as well.
- You can disable custom VAE with `--vae none`.

## Project files
- `app.py`: CLI entrypoint and argument parsing.
- `model.py`: model pipeline loading + generation workflow.
- `vae.py`: VAE loading helper and default VAE configuration.
