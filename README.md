# Goldman-ai-image

A modular latent-diffusion scaffold with:
- VAE pretraining
- text-conditioned latent diffusion training
- DDPM/DDIM/Euler sampling
- txt2img inference CLI
- data validation + safety filter hooks

## Quickstart (single-command local bootstrap)

```bash
make bootstrap
```

Then run:

```bash
make train-diffusion
make txt2img CKPT=checkpoints/final.pt PROMPT="a sunset over snowy mountains"
```

## Data format

Expected dataset root layout:

```text
<data_root>/
  images/
    0001.png
    0002.jpg
    ...
  metadata.jsonl
```

Each `metadata.jsonl` line must be JSON with:
- `file_name`: filename under `images/`
- `text`: caption string

Example line:

```json
{"file_name": "0001.png", "text": "a red sports car on a rainy street"}
```

Validate and preview batches:

```bash
python scripts/prepare_data.py --data-root ./data --batch-size 4 --num-preview-batches 1
```

## Training stages

### Stage 1: VAE pretraining
- Optimizes reconstruction loss + KL regularization.
- Controlled by `train.vae_pretrain_steps` in `configs/base.yaml`.

### Stage 2: Latent diffusion training
- Encodes images into VAE latents.
- Trains U-Net with epsilon prediction objective.
- Uses classifier-free guidance style dropout (`train.cond_dropout`) during training.

Run both stages through the unified trainer:

```bash
make train-vae CONFIG=configs/base.yaml
# or
make train-diffusion CONFIG=configs/base.yaml
```

## Hardware requirements

### Baseline
- **GPU**: 1x NVIDIA GPU (>=16 GB VRAM recommended for 256x256, batch size 8).
- **CPU RAM**: 16 GB+
- **Disk**: depends on dataset and checkpoints (suggest 50 GB+ free)

### GPU memory guidance
- Lower `train.batch_size` and/or `train.resolution` first.
- Use mixed precision (`torch.cuda.amp.autocast`) in training loops for substantial memory savings.
- Enable gradient checkpointing in model blocks to trade compute for memory.
- Reduce `unet.base_channels` and `unet.channel_mults` for smaller model footprint.

### Multi-GPU note (DDP / Accelerate)
- Current scripts are single-process by default.
- For multi-GPU, launch with PyTorch DDP or Hugging Face Accelerate.
- Typical path:
  - wrap dataloaders with distributed sampler
  - shard effective batch size per rank
  - all-reduce/log metrics from rank 0
  - checkpoint only on main process

## Inference examples

Single prompt:

```bash
python scripts/txt2img.py \
  --checkpoint checkpoints/final.pt \
  --prompt "a photorealistic koala astronaut" \
  --negative-prompt "blurry, distorted" \
  --steps 30 \
  --guidance-scale 7.5 \
  --seed 1234 \
  --height 256 \
  --width 256
```

Prompt list + batching:

```bash
python scripts/txt2img.py \
  --checkpoint checkpoints/final.pt \
  --prompt-file prompts.txt \
  --batch-size 4 \
  --steps 40 \
  --guidance-scale 6.5 \
  --seed 42
```

Safety filter toggle:

```bash
python scripts/txt2img.py --checkpoint checkpoints/final.pt --prompt "..." --safety-filter
```

## Makefile targets

- `make bootstrap` — install project locally
- `make train-vae` — run trainer (includes VAE stage)
- `make train-diffusion` — run trainer (full schedule)
- `make txt2img` — generate image(s) from checkpoint
- `make eval` — run dataset validation/preview helper

## Model card template

Use this template when publishing checkpoints.

### Model details
- **Model name**:
- **Version**:
- **Date**:
- **Authors / Contact**:

### Intended use
- Primary use cases:
- Out-of-scope use cases:

### Training data
- Dataset sources:
- Data filtering steps:
- Captioning strategy:

### Training procedure
- Hardware:
- Batch size / resolution:
- Total steps:
- Optimizer / LR:
- EMA usage:

### Evaluation
- Metrics reported (FID/CLIP/proxy):
- Validation split definition:
- Known failure cases:

### Safety and limitations
- Potential harmful outputs:
- Safety mitigations (prompt filter / classifier):
- Bias and fairness considerations:
- Not recommended for safety-critical decisions.
