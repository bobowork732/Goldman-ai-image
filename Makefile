PYTHON ?= python
CONFIG ?= configs/base.yaml
CKPT ?= checkpoints/final.pt
PROMPT ?= "a cinematic portrait of a robot"

.PHONY: bootstrap train-vae train-diffusion txt2img eval

bootstrap:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

train-vae:
	$(PYTHON) src/train/train_diffusion.py --config $(CONFIG)

train-diffusion:
	$(PYTHON) src/train/train_diffusion.py --config $(CONFIG)

txt2img:
	$(PYTHON) scripts/txt2img.py --checkpoint $(CKPT) --prompt $(PROMPT)

eval:
	$(PYTHON) scripts/prepare_data.py --data-root ./data --batch-size 4 --num-preview-batches 1
