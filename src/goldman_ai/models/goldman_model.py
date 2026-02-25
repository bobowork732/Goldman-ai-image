"""Model wrapper with optional Diffusers backend integration."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

from goldman_ai.config import InferenceConfig


class GoldmanAIModel:
    """Model wrapper with lazy backend loading and explicit inference APIs."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._loaded = False
        self._backend = "placeholder"

        # Persistent pipeline handles (Diffusers when available)
        self._txt2img_pipeline: Any | None = None
        self._img2img_pipeline: Any | None = None
        self._inpaint_pipeline: Any | None = None

    def load(self) -> None:
        """Initialize backend pipelines once and pin device/dtype options."""
        if self._loaded:
            return

        try:
            import torch  # type: ignore
            from diffusers import (  # type: ignore
                AutoPipelineForImage2Image,
                AutoPipelineForInpainting,
                AutoPipelineForText2Image,
            )

            model_id = self.config.model_name
            dtype = torch.float16 if self.config.device.startswith("cuda") else torch.float32

            self._txt2img_pipeline = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
            self._img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(model_id, torch_dtype=dtype)
            self._inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=dtype)

            self._txt2img_pipeline = self._txt2img_pipeline.to(self.config.device)
            self._img2img_pipeline = self._img2img_pipeline.to(self.config.device)
            self._inpaint_pipeline = self._inpaint_pipeline.to(self.config.device)

            for pipe in (self._txt2img_pipeline, self._img2img_pipeline, self._inpaint_pipeline):
                if hasattr(pipe, "set_progress_bar_config"):
                    pipe.set_progress_bar_config(disable=True)

            self._backend = "diffusers"
        except Exception:
            # Keep working with a deterministic placeholder backend.
            self._backend = "placeholder"

        self._loaded = True

    def get_generation_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Merge defaults with request-level overrides."""
        kwargs = asdict(self.config)
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        return kwargs

    def _placeholder_image(self, label: str) -> Any:
        """Return an in-memory placeholder image object/bytes."""
        try:
            from PIL import Image, ImageDraw

            image = Image.new("RGB", (self.config.image_width, self.config.image_height), color=(32, 32, 32))
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), f"Goldman AI\n{label}", fill=(255, 255, 255))
            return image
        except Exception:
            # Fallback to minimal valid PNG bytes if PIL is unavailable.
            return (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
                b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )

    def generate_text_to_image(self, *, prompt: str, **overrides: Any) -> list[Any]:
        settings = self.get_generation_kwargs(prompt=prompt, **overrides)
        self.load()

        if self._backend == "diffusers" and self._txt2img_pipeline is not None:
            result = self._txt2img_pipeline(
                prompt=prompt,
                num_images_per_prompt=settings["sample_count"],
                guidance_scale=settings["guidance_scale"],
                num_inference_steps=settings["num_inference_steps"],
            )
            return list(result.images)

        return [self._placeholder_image(f"text-to-image: {prompt}") for _ in range(settings["sample_count"])]

    def generate_image_to_image(self, *, prompt: str, image: str, strength: float = 0.8, **overrides: Any) -> list[Any]:
        settings = self.get_generation_kwargs(prompt=prompt, image=image, strength=strength, **overrides)
        self.load()

        if self._backend == "diffusers" and self._img2img_pipeline is not None:
            try:
                from PIL import Image

                init_image = Image.open(image).convert("RGB")
                result = self._img2img_pipeline(
                    prompt=prompt,
                    image=init_image,
                    strength=strength,
                    num_images_per_prompt=settings["sample_count"],
                    guidance_scale=settings["guidance_scale"],
                    num_inference_steps=settings["num_inference_steps"],
                )
                return list(result.images)
            except Exception:
                pass

        return [self._placeholder_image(f"image-to-image: {Path(image).name}") for _ in range(settings["sample_count"])]

    def edit_add_object(self, *, image: str, object_prompt: str, mask_or_box: str, **overrides: Any) -> list[Any]:
        settings = self.get_generation_kwargs(image=image, object_prompt=object_prompt, mask_or_box=mask_or_box, **overrides)
        self.load()

        if self._backend == "diffusers" and self._inpaint_pipeline is not None:
            try:
                from PIL import Image

                init_image = Image.open(image).convert("RGB")
                mask_image = Image.open(mask_or_box).convert("RGB")
                result = self._inpaint_pipeline(
                    prompt=object_prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_images_per_prompt=settings["sample_count"],
                    guidance_scale=settings["guidance_scale"],
                    num_inference_steps=settings["num_inference_steps"],
                )
                return list(result.images)
            except Exception:
                pass

        return [self._placeholder_image("edit-add-object") for _ in range(settings["sample_count"])]

    def edit_remove_object(self, *, image: str, mask_or_box: str, **overrides: Any) -> list[Any]:
        settings = self.get_generation_kwargs(image=image, mask_or_box=mask_or_box, **overrides)
        self.load()

        if self._backend == "diffusers" and self._inpaint_pipeline is not None:
            try:
                from PIL import Image

                init_image = Image.open(image).convert("RGB")
                mask_image = Image.open(mask_or_box).convert("RGB")
                result = self._inpaint_pipeline(
                    prompt="remove object",
                    image=init_image,
                    mask_image=mask_image,
                    num_images_per_prompt=settings["sample_count"],
                    guidance_scale=settings["guidance_scale"],
                    num_inference_steps=settings["num_inference_steps"],
                )
                return list(result.images)
            except Exception:
                pass

        return [self._placeholder_image("edit-remove-object") for _ in range(settings["sample_count"])]

    def edit_restyle(self, *, image: str, style_prompt: str, **overrides: Any) -> list[Any]:
        settings = self.get_generation_kwargs(image=image, style_prompt=style_prompt, **overrides)
        return self.generate_image_to_image(
            prompt=style_prompt,
            image=image,
            strength=0.8,
            sample_count=settings["sample_count"],
            guidance_scale=settings["guidance_scale"],
            num_inference_steps=settings["num_inference_steps"],
        )
