"""Global configuration defaults for local inference."""

from dataclasses import dataclass

DEFAULT_SAMPLES = 20
DEFAULT_SAMPLER_ITERATIONS = 20


@dataclass(slots=True)
class InferenceConfig:
    """Configuration shared by model wrappers and API endpoints."""

    model_name: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cpu"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    sample_count: int = DEFAULT_SAMPLES
    sampler_iterations: int = DEFAULT_SAMPLER_ITERATIONS
    seed: int | None = None
    image_height: int = 512
    image_width: int = 512
    output_dir: str = "./outputs"
