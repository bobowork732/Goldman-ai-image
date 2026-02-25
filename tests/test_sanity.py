from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from goldman_ai.models import GoldmanAIModel
from goldman_ai.pipelines import run_text_to_image


def test_pipeline_sanity_without_api(tmp_path: Path) -> None:
    model = GoldmanAIModel()
    model.config.output_dir = str(tmp_path / "outputs")
    res = run_text_to_image(model, "sanity", samples=1, sampler_iterations=1, include_sampler_debug=False)
    assert res["output_images"]
    assert res["sampler_debug"] is None
    artifact = Path(model.config.output_dir) / Path(res["output_images"][0]).name
    assert artifact.exists()
