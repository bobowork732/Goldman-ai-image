from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from goldman_ai.api.app import app, model

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    model.config.output_dir = str(tmp_path / "outputs")
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def image_files(tmp_path: Path) -> tuple[str, str]:
    image = tmp_path / "input.png"
    mask = tmp_path / "mask.png"
    image.write_bytes(PNG_BYTES)
    mask.write_bytes(PNG_BYTES)
    return str(image), str(mask)


def _assert_artifacts(resp_json: dict, output_dir: Path) -> None:
    assert "task" in resp_json
    assert isinstance(resp_json.get("output_images"), list)
    assert resp_json["output_images"], "expected at least one output image"
    for url in resp_json["output_images"]:
        assert url.startswith("/outputs/")
        artifact = output_dir / Path(url).name
        assert artifact.exists(), f"missing artifact: {artifact}"


@pytest.mark.expensive
@pytest.mark.parametrize(
    ("endpoint", "payload_builder"),
    [
        (
            "/generate/text-to-image",
            lambda files: {
                "prompt": "A calm lake",
                "samples": 1,
                "seed": 123,
                "sampler_iterations": 1,
                "include_sampler_debug": False,
            },
        ),
        (
            "/generate/image-to-image",
            lambda files: {
                "prompt": "Watercolor version",
                "image": files[0],
                "strength": 0.5,
                "samples": 1,
                "seed": 123,
                "sampler_iterations": 1,
                "include_sampler_debug": False,
            },
        ),
        (
            "/edit/add-object",
            lambda files: {
                "image": files[0],
                "object_prompt": "red balloon",
                "mask_or_box": files[1],
                "samples": 1,
                "seed": 123,
                "sampler_iterations": 1,
                "include_sampler_debug": False,
            },
        ),
        (
            "/edit/remove-object",
            lambda files: {
                "image": files[0],
                "mask_or_box": files[1],
                "samples": 1,
                "seed": 123,
                "sampler_iterations": 1,
                "include_sampler_debug": False,
            },
        ),
        (
            "/edit/restyle",
            lambda files: {
                "image": files[0],
                "style_prompt": "cyberpunk",
                "samples": 1,
                "seed": 123,
                "sampler_iterations": 1,
                "include_sampler_debug": False,
            },
        ),
    ],
)
def test_core_endpoints_create_artifacts(
    client: TestClient,
    image_files: tuple[str, str],
    endpoint: str,
    payload_builder,
) -> None:
    payload = payload_builder(image_files)
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("parameters", {}).get("seed") == 123
    assert body.get("run_info", {}).get("sampler_iterations") == 1
    _assert_artifacts(body, Path(model.config.output_dir))


@pytest.mark.expensive
@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        ("/generate/text-to-image", {"samples": 1}),
        (
            "/generate/image-to-image",
            {
                "prompt": "x",
                "image": "./does/not/exist.png",
                "samples": 1,
            },
        ),
        (
            "/edit/add-object",
            {
                "image": "./does/not/exist.png",
                "object_prompt": "x",
                "mask_or_box": "./no-mask.png",
            },
        ),
        (
            "/edit/remove-object",
            {
                "image": "./does/not/exist.png",
                "mask_or_box": "./no-mask.png",
            },
        ),
        (
            "/edit/restyle",
            {
                "image": "./does/not/exist.png",
                "style_prompt": "x",
            },
        ),
    ],
)
def test_core_endpoints_negative_inputs(client: TestClient, endpoint: str, payload: dict) -> None:
    response = client.post(endpoint, json=payload)
    assert response.status_code >= 400
