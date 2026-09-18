from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from tools.dataset_labeler.app import create_app
from tools.dataset_labeler.core import create_or_update_project, load_project, save_project
from tools.dataset_labeler.detection_providers import non_maximum_suppression


def _project(tmp_path: Path, mode: str = "detection") -> Path:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (100, 80), (230, 145, 40)).save(source / "orange.jpg", quality=95)
    project_dir = tmp_path / "project"
    project = create_or_update_project(source, project_dir, "citrus")
    project["annotation_mode"] = mode
    save_project(project, project_dir)
    return project_dir


def test_non_maximum_suppression_keeps_best_non_overlapping_boxes() -> None:
    proposals = [
        {"id": "best", "score": 0.9, "bbox": [10, 10, 30, 30]},
        {"id": "overlap", "score": 0.7, "bbox": [11, 11, 30, 30]},
        {"id": "separate", "score": 0.8, "bbox": [60, 10, 20, 20]},
    ]
    kept = non_maximum_suppression(proposals, threshold=0.55)
    assert [item["id"] for item in kept] == ["best", "separate"]


def test_detection_api_returns_preview_boxes_without_saving_them(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))
    status = client.get("/api/detection/status")
    assert status.status_code == 200
    assert status.json()["provider"] == "mock"
    assert client.post("/api/detection/load", json={}).json()["state"] == "ready"

    image = client.get("/api/state").json()["project"]["images"][0]
    response = client.post(
        "/api/detection/predict",
        json={
            "image_id": image["id"],
            "request_id": "request-1",
            "labels": ["orange", "naartjie"],
            "confidence": 0.3,
            "text_threshold": 0.25,
            "max_detections": 10,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "request-1"
    assert len(payload["proposals"]) == 2
    for proposal in payload["proposals"]:
        x, y, width, height = proposal["bbox"]
        assert 0 <= x < 100 and 0 <= y < 80
        assert width > 0 and height > 0
        assert x + width <= 100 and y + height <= 80

    assert load_project(project_dir)["annotations"] == []


def test_detection_api_is_hidden_from_segmentation_projects(tmp_path: Path) -> None:
    project_dir = _project(tmp_path, mode="segmentation")
    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))
    image = client.get("/api/state").json()["project"]["images"][0]
    response = client.post(
        "/api/detection/predict",
        json={"image_id": image["id"], "request_id": "request-2", "labels": ["orange"]},
    )
    assert response.status_code == 400
    assert "Detection" in response.json()["detail"]
