from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from tools.dataset_labeler.app import create_app
from tools.dataset_labeler.core import DEFAULT_SEGMENTATION_CLASS_NAMES, load_project, replace_image_annotations, save_project
from tools.dataset_labeler.exporters import export_dataset
from tools.dataset_labeler.splitting import deterministic_split_assignments
from tools.dataset_labeler.validation import validate_project
from tools.dataset_labeler.video_import import VideoImportCancelled, create_video_project, inspect_videos


def _video(path: Path, *, width: int = 96, height: int = 64, fps: float = 10.0, frames: int = 20) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        pytest.skip("The OpenCV MJPG test-video encoder is unavailable.")
    try:
        for index in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = index * 7 % 255
            frame[:, :, 1] = 80
            cv2.putText(frame, str(index), (5, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_video_project_extracts_full_frames_and_groups_split(tmp_path: Path) -> None:
    first = _video(tmp_path / "capture.avi", width=96, height=64)
    second_dir = tmp_path / "other"
    second_dir.mkdir()
    second = _video(second_dir / "capture.avi", width=80, height=60, frames=15)
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)}

    result = create_video_project(
        [first, second], tmp_path / "projects", "Video dataset", "detection", ["Fruit", "Leaf"]
    )
    project_dir = Path(result["project_dir"])
    project = load_project(project_dir)
    assert project["annotation_mode"] == "detection"
    assert project["source_kind"] == "video_frames"
    assert [item["display_name"] for item in project["detection_classes"]] == ["Fruit", "Leaf"]
    assert len(project["images"]) == 7
    assert {image["source_video_name"] for image in project["images"]} == {"capture.avi"}
    assert len({image["source_video_id"] for image in project["images"]}) == 2
    assert {image["split"] for image in project["images"]} == {"train", "val"}
    for video_id in {image["source_video_id"] for image in project["images"]}:
        assert len({image["split"] for image in project["images"] if image["source_video_id"] == video_id}) == 1
    assert sorted(image["sample_time_ms"] for image in project["images"] if image["width"] == 96) == [0, 500, 1000, 1500]
    for image in project["images"]:
        with Image.open(Path(project["source_root"]) / image["relative_path"]) as opened:
            assert opened.size == (image["width"], image["height"])
    manifest = json.loads((project_dir / "manifests" / "video_import_manifest.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["interval_seconds"] == 0.5
    assert manifest["settings"]["jpeg_quality"] == 95
    assert manifest["settings"]["split_ratios"] == {"train": 0.7, "val": 0.2, "test": 0.1}
    assert all("path" not in video for video in manifest["videos"])
    assert any("test set is empty" in warning for warning in result["warnings"])
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)} == source_hashes


def test_three_videos_create_atomic_train_validation_and_test_sets(tmp_path: Path) -> None:
    videos = [
        _video(tmp_path / f"source-{index}.avi", width=96 + index * 8, frames=5)
        for index in range(3)
    ]
    result = create_video_project(
        videos, tmp_path / "projects", "Three-way dataset", "detection", ["Object"]
    )
    project = load_project(Path(result["project_dir"]))

    assert len(project["images"]) == 3
    assert {image["split"] for image in project["images"]} == {"train", "val", "test"}
    assert {split: result["counts"]["splits"][split]["images"] for split in ("train", "val", "test")} == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    for video_id in {image["source_video_id"] for image in project["images"]}:
        assert len({image["split"] for image in project["images"] if image["source_video_id"] == video_id}) == 1

    exported = export_dataset(
        result["project_dir"], tmp_path / "exports", formats=["yolo26_detection"]
    )
    export_root = Path(exported["dataset_root"]) / "yolo26_detection"
    assert (export_root / "images" / "test").is_dir()
    assert (export_root / "labels" / "test").is_dir()
    assert "test: images/test" in (export_root / "data.yaml").read_text(encoding="utf-8")


def test_classification_project_exports_trainable_imagefolder_dataset(tmp_path: Path) -> None:
    videos = [
        _video(tmp_path / f"class-source-{index}.avi", width=96 + index * 8, frames=5)
        for index in range(3)
    ]
    result = create_video_project(
        videos,
        tmp_path / "projects",
        "Fruit classifier",
        "classification",
        ["Healthy fruit", "Reject fruit"],
    )
    project_dir = Path(result["project_dir"])
    project = load_project(project_dir)
    assert project["annotation_mode"] == "classification"
    assert [item["display_name"] for item in project["classification_classes"]] == [
        "Healthy fruit",
        "Reject fruit",
    ]
    assert any("no image-level class" in error for error in validate_project(project_dir)["errors"])

    for index, image in enumerate(project["images"]):
        class_item = project["classification_classes"][index % 2]
        replace_image_annotations(
            project,
            image["id"],
            [],
            {"status": "approved", "classification_class_uuid": class_item["uuid"]},
        )
    save_project(project, project_dir)
    assert validate_project(project_dir)["ok"]

    exported = export_dataset(
        project_dir,
        tmp_path / "exports",
        formats=["image_classification"],
    )
    export_root = Path(exported["classification_dir"])
    metadata = yaml.safe_load((export_root / "dataset.yaml").read_text(encoding="utf-8"))
    assert metadata["task"] == "classify"
    assert metadata["names"] == {0: "healthy_fruit", 1: "reject_fruit"}
    assert metadata["class_folders"] == {0: "000-healthy_fruit", 1: "001-reject_fruit"}
    assert sum(1 for path in export_root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}) == 3
    assert all((export_root / split).is_dir() for split in ("train", "val", "test"))
    manifest = json.loads((Path(exported["dataset_root"]) / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"] == {"image_classification": "image_classification/dataset.yaml"}
    assert manifest["counts"]["classification_images"] == 3


def test_default_ten_group_split_is_seventy_twenty_ten() -> None:
    assignments = deterministic_split_assignments(
        [f"video-{index}" for index in range(10)],
        {"train": 0.7, "val": 0.2, "test": 0.1},
        seed=42,
    )
    assert list(assignments.values()).count("train") == 7
    assert list(assignments.values()).count("val") == 2
    assert list(assignments.values()).count("test") == 1


def test_video_inspection_deduplicates_and_skips_corrupt_inputs(tmp_path: Path) -> None:
    valid = _video(tmp_path / "valid.avi", frames=5)
    corrupt = tmp_path / "broken.mp4"
    corrupt.write_bytes(b"not a video")
    inspected = inspect_videos([valid, valid, corrupt])
    assert inspected["video_count"] == 1
    assert inspected["expected_frames"] == 1
    assert any("duplicate" in warning.lower() for warning in inspected["warnings"])
    assert any("broken.mp4" in warning for warning in inspected["warnings"])


def test_video_import_cancellation_and_destination_protection(tmp_path: Path) -> None:
    video = _video(tmp_path / "capture.avi")
    with pytest.raises(VideoImportCancelled):
        create_video_project(
            [video], tmp_path / "projects", "cancelled", "segmentation", ["Object"], cancel_check=lambda: True
        )
    projects = tmp_path / "projects"
    assert not (projects / "cancelled").exists()
    assert not list(projects.glob(".cancelled.importing-*"))

    occupied = projects / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_video_project([video], projects, "occupied", "detection", ["Object"])
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_task_specific_export_and_video_job_api(tmp_path: Path) -> None:
    videos = [_video(tmp_path / f"capture-{index}.avi", frames=8 + index) for index in range(2)]
    client = TestClient(create_app(default_project_dir=None, provider_name="mock"))
    inspection = client.post("/api/videos/inspect", json={"video_paths": [str(path) for path in videos]})
    assert inspection.status_code == 200
    assert inspection.json()["video_count"] == 2

    started = client.post("/api/videos/jobs", json={
        "video_paths": [str(path) for path in videos],
        "output_parent": str(tmp_path / "projects"),
        "project_name": "API project",
        "annotation_mode": "segmentation",
    })
    assert started.status_code == 200
    job_id = started.json()["job_id"]
    deadline = time.monotonic() + 20
    job = {}
    while time.monotonic() < deadline:
        job = client.get(f"/api/videos/jobs/{job_id}").json()
        if job["state"] in {"complete", "failed", "cancelled"}:
            break
        time.sleep(0.03)
    assert job["state"] == "complete", job
    assert [item["display_name"] for item in job["result"]["project"]["segmentation_classes"]] == DEFAULT_SEGMENTATION_CLASS_NAMES
    project_dir = Path(job["result"]["project_dir"])
    state = client.get("/api/state")
    assert state.status_code == 200
    assert state.json()["opened"] is True
    assert Path(state.json()["project_dir"]) == project_dir
    first_image_id = job["result"]["project"]["images"][0]["id"]
    loaded = client.get(f"/api/images/{first_image_id}")
    assert loaded.status_code == 200
    assert loaded.json()["image"]["id"] == first_image_id
    exported = export_dataset(
        project_dir,
        tmp_path / "exports",
        formats=["yolo26_segmentation"],
    )
    export_root = Path(exported["dataset_root"])
    assert (export_root / "yolo26_segmentation" / "data.yaml").is_file()
    assert not (export_root / "yolo26_detection").exists()
    assert not (export_root / "dfine_detection").exists()
    manifest = json.loads((export_root / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"] == {"yolo26_segmentation": "yolo26_segmentation/data.yaml"}
