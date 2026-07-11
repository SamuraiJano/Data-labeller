from __future__ import annotations

import json
from pathlib import Path

import cv2
from PIL import Image

from tools.dataset_labeler.core import create_or_update_project
from tools.dataset_labeler.hsv import create_hsv_from_yolo_rgb
from tools.dataset_labeler.splitting import create_split_manifest


def save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def test_deterministic_grouped_split_and_duplicate_leakage_prevention(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    save_image(raw / "healthy" / "fruit_001.png", (0, 255, 0))
    save_image(raw / "healthy" / "fruit_002.png", (0, 254, 0))
    save_image(raw / "citrus canker" / "duplicate_a.png", (255, 0, 0))
    save_image(raw / "melanose" / "duplicate_b.png", (255, 0, 0))
    save_image(raw / "melanose" / "other_001.png", (255, 150, 0))
    project_dir = tmp_path / "annotations"
    create_or_update_project(raw, project_dir)

    first = create_split_manifest(project_dir, seed=123)
    second = create_split_manifest(project_dir, seed=123)
    assert first["rows"] == second["rows"]

    by_hash: dict[str, set[str]] = {}
    for row in first["rows"]:
        by_hash.setdefault(row["sha256"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_hash.values())

    grouped = [row for row in first["rows"] if row["relative_path"].endswith(("fruit_001.png", "fruit_002.png"))]
    assert len({row["split"] for row in grouped}) == 1


def test_hsv_filename_mapping_and_channel_correctness(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb"
    for split in ("train", "val", "test"):
        (rgb_root / "images" / split).mkdir(parents=True)
        (rgb_root / "labels" / split).mkdir(parents=True)
    save_image(rgb_root / "images" / "train" / "red.png", (255, 0, 0))
    (rgb_root / "labels" / "train" / "red.txt").write_text("", encoding="utf-8")

    classes = [
        {"id": 0, "name": "healthy"},
        {"id": 1, "name": "citrus_canker"},
        {"id": 2, "name": "melanose"},
    ]
    hsv_root = tmp_path / "hsv"
    summary = create_hsv_from_yolo_rgb(rgb_root, hsv_root, classes)
    mapping = json.loads(Path(summary["mapping_json"]).read_text(encoding="utf-8"))
    assert mapping["rows"][0]["rgb_image"] == "images/train/red.png"
    assert mapping["rows"][0]["hsv_image"] == "images/train/red.png"
    assert (hsv_root / "labels" / "train" / "red.txt").read_text(encoding="utf-8") == ""

    hsv = cv2.imread(str(hsv_root / "images" / "train" / "red.png"), cv2.IMREAD_COLOR)
    assert hsv is not None
    assert hsv[0, 0].tolist() == [0, 255, 255]
