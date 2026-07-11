from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from tools.dataset_labeler.core import CLASS_DEFINITIONS, create_or_update_project, load_project, replace_image_annotations, save_project
from tools.dataset_labeler.exporters import export_coco, export_yolo
from tools.dataset_labeler.splitting import create_split_manifest


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), color).save(path)


def add_polygon(project_dir: Path, image_id: int, class_id: int, x_offset: int = 0) -> None:
    project = load_project(project_dir)
    points = [
        [10.0 + x_offset, 10.0],
        [40.0 + x_offset, 10.0],
        [40.0 + x_offset, 40.0],
        [10.0 + x_offset, 40.0],
    ]
    replace_image_annotations(
        project,
        image_id,
        [{"category_id": class_id, "points": points}],
        {"status": "reviewed", "reviewed": True},
    )
    save_project(project, project_dir)


def make_project(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "raw"
    make_image(source / "healthy" / "h1.png", (20, 200, 60))
    make_image(source / "citrus canker" / "c1.png", (200, 40, 40))
    make_image(source / "melanose" / "m1.png", (210, 140, 30))
    project_dir = tmp_path / "dataset" / "annotations"
    create_or_update_project(source, project_dir)
    return source, project_dir


def test_class_id_stability() -> None:
    assert [(cls["id"], cls["name"]) for cls in CLASS_DEFINITIONS] == [
        (0, "healthy"),
        (1, "citrus_canker"),
        (2, "melanose"),
    ]


def test_coco_export_contains_required_instance_fields(tmp_path: Path) -> None:
    _, project_dir = make_project(tmp_path)
    project = load_project(project_dir)
    healthy = next(img for img in project["images"] if img["source_class"] == "healthy")
    add_polygon(project_dir, healthy["id"], 0)
    for image in load_project(project_dir)["images"]:
        if image["id"] != healthy["id"]:
            image["excluded"] = True
            image["exclusion_reason"] = "test fixture"
    project = load_project(project_dir)
    save_project(project, project_dir)

    summary = export_coco(project_dir)
    coco = json.loads(Path(summary["path"]).read_text(encoding="utf-8"))
    assert coco["categories"][0]["name"] == "healthy"
    ann = coco["annotations"][0]
    for key in ("id", "image_id", "category_id", "segmentation", "bbox", "area", "iscrowd"):
        assert key in ann
    assert ann["category_id"] == 0
    assert ann["area"] > 0


def test_yolo_export_multiple_instances_and_matching_stems(tmp_path: Path) -> None:
    _, project_dir = make_project(tmp_path)
    project = load_project(project_dir)
    for image in project["images"]:
        class_id = int(image["source_class_id"])
        replace_image_annotations(
            project,
            image["id"],
            [
                {"category_id": class_id, "points": [[10, 10], [30, 10], [30, 30], [10, 30]]},
                {"category_id": class_id, "points": [[45, 15], [65, 15], [65, 35], [45, 35]]},
            ],
            {"status": "reviewed", "reviewed": True},
        )
    save_project(project, project_dir)

    output = tmp_path / "exports" / "yolo_seg"
    summary = export_yolo(project_dir, output, include_hsv=False)
    data_yaml = yaml.safe_load(Path(summary["data_yaml"]).read_text(encoding="utf-8"))
    assert data_yaml["names"] == {0: "healthy", 1: "citrus_canker", 2: "melanose"}
    manifest = json.loads((output / "manifests" / "export_manifest.json").read_text(encoding="utf-8"))
    row = manifest["rows"][0]
    assert Path(row["exported_image"]).stem == Path(row["exported_label"]).stem
    label_path = output / row["exported_label"]
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(len(line.split()) == 9 for line in lines)


def test_empty_label_file_for_unannotated_image_when_invalid_export_allowed(tmp_path: Path) -> None:
    _, project_dir = make_project(tmp_path)
    project = load_project(project_dir)
    for image in project["images"][1:]:
        image["excluded"] = True
        image["exclusion_reason"] = "not part of fixture"
    save_project(project, project_dir)
    create_split_manifest(project_dir)
    output = tmp_path / "exports" / "yolo_seg"
    summary = export_yolo(project_dir, output, include_hsv=False, allow_invalid=True)
    manifest = json.loads((output / "manifests" / "export_manifest.json").read_text(encoding="utf-8"))
    label_path = output / manifest["rows"][0]["exported_label"]
    assert label_path.exists()
    assert label_path.read_text(encoding="utf-8") == ""
    assert summary["validation_warnings"]
