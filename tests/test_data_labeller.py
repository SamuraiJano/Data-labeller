from __future__ import annotations

import base64
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from tools.dataset_labeler.app import SyntheticGenerateRequest, create_app
from tools.dataset_labeler.core import (
    DEFAULT_SEGMENTATION_CLASS_NAMES,
    LEGACY_GENERIC_SEGMENTATION_CLASS_UUID,
    SEGMENTATION_CLASS_DEFINITIONS,
    create_or_update_project,
    load_project,
    replace_image_annotations,
    save_project,
    sha256_file,
)
from tools.dataset_labeler.exporters import export_dataset, validate_yolo_row
from tools.dataset_labeler.providers import ProviderManager, SAM3_OFFICIAL_REPOSITORY
from tools.dataset_labeler.splitting import create_split_manifest
from tools.dataset_labeler.synthetic import _partition_foregrounds, _plan_object_counts, generate_synthetic_project
from tools.dataset_labeler.validation import validate_project


def create_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(5):
        Image.new("RGB", (64, 48), (20 + index, 100, 40)).save(source / f"image_{index}.png")
    project_dir = tmp_path / "project"
    project = create_or_update_project(source, project_dir, "fruit-project")
    detection = project["detection_classes"][0]
    defect = project["segmentation_classes"][0]
    for index, image in enumerate(project["images"]):
        annotations: list[dict] = []
        if index < 4:
            annotations.append({
                "type": "fruit_box",
                "class_uuid": detection["uuid"],
                "bbox": [8.0, 6.0, 40.0, 30.0],
                "source": "manual",
            })
        if index == 0:
            annotations.extend([
                {
                    "type": "defect_mask",
                    "class_uuid": defect["uuid"],
                    "points": [[10, 10], [22, 10], [22, 22], [10, 22]],
                    "source": "manual",
                },
                {
                    "type": "foreground_mask",
                    "points": [[8, 6], [48, 6], [48, 36], [8, 36]],
                    "source": "manual",
                },
            ])
        replace_image_annotations(project, image["id"], annotations, {"status": "approved"})
    save_project(project, project_dir)
    create_split_manifest(project_dir, {"train": 0.8, "val": 0.2, "test": 0.0}, seed=19)
    return project_dir


def test_default_segmentation_classes_use_requested_defect_order() -> None:
    assert DEFAULT_SEGMENTATION_CLASS_NAMES == [
        "Skin wrinkling",
        "Dark marking",
        "Dark scarring",
        "Wind scarring",
        "Light-brown discolouration",
        "Rough skin",
        "Open skin",
    ]
    assert [item["name"] for item in SEGMENTATION_CLASS_DEFINITIONS] == [
        "skin_wrinkling",
        "dark_marking",
        "dark_scarring",
        "wind_scarring",
        "light_brown_discolouration",
        "rough_skin",
        "open_skin",
    ]
    assert [item["export_index"] for item in SEGMENTATION_CLASS_DEFINITIONS] == list(range(7))
    assert len({item["uuid"] for item in SEGMENTATION_CLASS_DEFINITIONS}) == 7
    assert len({item["color"] for item in SEGMENTATION_CLASS_DEFINITIONS}) == 7


def test_unused_legacy_generic_defect_class_upgrades_on_full_load(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (64, 48), (20, 100, 40)).save(source / "image.png")
    project_dir = tmp_path / "project"
    project = create_or_update_project(source, project_dir, "legacy-default")
    project["annotation_mode"] = "segmentation"
    project["segmentation_classes"] = [{
        "uuid": LEGACY_GENERIC_SEGMENTATION_CLASS_UUID,
        "id": 0,
        "export_index": 0,
        "name": "defect",
        "display_name": "Defect",
        "meaning": None,
        "color": "#a77bf3",
    }]
    save_project(project, project_dir)

    loaded = load_project(project_dir)

    assert [item["display_name"] for item in loaded["segmentation_classes"]] == DEFAULT_SEGMENTATION_CLASS_NAMES
    persisted = json.loads((project_dir / "jano-project.json").read_text(encoding="utf-8"))
    assert [item["display_name"] for item in persisted["segmentation_classes"]] == DEFAULT_SEGMENTATION_CLASS_NAMES


def test_used_legacy_generic_defect_class_is_not_automatically_replaced(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (64, 48), (20, 100, 40)).save(source / "image.png")
    project_dir = tmp_path / "project"
    project = create_or_update_project(source, project_dir, "used-legacy-default")
    project["annotation_mode"] = "segmentation"
    project["segmentation_classes"] = [{
        "uuid": LEGACY_GENERIC_SEGMENTATION_CLASS_UUID,
        "id": 0,
        "export_index": 0,
        "name": "defect",
        "display_name": "Defect",
        "meaning": None,
        "color": "#a77bf3",
    }]
    replace_image_annotations(
        project,
        project["images"][0]["id"],
        [{
            "type": "defect_mask",
            "class_uuid": LEGACY_GENERIC_SEGMENTATION_CLASS_UUID,
            "points": [[5, 5], [20, 5], [20, 20], [5, 20]],
            "source": "manual",
        }],
    )
    save_project(project, project_dir)

    loaded = load_project(project_dir)

    assert [item["display_name"] for item in loaded["segmentation_classes"]] == ["Defect"]
    assert len(loaded["annotations"]) == 1


def test_canonical_types_persist_with_stable_uuids(tmp_path: Path) -> None:
    project_dir = create_fixture(tmp_path)
    first = load_project(project_dir)
    second = load_project(project_dir)
    assert [image["id"] for image in first["images"]] == [image["id"] for image in second["images"]]
    assert {item["annotation_kind"] for item in first["annotations"]} == {"fruit_box", "defect_mask", "foreground_mask"}
    assert all(item["uuid"] == item["id"] for item in first["annotations"])
    assert first["detection_classes"] != first["segmentation_classes"]
    assert validate_project(project_dir)["ok"]


def test_project_loading_can_target_one_image_record_for_fast_navigation(tmp_path: Path) -> None:
    project_dir = create_fixture(tmp_path)
    full = load_project(project_dir)
    first_id = full["images"][0]["id"]

    targeted = load_project(project_dir, annotation_image_ids={first_id})
    metadata_only = load_project(project_dir, annotation_image_ids=set())

    assert targeted["annotations"]
    assert {item["image_id"] for item in targeted["annotations"]} == {first_id}
    assert metadata_only["annotations"] == []
    assert metadata_only["images"] == full["images"]


def test_portable_training_export_and_round_trip_geometry(tmp_path: Path) -> None:
    project_dir = create_fixture(tmp_path)
    result = export_dataset(project_dir, tmp_path / "exports", simplification_pixels=0.0)
    root = Path(result["dataset_root"])
    assert root.name == "fruit-project-training-export"
    expected = [
        "export_manifest.json",
        "yolo26_detection/data.yaml",
        "yolo26_segmentation/data.yaml",
        "dfine_detection/dataset_info.yaml",
        "dfine_detection/annotations/instances_train.json",
        "dfine_detection/annotations/instances_val.json",
        "synthetic_assets/assets_manifest.json",
    ]
    assert all((root / path).is_file() for path in expected)
    assert not list(root.glob("*/images/test"))
    manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"]["seed"] == 19
    assert manifest["split"]["enabled"] == ["train", "val"]
    assert manifest["class_mappings"]["dfine_category_id_policy"] == "zero_based_contiguous"
    assert str(tmp_path.resolve()) not in json.dumps(manifest)

    detection_yaml = yaml.safe_load((root / "yolo26_detection" / "data.yaml").read_text(encoding="utf-8"))
    segmentation_yaml = yaml.safe_load((root / "yolo26_segmentation" / "data.yaml").read_text(encoding="utf-8"))
    assert detection_yaml["path"] == segmentation_yaml["path"] == "."
    assert "test" not in detection_yaml and "test" not in segmentation_yaml
    source_splits = {item["image_uuid"]: item["split"] for item in manifest["images"]}
    for task in ("yolo26_detection", "yolo26_segmentation", "dfine_detection"):
        for image_uuid, split in source_splits.items():
            assert list((root / task / "images" / split).glob(f"{image_uuid}.*"))

    detection_rows = [
        line
        for label in (root / "yolo26_detection" / "labels").rglob("*.txt")
        for line in label.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert detection_rows and all(validate_yolo_row(row, 64, 48, task="detect") == [] for row in detection_rows)
    segmentation_rows = [
        line
        for label in (root / "yolo26_segmentation" / "labels").rglob("*.txt")
        for line in label.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert segmentation_rows and all(validate_yolo_row(row, 64, 48) == [] for row in segmentation_rows)
    for coco_path in (root / "dfine_detection" / "annotations").glob("*.json"):
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        assert [item["id"] for item in coco["categories"]] == list(range(len(coco["categories"])))
        assert all(len(item["bbox"]) == 4 and item["area"] > 0 and item["iscrowd"] == 0 for item in coco["annotations"])
    foregrounds = list((root / "synthetic_assets" / "foregrounds").glob("*.png"))
    assert len(foregrounds) == 1
    with Image.open(foregrounds[0]) as asset:
        assert asset.mode == "RGBA" and asset.size == (41, 31)


def test_disconnected_defect_exports_separate_rows(tmp_path: Path) -> None:
    project_dir = create_fixture(tmp_path)
    project = load_project(project_dir)
    defect = next(item for item in project["annotations"] if item["annotation_kind"] == "defect_mask")
    mask_path = project_dir / defect["mask_path"]
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    mask[30:40, 44:56] = 255
    Image.fromarray(mask).save(mask_path)
    result = export_dataset(project_dir, tmp_path / "exports", simplification_pixels=0.0)
    labels = list((Path(result["dataset_root"]) / "yolo26_segmentation" / "labels").rglob(f"{defect['image_id']}*.txt"))
    assert len(labels) == 1
    assert len([line for line in labels[0].read_text(encoding="utf-8").splitlines() if line]) == 2


def test_sam3_failure_is_explicit_and_manual_api_still_works(tmp_path: Path) -> None:
    manager = ProviderManager("sam3")
    assert manager.status()["official_repository"] == SAM3_OFFICIAL_REPOSITORY
    status = manager.load()
    assert status["state"] in {"ready", "failed"}
    if status["state"] == "failed":
        assert status["error"] and "manual" in status["error"].lower()

    project_dir = create_fixture(tmp_path)
    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))
    state = client.get("/api/state").json()
    assert state["opened"] and state["detection_classes"] and state["segmentation_classes"]
    image_id = state["project"]["images"][0]["id"]
    loaded = client.get(f"/api/images/{image_id}").json()
    assert loaded["annotations"]
    assert client.post("/api/validate", json={}).status_code == 200

    cleared = client.put(
        f"/api/images/{image_id}",
        json={
            "expected_revision": loaded["image"]["revision"],
            "image": {
                "status": "unlabelled",
                "reviewed": False,
                "excluded": False,
                "exclusion_reason": "",
                "fruit_id": loaded["image"].get("fruit_id", ""),
                "view_id": loaded["image"].get("view_id", ""),
                "split": loaded["image"].get("split"),
            },
            "annotations": [],
        },
    )
    assert cleared.status_code == 200
    assert cleared.json()["annotations"] == []
    assert cleared.json()["image"]["status"] == "unlabelled"

    closed = client.post("/api/project/close", json={})
    assert closed.status_code == 200 and closed.json() == {"closed": True}
    assert client.get("/api/state").json()["opened"] is False
    assert load_project(project_dir)["annotations"]


def test_image_folder_project_can_start_in_instance_segmentation_mode(tmp_path: Path) -> None:
    source = tmp_path / "segmentation-source"
    source.mkdir()
    Image.new("RGB", (80, 60), (30, 90, 45)).save(source / "frame.png")
    project_dir = tmp_path / "segmentation-project"
    client = TestClient(create_app(provider_name="mock"))

    response = client.post(
        "/api/project/create",
        json={
            "source_dir": str(source),
            "project_dir": str(project_dir),
            "project_name": "Citrus instances",
            "annotation_mode": "segmentation",
            "class_names": ["Healthy", "Citrus canker", "Melanose"],
        },
    )

    assert response.status_code == 200, response.text
    project = response.json()["project"]
    assert project["annotation_mode"] == "segmentation"
    assert [item["display_name"] for item in project["segmentation_classes"]] == [
        "Healthy",
        "Citrus canker",
        "Melanose",
    ]
    assert [item["export_index"] for item in project["segmentation_classes"]] == [0, 1, 2]
    assert len(project["images"]) == 1


def test_image_folder_segmentation_project_uses_default_defect_classes(tmp_path: Path) -> None:
    source = tmp_path / "default-segmentation-source"
    source.mkdir()
    Image.new("RGB", (80, 60), (30, 90, 45)).save(source / "frame.png")
    client = TestClient(create_app(provider_name="mock"))

    response = client.post(
        "/api/project/create",
        json={
            "source_dir": str(source),
            "project_dir": str(tmp_path / "default-segmentation-project"),
            "project_name": "Default citrus defects",
            "annotation_mode": "segmentation",
        },
    )

    assert response.status_code == 200, response.text
    project = response.json()["project"]
    assert [item["display_name"] for item in project["segmentation_classes"]] == DEFAULT_SEGMENTATION_CLASS_NAMES


def test_approved_empty_image_is_supported_negative(tmp_path: Path) -> None:
    project_dir = create_fixture(tmp_path)
    report = validate_project(project_dir)
    assert report["ok"]
    assert any("valid negative" in warning for warning in report["warnings"])


def test_repeated_approved_negative_save_is_idempotent_and_image_get_is_read_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (48, 40), (140, 70, 20)).save(source / "first.png")
    Image.new("RGB", (48, 40), (150, 80, 25)).save(source / "second.png")
    project_dir = tmp_path / "project"
    create_or_update_project(source, project_dir, "negative-save")
    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))
    state = client.get("/api/state").json()
    first_id, second_id = [item["id"] for item in state["project"]["images"]]
    first = client.get(f"/api/images/{first_id}").json()["image"]
    payload = {
        "expected_revision": first["revision"],
        "image": {
            "status": "approved",
            "reviewed": True,
            "excluded": False,
            "exclusion_reason": "",
            "fruit_id": first.get("fruit_id", ""),
            "view_id": first.get("view_id", ""),
            "split": first.get("split"),
            "classification_class_uuid": None,
        },
        "annotations": [],
    }

    saved = client.put(f"/api/images/{first_id}", json=payload)
    assert saved.status_code == 200
    assert saved.json()["image"]["revision"] == 1
    repeated = client.put(f"/api/images/{first_id}", json=payload)
    assert repeated.status_code == 200
    assert repeated.json()["image"]["revision"] == 1
    assert repeated.json()["image"]["status"] == "approved"

    manifest_path = project_dir / "jano-project.json"
    before_get = manifest_path.read_bytes()
    assert client.get(f"/api/images/{second_id}").status_code == 200
    assert manifest_path.read_bytes() == before_get


def test_bulk_no_instances_is_consistent_and_conflict_safe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        Image.new("RGB", (48, 40), (140 + index, 70, 20)).save(source / f"image_{index}.png")
    project_dir = tmp_path / "project"
    project = create_or_update_project(source, project_dir, "bulk-negative")
    first = project["images"][0]
    defect_class = project["segmentation_classes"][0]
    replace_image_annotations(
        project,
        first["id"],
        [{
            "type": "defect_mask",
            "class_uuid": defect_class["uuid"],
            "points": [[5, 5], [20, 5], [20, 20], [5, 20]],
            "source": "manual",
        }],
        {"status": "draft"},
        first["revision"],
    )
    save_project(project, project_dir)

    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))
    state_images = client.get("/api/images").json()["images"]
    first, second, third = state_images
    response = client.post(
        "/api/images/bulk-no-instances",
        json={
            "image_ids": [first["id"], second["id"]],
            "expected_revisions": {
                first["id"]: first["revision"],
                second["id"]: second["revision"],
            },
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["selected_count"] == 2
    assert response.json()["changed_count"] == 2
    saved = load_project(project_dir)
    assert all(image["status"] == "approved" for image in saved["images"][:2])
    assert not [item for item in saved["annotations"] if item["image_id"] in {first["id"], second["id"]}]

    conflict = client.post(
        "/api/images/bulk-no-instances",
        json={
            "image_ids": [first["id"], third["id"]],
            "expected_revisions": {first["id"]: 0, third["id"]: third["revision"]},
        },
    )
    assert conflict.status_code == 409
    unchanged_third = next(image for image in load_project(project_dir)["images"] if image["id"] == third["id"])
    assert unchanged_third["status"] == "unlabelled"
    assert unchanged_third["revision"] == third["revision"]


def test_source_image_edit_rotates_pixels_boxes_and_masks_with_a_backup(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_image = Image.new("RGB", (4, 3), (0, 0, 0))
    source_image.putpixel((0, 0), (255, 0, 0))
    source_image.save(source / "frame.png")
    project_dir = tmp_path / "project"
    project = create_or_update_project(source, project_dir, "image-edit")
    image = project["images"][0]
    detection_class = project["detection_classes"][0]
    segmentation_class = project["segmentation_classes"][0]
    replace_image_annotations(
        project,
        image["id"],
        [
            {
                "type": "fruit_box",
                "class_uuid": detection_class["uuid"],
                "bbox": [0.0, 0.0, 2.0, 1.0],
                "source": "manual",
            },
            {
                "type": "defect_mask",
                "class_uuid": segmentation_class["uuid"],
                "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "source": "manual",
            },
        ],
        {"status": "draft"},
        image["revision"],
    )
    save_project(project, project_dir)
    saved = load_project(project_dir)
    saved_image = saved["images"][0]
    mask_instance = next(item for item in saved["annotations"] if item["annotation_kind"] == "defect_mask")
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[0, 0:2] = 255
    Image.fromarray(mask).save(project_dir / mask_instance["mask_path"])

    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))
    response = client.post(
        f"/api/images/{saved_image['id']}/edit",
        json={"operation": "rotate_right", "expected_revision": saved_image["revision"]},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert (payload["image"]["width"], payload["image"]["height"]) == (3, 4)
    assert payload["image"]["revision"] == saved_image["revision"] + 1
    box = next(item for item in payload["annotations"] if item["annotation_kind"] == "fruit_box")
    assert box["bbox"] == [2.0, 0.0, 1.0, 2.0]

    with Image.open(source / "frame.png") as edited:
        assert edited.size == (3, 4)
        assert edited.convert("RGB").getpixel((2, 0)) == (255, 0, 0)
    edited_mask = next(item for item in payload["annotations"] if item["annotation_kind"] == "defect_mask")
    with Image.open(project_dir / edited_mask["mask_path"]) as opened_mask:
        pixels = np.asarray(opened_mask)
        assert pixels.shape == (4, 3)
        assert pixels[0, 2] == 255 and pixels[1, 2] == 255
        assert int(np.count_nonzero(pixels)) == 2

    backup = project_dir / payload["backup_path"]
    assert backup.is_file()
    with Image.open(backup) as original:
        assert original.size == (4, 3)
        assert original.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
    persisted = load_project(project_dir)
    assert persisted["images"][0]["image_edit_history"][-1]["operation"] == "rotate_right"

    conflict = client.post(
        f"/api/images/{saved_image['id']}/edit",
        json={"operation": "flip_horizontal", "expected_revision": saved_image["revision"]},
    )
    assert conflict.status_code == 409


def test_source_pixel_eraser_writes_transparency_without_changing_annotations(tmp_path: Path) -> None:
    project_dir = create_fixture(tmp_path)
    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))
    loaded = client.get("/api/images").json()["images"][0]
    before = client.get(f"/api/images/{loaded['id']}").json()
    project = load_project(project_dir)
    source_path = Path(project["source_root"]) / loaded["relative_path"]
    mask_files = {
        item["id"]: (project_dir / item["mask_path"]).read_bytes()
        for item in project["annotations"]
        if item["image_id"] == loaded["id"] and item.get("mask_path")
    }
    erase_mask = np.zeros((loaded["height"], loaded["width"]), dtype=np.uint8)
    erase_mask[10:12, 20:22] = 255
    encoded_mask = io.BytesIO()
    Image.fromarray(erase_mask).save(encoded_mask, format="PNG")

    response = client.post(
        f"/api/images/{loaded['id']}/erase-pixels",
        json={
            "erase_mask_data": "data:image/png;base64," + base64.b64encode(encoded_mask.getvalue()).decode("ascii"),
            "expected_revision": before["image"]["revision"],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["erased_pixel_count"] == 4
    assert payload["image"]["revision"] == before["image"]["revision"] + 1
    assert payload["image"]["status"] == before["image"]["status"]
    assert [item.get("bbox") for item in payload["annotations"]] == [
        item.get("bbox") for item in before["annotations"]
    ]
    with Image.open(source_path) as edited:
        rgba = edited.convert("RGBA")
        assert rgba.getpixel((20, 10))[3] == 0
        assert rgba.getpixel((21, 11))[3] == 0
        assert rgba.getpixel((19, 10))[3] == 255
    assert all((project_dir / next(
        item["mask_path"] for item in load_project(project_dir)["annotations"] if item["id"] == instance_id
    )).read_bytes() == original for instance_id, original in mask_files.items())
    backup = project_dir / payload["backup_path"]
    with Image.open(backup) as original:
        assert original.convert("RGBA").getpixel((20, 10))[3] == 255
    assert load_project(project_dir)["images"][0]["image_edit_history"][-1]["operation"] == "erase_pixels"


def test_source_pixel_eraser_converts_jpeg_to_png_for_transparency(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    jpeg_path = source / "frame.jpg"
    Image.new("RGB", (12, 8), (220, 90, 30)).save(jpeg_path, quality=95)
    project_dir = tmp_path / "project"
    project = create_or_update_project(source, project_dir, "jpeg-erase")
    image = project["images"][0]
    erase_mask = Image.new("L", (12, 8), 0)
    erase_mask.putpixel((3, 2), 255)
    encoded_mask = io.BytesIO()
    erase_mask.save(encoded_mask, format="PNG")
    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))

    response = client.post(
        f"/api/images/{image['id']}/erase-pixels",
        json={
            "erase_mask_data": base64.b64encode(encoded_mask.getvalue()).decode("ascii"),
            "expected_revision": image["revision"],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["converted_to_png"] is True
    assert payload["image"]["relative_path"] == "frame.png"
    assert not jpeg_path.exists()
    with Image.open(source / "frame.png") as edited:
        assert edited.mode == "RGBA"
        assert edited.getpixel((3, 2))[3] == 0
        assert edited.getpixel((2, 2))[3] == 255
    assert (project_dir / payload["backup_path"]).suffix.lower() == ".jpg"


def test_project_storage_images_are_never_imported_and_old_entries_are_repaired(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    source_class = root / "Klas 1"
    thumbnail_dir = root / "thumbnails"
    source_class.mkdir(parents=True)
    thumbnail_dir.mkdir()
    Image.new("RGB", (48, 40), (140, 70, 20)).save(source_class / "klas1_001_1.png")
    Image.new("RGB", (20, 20), (20, 20, 20)).save(thumbnail_dir / "generated.jpg")

    project = create_or_update_project(root, root, "same-root")
    assert [image["relative_path"] for image in project["images"]] == ["Klas 1/klas1_001_1.png"]

    polluted = dict(project["images"][0])
    polluted.update({"id": "old-thumbnail-entry", "relative_path": "thumbnails/generated.jpg", "file_name": "generated.jpg"})
    project["images"].append(polluted)
    save_project(project, root)
    repaired = create_or_update_project(root, root, "same-root")
    assert [image["relative_path"] for image in repaired["images"]] == ["Klas 1/klas1_001_1.png"]
    assert repaired["import_summary"]["removed_project_artifacts"] == 1


def test_split_is_stratified_per_source_class_and_keeps_fruit_views_together(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for class_index in range(1, 4):
        class_dir = source / f"Klas {class_index}"
        class_dir.mkdir(parents=True)
        for fruit_index in range(1, 11):
            for view_index in range(1, 3):
                Image.new("RGB", (20, 16), (class_index * 30, fruit_index, view_index)).save(
                    class_dir / f"klas{class_index}_{fruit_index:03d}_{view_index}.png"
                )
    project_dir = tmp_path / "project"
    create_or_update_project(source, project_dir, "stratified-fruit-split")
    manifest = create_split_manifest(
        project_dir,
        {"train": 0.7, "val": 0.2, "test": 0.1},
        seed=42,
        freeze_test=True,
    )

    assert manifest["stratified_by"] == "source_class"
    assert manifest["test_frozen"] is True
    assert set(manifest["class_counts"]) == {"Klas 1", "Klas 2", "Klas 3"}
    for counts in manifest["class_counts"].values():
        assert {split: counts[split]["images"] for split in ("train", "val", "test")} == {
            "train": 14,
            "val": 4,
            "test": 2,
        }
        assert {split: counts[split]["physical_fruits"] for split in ("train", "val", "test")} == {
            "train": 7,
            "val": 2,
            "test": 1,
        }
    split_by_group: dict[str, set[str]] = {}
    for row in manifest["rows"]:
        split_by_group.setdefault(row["group_id"], set()).add(row["split"])
        assert row["group_kind"] == "inferred_fruit_id"
    assert all(len(splits) == 1 for splits in split_by_group.values())
    assert not list((project_dir / "annotations").glob("*.json"))


def test_per_image_record_repairs_stale_manifest_status_and_revision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (48, 40), (140, 70, 20)).save(source / "image.png")
    project_dir = tmp_path / "project"
    project = create_or_update_project(source, project_dir, "record-authority")
    image = project["images"][0]
    replace_image_annotations(project, image["id"], [], {"status": "approved"}, image["revision"])
    save_project(project, project_dir)

    manifest_path = project_dir / "jano-project.json"
    stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale["images"][0].update({"status": "unlabelled", "reviewed": False, "revision": 0})
    manifest_path.write_text(json.dumps(stale), encoding="utf-8")

    repaired = load_project(project_dir)
    assert repaired["images"][0]["status"] == "approved"
    assert repaired["images"][0]["reviewed"] is True
    assert repaired["images"][0]["revision"] == 1


def test_sidebar_api_uses_authoritative_per_image_status(tmp_path: Path) -> None:
    project_dir = create_fixture(tmp_path)
    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))
    manifest_path = project_dir / "jano-project.json"
    stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale["images"][0].update({"status": "unlabelled", "reviewed": False, "revision": 0})
    manifest_path.write_text(json.dumps(stale), encoding="utf-8")

    response = client.get("/api/images")

    assert response.status_code == 200, response.text
    first = response.json()["images"][0]
    assert first["status"] == "approved"
    assert first["reviewed"] is True
    assert first["revision"] > 0


def test_opening_an_image_persists_it_as_the_resume_position(tmp_path: Path) -> None:
    project_dir = create_fixture(tmp_path)
    project = load_project(project_dir)
    target = project["images"][1]
    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))

    response = client.post(f"/api/images/{target['id']}/activate", json={})

    assert response.status_code == 200, response.text
    reopened = load_project(project_dir)
    assert reopened["last_image_id"] == target["id"]


def test_synthetic_object_count_plan_is_seeded_variable_and_bounded() -> None:
    first = _plan_object_counts(100, 6, 20, random.Random(42))
    second = _plan_object_counts(100, 6, 20, random.Random(42))
    assert first == second
    assert min(first) == 0
    assert max(first) <= 6
    assert len(set(first)) > 2
    assert sum(first) >= 20


def test_synthetic_foregrounds_are_partitioned_80_20_and_seeded() -> None:
    ratios = {"train": 0.8, "val": 0.2, "test": 0.0}
    first, counts = _partition_foregrounds(10, ["train", "val"], ratios, seed=42)
    second, second_counts = _partition_foregrounds(10, ["train", "val"], ratios, seed=42)

    assert first == second
    assert counts == second_counts == {"train": 8, "val": 2}
    assert set(first) == set(range(10))
    assert {index for index, split in first.items() if split == "train"}.isdisjoint(
        index for index, split in first.items() if split == "val"
    )


def test_synthetic_request_allows_up_to_300_objects_per_image() -> None:
    request = SyntheticGenerateRequest(
        foreground_dir="foregrounds",
        background_dir="backgrounds",
        output_dir="output",
        objects_per_image=216,
    )
    assert request.objects_per_image == 216
    assert SyntheticGenerateRequest(
        foreground_dir="foregrounds",
        background_dir="backgrounds",
        output_dir="output",
        objects_per_image=300,
    ).objects_per_image == 300
    with pytest.raises(ValueError):
        SyntheticGenerateRequest(
            foreground_dir="foregrounds",
            background_dir="backgrounds",
            output_dir="output",
            objects_per_image=301,
        )


def test_synthetic_request_allows_foreground_scales_above_100_percent() -> None:
    request = SyntheticGenerateRequest(
        foreground_dir="foregrounds",
        background_dir="backgrounds",
        output_dir="output",
        min_scale_percent=125,
        max_scale_percent=250,
    )
    assert request.min_scale_percent == 125
    assert request.max_scale_percent == 250


def test_synthetic_generation_allows_foreground_scales_above_100_percent(tmp_path: Path) -> None:
    foreground_dir = tmp_path / "foregrounds"
    background_dir = tmp_path / "backgrounds"
    foreground_dir.mkdir()
    background_dir.mkdir()
    Image.new("RGBA", (8, 8), (220, 60, 30, 255)).save(foreground_dir / "fruit.png")
    Image.new("RGB", (64, 64), (30, 40, 50)).save(background_dir / "background.png")

    result = generate_synthetic_project(
        foreground_dir,
        background_dir,
        tmp_path / "generated-scaled",
        objects_per_image=1,
        total_images=2,
        seed=719,
        min_scale=1.25,
        max_scale=1.50,
        max_rotation_degrees=0,
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    placement = next(item for image in manifest["images"] for item in image["placements"])
    assert 1.25 <= placement["scale"] <= 1.50


def test_synthetic_generation_defers_objects_when_an_image_reaches_capacity(tmp_path: Path) -> None:
    foreground_dir = tmp_path / "foregrounds"
    background_dir = tmp_path / "backgrounds"
    foreground_dir.mkdir()
    background_dir.mkdir()
    Image.new("RGBA", (60, 60), (220, 60, 30, 255)).save(foreground_dir / "fruit_a.png")
    Image.new("RGBA", (60, 60), (240, 150, 40, 255)).save(foreground_dir / "fruit_b.png")
    Image.new("RGB", (60, 60), (30, 40, 50)).save(background_dir / "background.png")

    result = generate_synthetic_project(
        foreground_dir,
        background_dir,
        tmp_path / "generated-capacity",
        objects_per_image=3,
        total_images=3,
        seed=719,
        min_scale=1.0,
        max_scale=1.0,
        max_rotation_degrees=0,
        overlap_probability=0,
        max_overlap_fraction=0.20,
        placement_attempts=3,
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert [item["requested_object_count"] for item in manifest["images"]] == [2, 1, 0]
    assert [item["object_count"] for item in manifest["images"]] == [1, 1, 0]
    assert manifest["counts"]["capacity_limited_images"] == 1
    assert manifest["counts"]["capacity_limited_instances"] == 1
    assert manifest["counts"]["placed_instances"] == 2
    assert manifest["object_count_distribution"] == {"0": 1, "1": 2}
    assert manifest["planned_object_count_distribution"] == {"0": 1, "1": 1, "2": 1}
    assert len(manifest["capacity_shortfalls"]) == 1
    assert all(count == 1 for count in manifest["foreground_usage"].values())
    assert "continued on later images" in manifest["warnings"][0]


def test_synthetic_generation_is_complete_non_overlapping_and_reproducible(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    foreground_dir = asset_root / "foregrounds"
    background_dir = asset_root / "backgrounds"
    foreground_dir.mkdir(parents=True)
    background_dir.mkdir(parents=True)

    foreground_paths = []
    for index, color in enumerate(((225, 45, 30), (40, 190, 80))):
        pixels = np.zeros((60, 80, 4), dtype=np.uint8)
        pixels[8:53, 12:69, :3] = color
        pixels[8:53, 12:69, 3] = 255
        path = foreground_dir / f"fruit_{index}.png"
        Image.fromarray(pixels, mode="RGBA").save(path)
        foreground_paths.append(path)
    for index, color in enumerate(((35, 40, 45), (65, 70, 75))):
        Image.new("RGB", (320, 240), color).save(background_dir / f"background_{index}.png")

    category_uuid = "6a493d78-5a2d-46ef-81c7-100cc4212b0a"
    (asset_root / "assets_manifest.json").write_text(json.dumps({
        "schema_version": "3.0.0",
        "foregrounds": [
            {
                "path": f"foregrounds/{path.name}",
                "annotation_uuid": f"source-{index}",
                "category": {
                    "uuid": category_uuid,
                    "name": "fruit",
                    "display_name": "Fruit",
                    "export_index": 0,
                    "color": "#42b883",
                },
            }
            for index, path in enumerate(foreground_paths)
        ],
    }), encoding="utf-8")

    request = {
        "foreground_dir": str(foreground_dir),
        "background_dir": str(background_dir),
        "output_dir": str(tmp_path / "generated-a"),
        "objects_per_image": 2,
        "total_images": 3,
        "seed": 719,
        "min_scale_percent": 10,
        "max_scale_percent": 17,
        "max_rotation_degrees": 360,
        "overlap_probability_percent": 0,
        "max_overlap_percent": 20,
    }
    client = TestClient(create_app(provider_name="mock"))
    started = client.post("/api/synthetic/jobs", json=request)
    assert started.status_code == 200, started.text
    job_id = started.json()["job_id"]
    deadline = time.monotonic() + 15
    statuses = []
    while time.monotonic() < deadline:
        status_response = client.get(f"/api/synthetic/jobs/{job_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        statuses.append(status)
        if status["state"] in {"complete", "failed"}:
            break
        time.sleep(0.01)
    assert statuses[-1]["state"] == "complete", statuses[-1]
    assert statuses[-1]["percent"] == 100
    assert statuses[-1]["completed"] == statuses[-1]["total"] == 3
    first_result = statuses[-1]["result"]
    assert first_result["counts"]["foreground_sources"] == 2
    assert first_result["counts"]["background_sources"] == 2
    assert first_result["counts"]["generated_images"] == 3
    assert 2 <= first_result["counts"]["placed_instances"] <= 6
    assert first_result["counts"]["plain_background_images"] >= 1
    first_root = Path(first_result["dataset_root"])
    first_manifest = json.loads((first_root / "synthetic_generation_manifest.json").read_text(encoding="utf-8"))
    assert not (first_root / "images").exists()
    assert first_manifest["paths"]["images"] == "yolo_detection/images"
    object_counts = [image["object_count"] for image in first_manifest["images"]]
    assert object_counts == [len(image["placements"]) for image in first_manifest["images"]]
    assert all(0 <= count <= 2 for count in object_counts)
    assert 0 in object_counts
    assert len(set(object_counts)) > 1
    assert sum(object_counts) == first_result["counts"]["placed_instances"]
    assert object_counts.count(0) == first_result["counts"]["plain_background_images"]
    plain_record = next(image for image in first_manifest["images"] if image["object_count"] == 0)
    with Image.open(first_root / plain_record["image_relative_path"]) as generated_background:
        with Image.open(background_dir / plain_record["background_relative_path"]) as source_background:
            assert np.array_equal(np.asarray(generated_background.convert("RGB")), np.asarray(source_background.convert("RGB")))
    assert all(count > 0 for count in first_manifest["foreground_usage"].values())
    assert all(count > 0 for count in first_manifest["background_usage"].values())
    assert all(
        placement["overlap_pixels"] == 0
        and 0.10 <= placement["scale"] <= 0.17
        and 0 <= placement["rotation_degrees"] <= 360
        for image in first_manifest["images"]
        for placement in image["placements"]
    )
    assert first_manifest["schema_version"] == "1.5"
    assert first_manifest["settings"]["foreground_split_policy"] == "seeded_disjoint_train_validation_pools"
    assert first_manifest["foreground_split_counts"] == {"train": 1, "val": 1}
    foregrounds_by_split = {
        split: {
            placement["foreground_relative_path"]
            for image in first_manifest["images"]
            if image["split"] == split
            for placement in image["placements"]
        }
        for split in ("train", "val")
    }
    assert foregrounds_by_split["train"].isdisjoint(foregrounds_by_split["val"])
    assert all(
        placement["foreground_split"] == image["split"]
        for image in first_manifest["images"]
        for placement in image["placements"]
    )
    assert first_manifest["settings"]["total_images"] == 3
    assert first_manifest["settings"]["max_objects_per_image"] == 2
    assert first_manifest["settings"]["overlap_probability"] == 0
    assert first_manifest["settings"]["max_overlap_fraction"] == 0.20
    assert first_manifest["counts"]["overlapping_instances"] == 0
    assert first_manifest["counts"]["occluded_instances"] == 0
    assert first_manifest["object_count_distribution"] == {
        str(count): object_counts.count(count) for count in sorted(set(object_counts))
    }
    assert first_manifest["yolo_detection"]["labels"] == sum(object_counts)
    yolo_root = Path(first_result["yolo_detection_dir"])
    detection_yaml = yaml.safe_load(Path(first_result["detection_yaml"]).read_text(encoding="utf-8"))
    assert detection_yaml == {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "fruit"},
    }
    assert len(list((yolo_root / "images" / "train").glob("*.png"))) == 2
    assert len(list((yolo_root / "images" / "val").glob("*.png"))) == 1
    yolo_labels = list((yolo_root / "labels").rglob("*.txt"))
    assert len(yolo_labels) == 3
    assert all(
        validate_yolo_row(row, 320, 240, task="detect") == []
        for label in yolo_labels
        for row in label.read_text(encoding="utf-8").splitlines()
        if row
    )
    assert sum(len(label.read_text(encoding="utf-8").splitlines()) for label in yolo_labels) == sum(object_counts)
    assert any(not label.read_text(encoding="utf-8") for label in yolo_labels)

    opened = client.post("/api/project/open", json={"project_dir": first_result["project_dir"]})
    assert opened.status_code == 200
    first_image_id = opened.json()["project"]["images"][0]["id"]
    detail = client.get(f"/api/images/{first_image_id}").json()
    saved = client.put(f"/api/images/{first_image_id}", json={
        "expected_revision": detail["image"]["revision"],
        "image": {
            "status": detail["image"]["status"],
            "reviewed": detail["image"]["reviewed"],
            "excluded": detail["image"]["excluded"],
            "exclusion_reason": detail["image"]["exclusion_reason"],
            "fruit_id": detail["image"]["fruit_id"],
            "view_id": detail["image"]["view_id"],
            "split": detail["image"]["split"],
        },
        "annotations": [
            {key: value for key, value in annotation.items() if key not in {"mask_url", "image_id", "mask_path"}}
            for annotation in detail["annotations"]
        ],
    })
    assert saved.status_code == 200, saved.text

    project = load_project(first_result["project_dir"])
    assert Path(project["source_root"]) == first_root / "yolo_detection" / "images"
    assert all((Path(project["source_root"]) / image["relative_path"]).is_file() for image in project["images"])
    assert Path(project["split"]["manifest_json"]).is_file()
    assert Path(project["split"]["manifest_csv"]).is_file()
    boxes = [item for item in project["annotations"] if item["annotation_kind"] == "fruit_box"]
    masks = [item for item in project["annotations"] if item["annotation_kind"] == "foreground_mask"]
    assert len(boxes) == len(masks) == sum(object_counts)
    assert {item["class_uuid"] for item in boxes} == {category_uuid}
    assert {item["parent_fruit_uuid"] for item in masks} == {item["uuid"] for item in boxes}
    for image in project["images"]:
        image_masks = [item for item in masks if item["image_id"] == image["id"]]
        image_boxes = {item["uuid"]: item for item in boxes if item["image_id"] == image["id"]}
        occupancy = np.zeros((image["height"], image["width"]), dtype=bool)
        for item in image_masks:
            with Image.open(Path(first_result["project_dir"]) / item["mask_path"]) as opened:
                mask = np.asarray(opened) > 0
            assert not np.any(occupancy & mask)
            occupancy |= mask
            ys, xs = np.where(mask)
            mask_bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
            assert image_boxes[item["parent_fruit_uuid"]]["bbox"] == pytest.approx(mask_bbox)

    progress_events = []
    second_result = generate_synthetic_project(
        foreground_dir,
        background_dir,
        tmp_path / "generated-b",
        objects_per_image=2,
        total_images=3,
        seed=719,
        min_scale=0.10,
        max_scale=0.17,
        max_rotation_degrees=360,
        overlap_probability=0,
        max_overlap_fraction=0.20,
        progress_callback=progress_events.append,
    )
    assert progress_events[0]["stage"] == "preparing" and progress_events[0]["percent"] == 0
    assert [event["completed"] for event in progress_events if event["stage"] == "compositing"] == [1, 2, 3]
    assert progress_events[-1] == {
        "completed": 3,
        "total": 3,
        "percent": 100,
        "stage": "complete",
        "message": "Synthetic dataset is ready.",
    }
    assert [event["percent"] for event in progress_events] == sorted(event["percent"] for event in progress_events)
    second_root = Path(second_result["dataset_root"])
    second_manifest = json.loads((second_root / "synthetic_generation_manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["images"] == second_manifest["images"]
    first_hashes = [sha256_file(path) for path in sorted((first_root / "yolo_detection" / "images").rglob("*.png"))]
    second_hashes = [sha256_file(path) for path in sorted((second_root / "yolo_detection" / "images").rglob("*.png"))]
    assert first_hashes == second_hashes


def test_synthetic_generation_creates_bounded_overlap_with_visible_masks(tmp_path: Path) -> None:
    foreground_dir = tmp_path / "foregrounds"
    background_dir = tmp_path / "backgrounds"
    foreground_dir.mkdir()
    background_dir.mkdir()
    Image.new("RGBA", (60, 60), (230, 70, 30, 255)).save(foreground_dir / "fruit_a.png")
    Image.new("RGBA", (60, 60), (245, 145, 35, 255)).save(foreground_dir / "fruit_b.png")
    Image.new("RGB", (140, 100), (35, 45, 55)).save(background_dir / "background.png")

    result = generate_synthetic_project(
        foreground_dir,
        background_dir,
        tmp_path / "generated-overlap",
        objects_per_image=2,
        total_images=3,
        seed=2026,
        min_scale=1.0,
        max_scale=1.0,
        max_rotation_degrees=0,
        overlap_probability=1.0,
        max_overlap_fraction=0.25,
    )

    root = Path(result["dataset_root"])
    manifest = json.loads((root / "synthetic_generation_manifest.json").read_text(encoding="utf-8"))
    composite = next(image for image in manifest["images"] if image["object_count"] == 2)
    placements = composite["placements"]
    assert manifest["settings"]["overlap_policy"] == "bounded_alpha_overlap_with_visible_masks"
    assert manifest["settings"]["overlap_probability"] == 1.0
    assert manifest["settings"]["max_overlap_fraction"] == 0.25
    assert manifest["counts"]["overlapping_instances"] == 2
    assert manifest["counts"]["occluded_instances"] == 1
    assert sum(placement["overlap_requested"] for placement in placements) == 1
    assert all(0 < placement["overlap_fraction"] <= 0.25 for placement in placements)
    assert all(placement["visible_fraction"] >= 0.75 for placement in placements)
    assert sum(placement["occluded_pixels"] > 0 for placement in placements) == 1

    project = load_project(result["project_dir"])
    project_image = next(image for image in project["images"] if image["file_name"] == composite["file_name"])
    boxes = {
        item["uuid"]: item
        for item in project["annotations"]
        if item["image_id"] == project_image["id"] and item["annotation_kind"] == "fruit_box"
    }
    masks = [
        item
        for item in project["annotations"]
        if item["image_id"] == project_image["id"] and item["annotation_kind"] == "foreground_mask"
    ]
    occupancy = np.zeros((project_image["height"], project_image["width"]), dtype=bool)
    for item in masks:
        with Image.open(Path(result["project_dir"]) / item["mask_path"]) as opened:
            mask = np.asarray(opened) > 0
        assert not np.any(occupancy & mask)
        occupancy |= mask
        ys, xs = np.where(mask)
        visible_bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
        assert boxes[item["parent_fruit_uuid"]]["bbox"] == pytest.approx(visible_bbox)
