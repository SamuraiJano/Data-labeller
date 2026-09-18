from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

from .core import (
    MASTER_COCO_FILENAME,
    atomic_write_json,
    atomic_write_text,
    class_schema_payload,
    image_abs_path,
    load_project,
    safe_resolve,
    utc_now,
)
from .geometry import denormalize_polygon, validate_polygon
from .mask_ops import load_binary_mask, mask_to_polygon, mask_to_polygons, normalized_yolo_row
from .splitting import create_split_manifest, ensure_split_manifest
from .validation import validate_project

EXPORT_SCHEMA_VERSION = "1.0"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned or "data-labeller-project"


def _classes(project: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return sorted(project.get(key, []), key=lambda item: int(item["export_index"]))


def _classification_folders(classes: list[dict[str, Any]]) -> dict[str, str]:
    """Return safe ImageFolder names whose lexical order preserves class IDs."""
    return {
        str(item["uuid"]): f"{int(item['export_index']):03d}-{_slug(str(item['name']))}"
        for item in classes
    }


def _instances_by_image(project: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in project.get("annotations", []):
        grouped[str(instance["image_id"])].append(instance)
    for values in grouped.values():
        values.sort(key=lambda item: str(item["uuid"]))
    return grouped


def _yaml_payload(names: dict[int, str], enabled_splits: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": ".", "train": "images/train", "val": "images/val"}
    if "test" in enabled_splits:
        payload["test"] = "images/test"
    payload["names"] = names
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _export_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        oriented = ImageOps.exif_transpose(opened)
        oriented.save(destination)


def _image_name(image: dict[str, Any]) -> str:
    suffix = Path(image["relative_path"]).suffix.lower() or ".png"
    return f"{image['id']}{suffix}"


def _bbox_row(class_index: int, bbox: list[float], width: int, height: int, precision: int) -> str:
    x, y, box_width, box_height = [float(value) for value in bbox]
    if box_width <= 0 or box_height <= 0 or x < 0 or y < 0 or x + box_width > width or y + box_height > height:
        raise ValueError(f"Invalid fruit_box bbox {bbox} for {width}x{height} image.")
    values = ((x + box_width / 2) / width, (y + box_height / 2) / height, box_width / width, box_height / height)
    return " ".join([str(class_index), *(f"{value:.{precision}f}" for value in values)])


def _manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted((candidate for candidate in path.rglob("*") if candidate.is_file()), key=lambda value: value.as_posix()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _prepare_export_root(output_dir: Path | str, project_name: str) -> tuple[Path, Path]:
    selected = Path(output_dir).expanduser().resolve()
    final = selected if selected.name.endswith("-training-export") else selected / f"{_slug(project_name)}-training-export"
    if final == final.anchor or len(final.parts) < 3:
        raise ValueError(f"Refusing unsafe export target: {final}")
    staging = final.parent / f".{final.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    return staging, final


def _commit_export(staging: Path, final: Path) -> None:
    backup: Path | None = None
    if final.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = final.parent / f"{final.name}.previous-{stamp}"
        counter = 1
        while backup.exists():
            backup = final.parent / f"{final.name}.previous-{stamp}-{counter}"
            counter += 1
        final.replace(backup)
    try:
        staging.replace(final)
    except Exception:
        if backup and backup.exists() and not final.exists():
            backup.replace(final)
        raise


def export_dataset(
    project_dir: Path | str,
    output_dir: Path | str,
    *,
    minimum_iou: float = 0.98,
    simplification_pixels: float = 0.5,
    precision: int = 6,
    force_split: bool = False,
    allow_warnings: bool = True,
    formats: list[str] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Export one portable training contract for all supported trainers."""
    root = Path(project_dir).expanduser().resolve()
    project = load_project(root)
    report = validate_project(root)
    if report["errors"]:
        raise ValueError("Export blocked by validation errors:\n- " + "\n- ".join(report["errors"]))
    if report["warnings"] and not allow_warnings:
        raise ValueError("Export has validation warnings; review them or explicitly allow warnings.")

    supported_formats = {"yolo26_detection", "yolo26_segmentation", "dfine_detection", "synthetic_assets", "image_classification"}
    if formats is not None:
        requested_formats = set(formats)
    elif project.get("annotation_mode") == "classification":
        requested_formats = {"image_classification"}
    else:
        requested_formats = supported_formats - {"image_classification"}
    unknown_formats = requested_formats - supported_formats
    if unknown_formats:
        raise ValueError(f"Unknown export format(s): {', '.join(sorted(unknown_formats))}")
    if not requested_formats:
        raise ValueError("Select at least one export format.")

    split = create_split_manifest(root) if force_split else ensure_split_manifest(root)
    split_by_image = {str(row["image_id"]): str(row["split"]) for row in split["rows"]}
    images = [image for image in project["images"] if not image.get("excluded") and str(image["id"]) in split_by_image]
    enabled_splits = [name for name in ("train", "val", "test") if any(split_by_image[str(image["id"])] == name for image in images)]
    if "train" not in enabled_splits or "val" not in enabled_splits:
        raise ValueError("Training export requires at least one train image and one validation image.")

    staging, final = _prepare_export_root(output_dir, str(project.get("project_name") or root.name))
    grouped = _instances_by_image(project)
    detection_classes = _classes(project, "detection_classes")
    segmentation_classes = _classes(project, "segmentation_classes")
    classification_classes = _classes(project, "classification_classes")
    detection_names = {int(item["export_index"]): str(item["name"]) for item in detection_classes}
    segmentation_names = {int(item["export_index"]): str(item["name"]) for item in segmentation_classes}
    detection_uuids = {str(item["uuid"]): int(item["export_index"]) for item in detection_classes}
    segmentation_uuids = {str(item["uuid"]): int(item["export_index"]) for item in segmentation_classes}
    classification_folders = _classification_folders(classification_classes)
    classification_names = {int(item["export_index"]): str(item["name"]) for item in classification_classes}
    if "image_classification" in requested_formats:
        for split_name in enabled_splits:
            for folder_name in classification_folders.values():
                (staging / "image_classification" / split_name / folder_name).mkdir(parents=True, exist_ok=True)
    warnings = list(report["warnings"])
    source_records: list[dict[str, Any]] = []
    foreground_asset_records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    coco_by_split: dict[str, dict[str, Any]] = {
        split_name: {
            "info": {"description": "Data Labeller fruit_box detection export", "version": EXPORT_SCHEMA_VERSION},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": index, "name": name, "supercategory": "fruit"} for index, name in detection_names.items()],
        }
        for split_name in enabled_splits
    }

    for image_number, image in enumerate(sorted(images, key=lambda item: str(item["id"])), start=1):
        image_id = str(image["id"])
        split_name = split_by_image[image_id]
        split_counts[split_name] += 1
        file_name = _image_name(image)
        source = image_abs_path(project, image)
        instances = grouped.get(image_id, [])
        fruit_boxes = [item for item in instances if item.get("annotation_kind") == "fruit_box"]
        defects = [item for item in instances if item.get("annotation_kind") == "defect_mask"]
        foregrounds = [item for item in instances if item.get("annotation_kind") == "foreground_mask"]

        destinations = []
        if "yolo26_detection" in requested_formats:
            destinations.append(staging / "yolo26_detection" / "images" / split_name / file_name)
        if "yolo26_segmentation" in requested_formats:
            destinations.append(staging / "yolo26_segmentation" / "images" / split_name / file_name)
        if "dfine_detection" in requested_formats:
            destinations.append(staging / "dfine_detection" / "images" / split_name / file_name)
        if "image_classification" in requested_formats:
            class_uuid = str(image.get("classification_class_uuid") or "")
            folder_name = classification_folders.get(class_uuid)
            if folder_name is None:
                raise ValueError(f"classification image {image_id} has no configured image-level class.")
            destinations.append(staging / "image_classification" / split_name / folder_name / file_name)
            instance_counts["classification_images"] += 1
        for destination in destinations:
            _export_image(source, destination)

        detection_rows: list[str] = []
        for box in fruit_boxes:
            class_uuid = str(box.get("class_uuid") or "")
            if class_uuid not in detection_uuids:
                raise ValueError(f"fruit_box {box['uuid']} has an unknown detection class UUID.")
            class_index = detection_uuids[class_uuid]
            detection_rows.append(_bbox_row(class_index, box["bbox"], int(image["width"]), int(image["height"]), precision))
            instance_counts["fruit_box"] += 1
        if "yolo26_detection" in requested_formats:
            atomic_write_text(staging / "yolo26_detection" / "labels" / split_name / f"{Path(file_name).stem}.txt", "\n".join(detection_rows) + ("\n" if detection_rows else ""))

        segmentation_rows: list[str] = []
        for defect in defects:
            class_uuid = str(defect.get("class_uuid") or "")
            if class_uuid not in segmentation_uuids:
                raise ValueError(f"defect_mask {defect['uuid']} has an unknown segmentation class UUID.")
            components = mask_to_polygons(load_binary_mask(project, defect), simplification_pixels)
            if len(components) > 1:
                warnings.append(f"defect_mask {defect['uuid']} exported as {len(components)} separate YOLO instances")
            for component in components:
                if component["iou"] < minimum_iou:
                    warnings.append(f"defect_mask {defect['uuid']} component polygon IoU {component['iou']:.6f} is below {minimum_iou:.6f}")
                segmentation_rows.append(normalized_yolo_row(segmentation_uuids[class_uuid], component["points"], int(image["width"]), int(image["height"]), precision))
                instance_counts["defect_mask_export_rows"] += 1
            instance_counts["defect_mask"] += 1
        if "yolo26_segmentation" in requested_formats:
            atomic_write_text(staging / "yolo26_segmentation" / "labels" / split_name / f"{Path(file_name).stem}.txt", "\n".join(segmentation_rows) + ("\n" if segmentation_rows else ""))

        if "dfine_detection" in requested_formats:
            coco = coco_by_split[split_name]
            coco_image_id = len(coco["images"]) + 1
            # Official Peterande/D-FINE joins img_folder with file_name. Because
            # img_folder is images/<split>, COCO file_name must remain a basename.
            coco["images"].append({"id": coco_image_id, "file_name": file_name, "width": int(image["width"]), "height": int(image["height"])})
            for box in fruit_boxes:
                class_index = detection_uuids[str(box["class_uuid"])]
                x, y, box_width, box_height = [float(value) for value in box["bbox"]]
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1,
                    "image_id": coco_image_id,
                    "category_id": class_index,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                })

        for foreground in foregrounds if "synthetic_assets" in requested_formats else []:
            mask = load_binary_mask(project, foreground)
            ys, xs = np.where(mask)
            if not len(xs):
                raise ValueError(f"foreground_mask {foreground['uuid']} is empty.")
            left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            with Image.open(source) as opened:
                rgba = ImageOps.exif_transpose(opened).convert("RGBA")
            pixels = np.asarray(rgba).copy()
            pixels[:, :, 3] = np.where(mask, pixels[:, :, 3], 0)
            asset = Image.fromarray(pixels, mode="RGBA").crop((left, top, right, bottom))
            asset_name = f"{foreground['uuid']}.png"
            asset_path = staging / "synthetic_assets" / "foregrounds" / asset_name
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset.save(asset_path)
            parent_uuid = str(foreground.get("parent_fruit_uuid") or foreground.get("parent_fruit_instance_id") or "")
            parent = next((item for item in fruit_boxes if str(item.get("uuid")) == parent_uuid), None)
            if parent is None and len(fruit_boxes) == 1:
                parent = fruit_boxes[0]
            category = next(
                (item for item in detection_classes if parent and str(item["uuid"]) == str(parent.get("class_uuid"))),
                detection_classes[0],
            )
            if parent is None:
                warnings.append(
                    f"foreground_mask {foreground['uuid']} has no parent fruit; synthetic generation will use detection class {category['name']}"
                )
            foreground_asset_records.append({
                "path": f"foregrounds/{asset_name}",
                "annotation_uuid": str(foreground["uuid"]),
                "category": {
                    "uuid": str(category["uuid"]),
                    "name": str(category["name"]),
                    "display_name": str(category["display_name"]),
                    "export_index": int(category["export_index"]),
                    "color": str(category["color"]),
                    "meaning": category.get("meaning"),
                },
                "source": {
                    "image_uuid": image_id,
                    "image_relative_path": str(image["relative_path"]).replace("\\", "/"),
                    "original_bbox": [left, top, right - left, bottom - top],
                    "mask_uuid": str(foreground["uuid"]),
                    "parent_fruit_uuid": str(parent["uuid"]) if parent else None,
                    "provenance": foreground.get("source") or foreground.get("provenance") or "manual",
                },
            })
            instance_counts["foreground_mask"] += 1

        if "synthetic_assets" in requested_formats and (image.get("asset_role") == "background" or image.get("background_asset") is True):
            _export_image(source, staging / "synthetic_assets" / "backgrounds" / file_name)
            instance_counts["background"] += 1

        source_records.append({
            "image_uuid": image_id,
            "source_relative_path": str(image["relative_path"]).replace("\\", "/"),
            "export_file_name": file_name,
            "split": split_name,
            "width": int(image["width"]),
            "height": int(image["height"]),
            "sha256": image.get("sha256"),
            "annotation_uuids": [str(item["uuid"]) for item in instances],
            "status": image.get("status"),
            "source_video_id": image.get("source_video_id"),
            "source_video_name": image.get("source_video_name"),
            "sample_time_ms": image.get("sample_time_ms"),
            "actual_time_ms": image.get("actual_time_ms"),
            "source_frame_index": image.get("source_frame_index"),
            "classification_class_uuid": image.get("classification_class_uuid"),
            "classification_class_name": image.get("classification_class_name"),
            "classification_export_index": image.get("classification_export_index"),
        })

    if "yolo26_detection" in requested_formats:
        _write_yaml(staging / "yolo26_detection" / "data.yaml", _yaml_payload(detection_names, enabled_splits))
    if "yolo26_segmentation" in requested_formats:
        _write_yaml(staging / "yolo26_segmentation" / "data.yaml", _yaml_payload(segmentation_names, enabled_splits))
    if "image_classification" in requested_formats:
        classification_payload: dict[str, Any] = {
            "task": "classify",
            "path": ".",
            "train": "train",
            "val": "val",
            "names": classification_names,
            "class_folders": {
                int(item["export_index"]): classification_folders[str(item["uuid"])]
                for item in classification_classes
            },
        }
        if "test" in enabled_splits:
            classification_payload["test"] = "test"
        _write_yaml(staging / "image_classification" / "dataset.yaml", classification_payload)
    if "dfine_detection" in requested_formats:
        for split_name, coco in coco_by_split.items():
            atomic_write_json(staging / "dfine_detection" / "annotations" / f"instances_{split_name}.json", coco)
        _write_yaml(staging / "dfine_detection" / "dataset_info.yaml", {
            "task": "detection",
            "images": "images",
            "annotations": {name: f"annotations/instances_{name}.json" for name in enabled_splits},
            "num_classes": len(detection_names),
            "categories": detection_names,
            "remap_mscoco_category": False,
            "category_id_policy": "zero_based_contiguous",
        })

    background_files = sorted((staging / "synthetic_assets" / "backgrounds").glob("*")) if (staging / "synthetic_assets" / "backgrounds").exists() else []
    if "synthetic_assets" in requested_formats:
        atomic_write_json(staging / "synthetic_assets" / "assets_manifest.json", {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "foregrounds": foreground_asset_records,
            "backgrounds": [{"path": f"backgrounds/{path.name}"} for path in background_files],
            "counts": {"foregrounds": len(foreground_asset_records), "backgrounds": len(background_files)},
        })
    if "dfine_detection" in requested_formats:
        _write_yaml(staging / "dfine_detection" / "custom_detection.yml", {
            "task": "detection",
            "evaluator": {"type": "CocoEvaluator", "iou_types": ["bbox"]},
            "num_classes": len(detection_names),
            "remap_mscoco_category": False,
            "train_dataloader": {"dataset": {"img_folder": "images/train", "ann_file": "annotations/instances_train.json", "remap_mscoco_category": False}},
            "val_dataloader": {"dataset": {"img_folder": "images/val", "ann_file": "annotations/instances_val.json", "remap_mscoco_category": False}},
        })

    schema = class_schema_payload(project)
    atomic_write_json(staging / "class_schema.json", schema)
    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "project_name": project.get("project_name"),
        "annotation_mode": project.get("annotation_mode", "mixed"),
        "source_project_schema": project.get("schema_version"),
        "split": {"seed": int(split["seed"]), "ratios": split["ratios"], "counts": dict(split_counts), "enabled": enabled_splits},
        "class_mappings": {
            "detection": detection_classes,
            "segmentation": segmentation_classes,
            "classification": classification_classes,
            "dfine_category_id_policy": "zero_based_contiguous",
        },
        "counts": {"images": len(images), **dict(instance_counts)},
        "formats": {
            key: value for key, value in {
                "yolo26_detection": "yolo26_detection/data.yaml",
                "yolo26_segmentation": "yolo26_segmentation/data.yaml",
                "dfine_detection": "dfine_detection/dataset_info.yaml",
                "synthetic_assets": "synthetic_assets/assets_manifest.json",
                "image_classification": "image_classification/dataset.yaml",
            }.items() if key in requested_formats
        },
        "images": source_records,
        "warnings": sorted(set(warnings)),
    }
    atomic_write_json(staging / "export_manifest.json", manifest)
    manifest["export_hash"] = _manifest_hash(staging)
    atomic_write_json(staging / "export_manifest.json", manifest)
    _commit_export(staging, final)
    return {
        "dataset_root": str(final),
        "export_manifest": str(final / "export_manifest.json"),
        "data_yaml": str(final / "yolo26_segmentation" / "data.yaml") if "yolo26_segmentation" in requested_formats else None,
        "detection_yaml": str(final / "yolo26_detection" / "data.yaml") if "yolo26_detection" in requested_formats else None,
        "dfine_dataset_info": str(final / "dfine_detection" / "dataset_info.yaml") if "dfine_detection" in requested_formats else None,
        "classification_dir": str(final / "image_classification") if "image_classification" in requested_formats else None,
        "warning_count": len(manifest["warnings"]),
        "warnings": manifest["warnings"],
    }


def export_yolo(project_dir: Path | str, output_dir: Path | str, **kwargs: Any) -> dict[str, Any]:
    return export_dataset(project_dir, output_dir, **kwargs)


def export_coco(project_dir: Path | str, output_dir: Path | str | None = None) -> dict[str, Any]:
    target = output_dir or (Path(project_dir).expanduser().resolve() / "exports")
    summary = export_dataset(project_dir, target)
    root = Path(summary["dataset_root"])
    paths = sorted((root / "dfine_detection" / "annotations").glob("instances_*.json"))
    combined = {"info": {"description": "Data Labeller detection export"}, "licenses": [], "images": [], "annotations": [], "categories": []}
    image_offset = 0
    annotation_id = 1
    for path in paths:
        split_name = path.stem.removeprefix("instances_")
        payload = json.loads(path.read_text(encoding="utf-8"))
        combined["categories"] = payload["categories"]
        id_map = {int(item["id"]): image_offset + index for index, item in enumerate(payload["images"], start=1)}
        for item in payload["images"]:
            combined["images"].append({**item, "id": id_map[int(item["id"])], "file_name": f"{split_name}/{Path(item['file_name']).name}"})
        for item in payload["annotations"]:
            combined["annotations"].append({**item, "id": annotation_id, "image_id": id_map[int(item["image_id"])]})
            annotation_id += 1
        image_offset += len(payload["images"])
    destination = root / MASTER_COCO_FILENAME
    atomic_write_json(destination, combined)
    return {"path": str(destination), "images": len(combined["images"]), "annotations": len(combined["annotations"])}


def yolo_line(instance: dict[str, Any], width: int, height: int, project: dict[str, Any] | None = None) -> str:
    if instance.get("annotation_kind") == "fruit_box" or instance.get("bbox"):
        return _bbox_row(int(instance.get("export_index", instance.get("training_class_id", 0))), list(instance["bbox"]), width, height, 6)
    if project is None:
        points = instance.get("points")
        if not points:
            flat = (instance.get("segmentation") or [[]])[0]
            points = list(zip(flat[::2], flat[1::2], strict=False))
        return normalized_yolo_row(int(instance.get("export_index", instance.get("training_class_id", instance.get("category_id")))), points, width, height)
    polygon = mask_to_polygon(load_binary_mask(project, instance))
    return normalized_yolo_row(int(instance.get("export_index", instance["training_class_id"])), polygon["points"], width, height)


def validate_yolo_row(line: str, width: int, height: int, task: str = "segment") -> list[str]:
    parts = line.split()
    expected = 5 if task == "detect" else None
    if expected and len(parts) != expected:
        return ["YOLO detection row must contain class_id x_center y_center width height"]
    if not expected and (len(parts) < 7 or len(parts[1:]) % 2):
        return ["YOLO segmentation row must contain a class ID and at least three x/y pairs"]
    try:
        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]
    except ValueError:
        return ["YOLO row contains a non-numeric class ID or coordinate"]
    errors: list[str] = []
    if class_id < 0:
        errors.append("YOLO class ID cannot be negative")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coordinates):
        errors.append("YOLO coordinates must be finite and normalized to 0..1")
    if task == "segment" and not errors:
        errors.extend(validate_polygon(denormalize_polygon(coordinates, width, height), width, height))
    if task == "detect" and len(coordinates) == 4 and (coordinates[2] <= 0 or coordinates[3] <= 0):
        errors.append("YOLO detection width and height must be positive")
    return errors
