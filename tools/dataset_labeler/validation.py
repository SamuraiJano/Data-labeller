from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .core import ANNOTATION_KINDS, IMAGE_EXTENSIONS, IMAGE_STATUSES, atomic_write_json, atomic_write_text, image_abs_path, load_project, save_project
from .mask_ops import load_binary_mask, mask_to_polygon


def validate_project(project_dir: Path | str) -> dict[str, Any]:
    project = load_project(project_dir)
    errors: list[str] = []
    warnings: list[str] = []
    images_by_id = {str(image["id"]): image for image in project["images"]}
    instances_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in project.get("annotations", []):
        instances_by_image[str(instance.get("image_id"))].append(instance)
    class_counts: Counter[int] = Counter()
    status_counts: Counter[str] = Counter()
    sha_groups: dict[str, list[str]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)
    instance_ids: set[str] = set()
    annotation_mode = str(project.get("annotation_mode") or "mixed")
    classification_classes = {
        str(item["uuid"]): item for item in project.get("classification_classes", [])
    }
    classification_counts: Counter[int] = Counter()
    manifest_path = project.get("split", {}).get("manifest_json")
    has_split_manifest = bool(manifest_path and Path(manifest_path).exists())

    for image in project["images"]:
        image_id = str(image["id"])
        relative = image["relative_path"]
        status = str(image.get("status", "unlabelled"))
        status_counts[status] += 1
        if status not in IMAGE_STATUSES:
            errors.append(f"image {image_id} has invalid status: {status}")
        try:
            path = image_abs_path(project, image)
        except ValueError as exc:
            errors.append(f"image {image_id} has unsafe relative path: {exc}")
            continue
        if not path.exists():
            errors.append(f"image {image_id} is missing from disk: {relative}")
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            errors.append(f"image {image_id} has unsupported extension: {relative}")
        try:
            with Image.open(path) as opened:
                oriented = ImageOps.exif_transpose(opened)
                dimensions = (int(oriented.width), int(oriented.height))
            if dimensions != (int(image["width"]), int(image["height"])):
                errors.append(
                    f"image {image_id} dimension mismatch: metadata={image['width']}x{image['height']} file={dimensions[0]}x{dimensions[1]}"
                )
        except Exception as exc:
            errors.append(f"image {image_id} is unreadable: {relative}: {exc}")
        if image.get("sha256"):
            sha_groups[str(image["sha256"])].append(relative)
        group_id = str(image.get("source_video_id") or image.get("capture_group") or image.get("fruit_id") or "").strip()
        if image.get("split"):
            if group_id:
                group_splits[group_id].add(str(image["split"]))
            elif not has_split_manifest:
                warnings.append(f"image {image_id} is assigned to {image['split']} without grouping metadata")
        if image.get("excluded"):
            if not str(image.get("exclusion_reason") or "").strip():
                warnings.append(f"image {image_id} is excluded without a reason")
            continue
        instances = instances_by_image.get(image_id, [])
        if annotation_mode == "classification":
            class_uuid = str(image.get("classification_class_uuid") or "")
            class_item = classification_classes.get(class_uuid)
            if not class_uuid:
                errors.append(f"classification image {image_id} has no image-level class: {relative}")
            elif class_item is None:
                errors.append(f"classification image {image_id} references unknown class UUID {class_uuid}")
            elif int(image.get("classification_export_index", -1)) != int(class_item["export_index"]):
                errors.append(f"classification image {image_id} has an inconsistent class export index")
            else:
                classification_counts[int(class_item["export_index"])] += 1
            if instances:
                errors.append(f"classification image {image_id} contains instance annotations")
        if annotation_mode != "classification" and status == "approved" and not instances:
            warnings.append(f"approved image {image_id} is a valid negative image with no annotations: {relative}")
        if status == "unlabelled" and instances:
            warnings.append(f"image {image_id} has instances but is marked unlabelled")
        if status != "approved":
            warnings.append(f"image {image_id} is not approved ({status})")

    for checksum, relative_paths in sha_groups.items():
        if len(relative_paths) > 1:
            warnings.append(f"exact duplicate checksum group contains {len(relative_paths)} images: {relative_paths[:3]}")
    for group_id, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"capture group {group_id!r} leaks across splits: {sorted(splits)}")

    for instance in project.get("annotations", []):
        instance_id = str(instance.get("id") or "")
        image_id = str(instance.get("image_id") or "")
        if not instance_id:
            errors.append("an instance is missing its UUID")
            continue
        if instance_id in instance_ids:
            errors.append(f"duplicate instance UUID: {instance_id}")
        instance_ids.add(instance_id)
        if image_id not in images_by_id:
            errors.append(f"instance {instance_id} is orphaned; unknown image_id {image_id}")
            continue
        kind = instance.get("annotation_kind")
        if kind not in ANNOTATION_KINDS:
            errors.append(f"instance {instance_id} has invalid annotation kind: {kind}")
        training_id = instance.get("export_index", instance.get("training_class_id"))
        class_key = "detection_classes" if kind == "fruit_box" else "segmentation_classes"
        configured = project.get(class_key, [])
        if kind != "foreground_mask" and not any(
            str(item.get("uuid")) == str(instance.get("class_uuid"))
            and int(item.get("export_index", -1)) == int(training_id if training_id is not None else -1)
            for item in configured
        ):
            errors.append(f"instance {instance_id} does not map to a configured {class_key.replace('_', ' ')} entry")
        if training_id is not None:
            class_counts[int(training_id)] += 1
        parent = instance.get("parent_fruit_instance_id")
        if parent:
            parent_instance = next((item for item in project.get("annotations", []) if str(item.get("id")) == str(parent)), None)
            if parent_instance is None:
                errors.append(f"defect instance {instance_id} refers to missing parent fruit instance {parent}")
            elif str(parent_instance.get("image_id")) != image_id or parent_instance.get("annotation_kind") != "fruit_box":
                errors.append(f"defect instance {instance_id} parent must be a whole-fruit instance on the same image")
        if kind == "fruit_box":
            bbox = instance.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"fruit_box {instance_id} is missing pixel bbox geometry")
                continue
            x, y, width, height = [float(value) for value in bbox]
            image = images_by_id[image_id]
            if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > int(image["width"]) or y + height > int(image["height"]):
                errors.append(f"fruit_box {instance_id} has invalid or out-of-bounds geometry {bbox}")
            continue
        try:
            mask = load_binary_mask(project, instance)
            image = images_by_id[image_id]
            if mask.shape != (int(image["height"]), int(image["width"])):
                errors.append(
                    f"instance {instance_id} mask dimensions {mask.shape[1]}x{mask.shape[0]} do not match image {image['width']}x{image['height']}"
                )
                continue
            if not mask.any():
                errors.append(f"instance {instance_id} mask is empty")
                continue
            analysis = mask_to_polygon(mask, float(project.get("export_settings", {}).get("polygon_simplification_pixels", 0.5)))
            warnings.extend(f"instance {instance_id}: {message}" for message in analysis["warnings"])
            threshold = float(project.get("export_settings", {}).get("minimum_iou", 0.98))
            if analysis["iou"] < threshold:
                warnings.append(f"instance {instance_id}: polygon round-trip IoU {analysis['iou']:.6f} is below {threshold:.6f}")
        except Exception as exc:
            errors.append(f"instance {instance_id} mask is invalid: {exc}")

    if has_split_manifest:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        rows = manifest.get("rows", [])
        groups_to_split: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            group_id = row.get("group_id") or row.get("fruit_id")
            if group_id:
                groups_to_split[str(group_id)].add(str(row["split"]))
        for group_id, splits in groups_to_split.items():
            if len(splits) > 1:
                errors.append(f"split manifest leaks capture group {group_id!r}: {sorted(splits)}")
    else:
        warnings.append("split manifest has not been created")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "images": len(project["images"]),
            "instances": len(project.get("annotations", [])),
            "status_counts": dict(sorted(status_counts.items())),
            "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
            "classification_class_counts": {
                str(key): value for key, value in sorted(classification_counts.items())
            },
            "physical_groups": len({
                str(image.get("source_video_id") or image.get("capture_group") or image.get("fruit_id"))
                for image in project["images"]
                if image.get("source_video_id") or image.get("capture_group") or image.get("fruit_id")
            }),
            "physical_fruits": len({str(image.get("fruit_id")) for image in project["images"] if image.get("fruit_id")}),
        },
    }


def write_validation_report(project_dir: Path | str, output_dir: Path | str | None = None) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else root / "reports"
    report = validate_project(root)
    project = load_project(root)
    messages = "\n".join([*report["errors"], *report["warnings"]])
    instances_by_image: dict[str, list[str]] = defaultdict(list)
    for instance in project.get("annotations", []):
        instances_by_image[str(instance["image_id"])].append(str(instance["id"]))
    for image in project["images"]:
        image_id = str(image["id"])
        image["has_warning"] = (
            f"image {image_id}" in messages
            or image["relative_path"] in messages
            or any(f"instance {instance_id}" in messages for instance_id in instances_by_image.get(image_id, []))
        )
    save_project(project, root)
    json_path = destination / "validation_report.json"
    text_path = destination / "validation_summary.txt"
    atomic_write_json(json_path, report)
    lines = [
        "Data Labeller Validation",
        f"OK: {report['ok']}",
        f"Errors: {len(report['errors'])}",
        f"Warnings: {len(report['warnings'])}",
        "",
        "Errors:",
        *[f"- {message}" for message in report["errors"]],
        "",
        "Warnings:",
        *[f"- {message}" for message in report["warnings"]],
        "",
        "Stats:",
        json.dumps(report["stats"], indent=2),
    ]
    atomic_write_text(text_path, "\n".join(lines) + "\n")
    return {"report": report, "json": str(json_path), "summary": str(text_path)}
