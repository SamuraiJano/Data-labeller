from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .core import MASTER_COCO_FILENAME, annotations_dir, class_by_id, image_abs_path, load_project, save_project
from .geometry import denormalize_polygon, flatten_points, normalize_polygon, unflatten_points, validate_polygon
from .hsv import create_hsv_from_yolo_rgb
from .splitting import ensure_split_manifest
from .validation import validate_project

SPLITS = ("train", "val", "test")


def _categories(project: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": int(cls["id"]), "name": cls["name"], "supercategory": "whole_citrus_fruit"}
        for cls in sorted(project["classes"], key=lambda c: int(c["id"]))
    ]


def annotation_to_coco(ann: dict[str, Any]) -> dict[str, Any]:
    segmentation = ann.get("segmentation")
    if not segmentation and ann.get("points"):
        segmentation = [flatten_points([(float(x), float(y)) for x, y in ann["points"]])]
    return {
        "id": int(ann["id"]),
        "image_id": int(ann["image_id"]),
        "category_id": int(ann["category_id"]),
        "segmentation": segmentation,
        "bbox": [float(v) for v in ann.get("bbox", [0, 0, 0, 0])],
        "area": float(ann.get("area", 0.0)),
        "iscrowd": int(ann.get("iscrowd", 0)),
    }


def export_coco(project_dir: Path | str, output_dir: Path | str | None = None) -> dict[str, Any]:
    project = load_project(project_dir)
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else annotations_dir(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_ids = {int(img["id"]) for img in project["images"] if not img.get("excluded")}
    coco = {
        "info": {
            "description": "Whole-fruit citrus instance segmentation annotations",
            "version": "1.0",
            "segmentation_definition": project["segmentation_definition"],
            "class_id_policy": project.get("class_id_policy", "fixed_manual_order"),
        },
        "licenses": [],
        "images": [
            {
                "id": int(img["id"]),
                "file_name": img["relative_path"].replace("\\", "/"),
                "width": int(img["width"]),
                "height": int(img["height"]),
                "sha256": img.get("sha256"),
                "source_class": img.get("source_class"),
                "source_folder": img.get("source_folder"),
                "capture_group": img.get("capture_group"),
                "reviewed": bool(img.get("reviewed")),
            }
            for img in project["images"]
            if not img.get("excluded")
        ],
        "annotations": [
            annotation_to_coco(ann)
            for ann in project["annotations"]
            if int(ann["image_id"]) in image_ids
        ],
        "categories": _categories(project),
    }
    path = out_dir / MASTER_COCO_FILENAME
    path.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path), "images": len(coco["images"]), "annotations": len(coco["annotations"])}


def _safe_stem(image: dict[str, Any]) -> str:
    rel = Path(image["relative_path"])
    pathish = "_".join(rel.with_suffix("").parts)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pathish)
    return f"img_{int(image['id']):06d}_{safe}"


def yolo_line(ann: dict[str, Any], width: int, height: int) -> str:
    segmentation = ann.get("segmentation", [])
    if not segmentation and ann.get("points"):
        segmentation = [flatten_points([(float(x), float(y)) for x, y in ann["points"]])]
    if not segmentation:
        raise ValueError(f"Annotation {ann['id']} has no polygon segmentation.")
    if len(segmentation) > 1:
        segmentation = [max(segmentation, key=len)]
    points = unflatten_points(segmentation[0])
    errors = validate_polygon(points, width, height)
    if errors:
        raise ValueError(f"Annotation {ann['id']} is invalid for YOLO export: {', '.join(errors)}")
    coords = normalize_polygon(points, width, height)
    return " ".join([str(int(ann["category_id"]))] + [f"{v:.6f}" for v in coords])


def write_dataset_yaml(path: Path, dataset_root: Path, classes: list[dict[str, Any]]) -> None:
    names = {int(cls["id"]): cls["name"] for cls in sorted(classes, key=lambda c: int(c["id"]))}
    data = {
        "path": dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _annotations_by_image(project: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for ann in project["annotations"]:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    return grouped


def export_split_coco(project: dict[str, Any], manifest_rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    split_dir = out_dir / "coco"
    split_dir.mkdir(parents=True, exist_ok=True)
    anns_by_image = _annotations_by_image(project)
    images_by_id = {int(img["id"]): img for img in project["images"]}
    paths: dict[str, str] = {}
    for split in SPLITS:
        split_ids = {int(row["image_id"]) for row in manifest_rows if row["split"] == split}
        coco = {
            "info": {
                "description": f"Whole-fruit citrus instance segmentation {split} split",
                "segmentation_definition": project["segmentation_definition"],
            },
            "licenses": [],
            "images": [
                {
                    "id": int(images_by_id[i]["id"]),
                    "file_name": images_by_id[i]["relative_path"].replace("\\", "/"),
                    "width": int(images_by_id[i]["width"]),
                    "height": int(images_by_id[i]["height"]),
                    "sha256": images_by_id[i].get("sha256"),
                    "source_class": images_by_id[i].get("source_class"),
                    "capture_group": images_by_id[i].get("capture_group"),
                }
                for i in sorted(split_ids)
            ],
            "annotations": [
                annotation_to_coco(ann)
                for image_id in sorted(split_ids)
                for ann in anns_by_image.get(image_id, [])
            ],
            "categories": _categories(project),
        }
        path = split_dir / f"instances_{split}.json"
        path.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
        paths[split] = str(path)
    return paths


def export_yolo(
    project_dir: Path | str,
    output_dir: Path | str,
    include_hsv: bool = True,
    force_split: bool = False,
    allow_invalid: bool = False,
) -> dict[str, Any]:
    project_dir = Path(project_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project = load_project(project_dir)
    report = validate_project(project_dir)
    if report["errors"] and not allow_invalid:
        raise ValueError("Validation failed; fix errors before export:\n" + "\n".join(report["errors"][:20]))

    manifest = ensure_split_manifest(project_dir, force=force_split)
    project = load_project(project_dir)
    images_by_id = {int(img["id"]): img for img in project["images"]}
    anns_by_image = _annotations_by_image(project)
    rgb_root = output_dir / "rgb"
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for split in SPLITS:
        (rgb_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (rgb_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in manifest["rows"]:
        image = images_by_id[int(row["image_id"])]
        split = row["split"]
        src = image_abs_path(project, image)
        exported_stem = _safe_stem(image)
        exported_name = exported_stem + src.suffix.lower()
        dst_image = rgb_root / "images" / split / exported_name
        dst_label = rgb_root / "labels" / split / f"{exported_stem}.txt"
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_image)
        lines = [yolo_line(ann, int(image["width"]), int(image["height"])) for ann in anns_by_image.get(int(image["id"]), [])]
        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        rows.append(
            {
                **row,
                "source_relative_path": image["relative_path"],
                "exported_image": str(dst_image.relative_to(output_dir)).replace("\\", "/"),
                "exported_label": str(dst_label.relative_to(output_dir)).replace("\\", "/"),
                "annotation_count": len(lines),
            }
        )

    write_dataset_yaml(rgb_root / "data.yaml", rgb_root.resolve(), project["classes"])
    write_dataset_yaml(output_dir / "data.yaml", rgb_root.resolve(), project["classes"])
    csv_path = manifest_dir / "export_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["image_id"])
        writer.writeheader()
        writer.writerows(rows)
    json_path = manifest_dir / "export_manifest.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")

    split_coco = export_split_coco(project, manifest["rows"], output_dir)
    hsv_summary = None
    if include_hsv:
        hsv_summary = create_hsv_from_yolo_rgb(rgb_root, output_dir / "hsv", project["classes"], output_dir / "manifests")

    return {
        "rgb_root": str(rgb_root),
        "data_yaml": str(output_dir / "data.yaml"),
        "manifest_csv": str(csv_path),
        "manifest_json": str(json_path),
        "split_coco": split_coco,
        "hsv": hsv_summary,
        "validation_warnings": report["warnings"],
    }


def validate_yolo_row(line: str, width: int, height: int) -> list[str]:
    parts = line.split()
    errors: list[str] = []
    if len(parts) < 7 or len(parts[1:]) % 2 != 0:
        return ["YOLO segmentation row must contain class id and at least three x/y pairs"]
    try:
        coords = [float(v) for v in parts[1:]]
        points = denormalize_polygon(coords, width, height)
    except ValueError as exc:
        return [str(exc)]
    if any(v < 0.0 or v > 1.0 for v in coords):
        errors.append("YOLO coordinates must be normalized to 0..1")
    errors.extend(validate_polygon(points, width, height))
    return errors
