from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .core import annotations_for_image, class_by_id, image_abs_path, load_project
from .geometry import coerce_points, validate_polygon


def validate_project(project_dir: Path | str) -> dict[str, Any]:
    project = load_project(project_dir)
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "images": len(project["images"]),
        "annotations": len(project["annotations"]),
        "classes": {},
        "splits": {},
        "duplicates": {},
        "mask_areas": [],
    }
    image_ids = {int(img["id"]) for img in project["images"]}
    class_ids = {int(cls["id"]) for cls in project["classes"]}
    sha_to_images: dict[str, list[str]] = defaultdict(list)
    phash_to_images: dict[str, list[str]] = defaultdict(list)
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for ann in project["annotations"]:
        anns_by_image[int(ann.get("image_id", -1))].append(ann)

    for image in project["images"]:
        image_id = int(image["id"])
        rel = image["relative_path"]
        path = image_abs_path(project, image)
        if not path.exists():
            errors.append(f"image {image_id} missing from disk: {rel}")
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            errors.append(f"image {image_id} has unsupported format: {rel}")
        try:
            with Image.open(path) as im:
                width, height = int(im.width), int(im.height)
                im.verify()
            if width != int(image["width"]) or height != int(image["height"]):
                errors.append(f"image {image_id} dimension mismatch: project={image['width']}x{image['height']} file={width}x{height}")
        except Exception as exc:
            errors.append(f"image {image_id} unreadable or corrupt: {rel}: {exc}")
        digest = image.get("sha256")
        if digest:
            sha_to_images[digest].append(rel)
        perceptual_hash = image.get("perceptual_hash")
        if perceptual_hash:
            phash_to_images[perceptual_hash].append(rel)
        if image.get("split"):
            stats["splits"].setdefault(image["split"], 0)
            stats["splits"][image["split"]] += 1
        if image.get("excluded"):
            if not str(image.get("exclusion_reason", "")).strip():
                warnings.append(f"image {image_id} is excluded without an exclusion reason")
            continue
        img_anns = anns_by_image.get(image_id, [])
        if project.get("segmentation_definition") == "whole_fruit" and not img_anns:
            errors.append(f"image {image_id} has no whole-fruit mask: {rel}")
        if not image.get("reviewed"):
            warnings.append(f"image {image_id} has not been reviewed")

    for digest, rels in sha_to_images.items():
        if len(rels) > 1:
            stats["duplicates"][digest] = rels
            warnings.append(f"exact duplicate SHA-256 image group contains {len(rels)} files: {rels[:3]}")
    stats["perceptual_duplicates"] = {}
    for digest, rels in phash_to_images.items():
        if len(rels) > 1:
            stats["perceptual_duplicates"][digest] = rels
            warnings.append(f"perceptual duplicate candidate group contains {len(rels)} files: {rels[:3]}")

    for ann in project["annotations"]:
        ann_id = int(ann.get("id", -1))
        image_id = int(ann.get("image_id", -1))
        if image_id not in image_ids:
            errors.append(f"annotation {ann_id} is orphaned; unknown image_id {image_id}")
            continue
        image = next(img for img in project["images"] if int(img["id"]) == image_id)
        category_id = int(ann.get("category_id", -1))
        if category_id not in class_ids:
            errors.append(f"annotation {ann_id} uses unknown class id {category_id}")
        elif image.get("source_class_id") is not None and category_id != int(image["source_class_id"]):
            expected = class_by_id(int(image["source_class_id"]), project)["name"]
            actual = class_by_id(category_id, project)["name"]
            warnings.append(f"annotation {ann_id} class {actual} differs from source folder class {expected}")
        points = coerce_points(ann.get("points") or (ann.get("segmentation") or [[]])[0])
        poly_errors = validate_polygon(points, int(image["width"]), int(image["height"]))
        errors.extend(f"annotation {ann_id} {err}" for err in poly_errors)
        stats["mask_areas"].append(float(ann.get("area", 0.0)))
        stats["classes"].setdefault(str(category_id), 0)
        stats["classes"][str(category_id)] += 1

    split_by_sha: dict[str, set[str]] = defaultdict(set)
    for image in project["images"]:
        if image.get("sha256") and image.get("split"):
            split_by_sha[image["sha256"]].add(image["split"])
    for digest, splits in split_by_sha.items():
        if len(splits) > 1:
            errors.append(f"exact duplicate image hash leaks across splits: {digest} -> {sorted(splits)}")

    manifest_path = project.get("split", {}).get("manifest_json")
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest_ids = {int(row["image_id"]) for row in manifest.get("rows", [])}
        missing = {int(img["id"]) for img in project["images"] if not img.get("excluded")} - manifest_ids
        if missing:
            errors.append(f"split manifest is missing {len(missing)} non-excluded image(s)")
    else:
        warnings.append("split manifest has not been created yet")

    per_source = Counter(img.get("source_class") or "unknown" for img in project["images"] if not img.get("excluded"))
    for class_name, count in sorted(per_source.items()):
        if count < 3:
            warnings.append(f"class {class_name} has only {count} image(s); train/val/test statistics will be weak")

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }
    return report


def write_validation_report(project_dir: Path | str, output_dir: Path | str | None = None) -> dict[str, Any]:
    project_dir = Path(project_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else project_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = validate_project(project_dir)
    json_path = output_dir / "validation_report.json"
    txt_path = output_dir / "validation_summary.txt"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "Segmentation Labeler Validation",
        f"OK: {report['ok']}",
        f"Errors: {len(report['errors'])}",
        f"Warnings: {len(report['warnings'])}",
        "",
        "Errors:",
        *[f"- {msg}" for msg in report["errors"]],
        "",
        "Warnings:",
        *[f"- {msg}" for msg in report["warnings"]],
        "",
        "Stats:",
        json.dumps(report["stats"], indent=2),
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": report, "json": str(json_path), "summary": str(txt_path)}
