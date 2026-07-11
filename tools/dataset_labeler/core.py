from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .geometry import coerce_points, flatten_points, polygon_area, polygon_bbox

SEGMENTATION_DEFINITION = "whole_fruit"
PROJECT_FILENAME = "project.json"
MASTER_COCO_FILENAME = "coco_instances.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CLASS_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": 0,
        "name": "healthy",
        "display_name": "Healthy",
        "source_folders": ["healthy"],
        "color": "#3aa655",
    },
    {
        "id": 1,
        "name": "citrus_canker",
        "display_name": "Citrus canker",
        "source_folders": ["citrus canker", "citrus_canker", "cankers", "canker"],
        "color": "#d94141",
    },
    {
        "id": 2,
        "name": "melanose",
        "display_name": "Melanose",
        "source_folders": ["melanose"],
        "color": "#d58a1f",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_path(project_dir: Path | str) -> Path:
    return Path(project_dir).expanduser().resolve() / PROJECT_FILENAME


def annotations_dir(project_dir: Path | str) -> Path:
    return Path(project_dir).expanduser().resolve()


def source_folder_map() -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for cls in CLASS_DEFINITIONS:
        for folder in cls["source_folders"]:
            mapping[normalize_name(folder)] = cls
    return mapping


def normalize_name(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value.strip().lower())


def class_by_id(class_id: int, project: dict[str, Any] | None = None) -> dict[str, Any]:
    classes = project.get("classes", CLASS_DEFINITIONS) if project else CLASS_DEFINITIONS
    for cls in classes:
        if int(cls["id"]) == int(class_id):
            return cls
    raise KeyError(f"Unknown class id: {class_id}")


def class_id_by_name(name: str, project: dict[str, Any] | None = None) -> int | None:
    classes = project.get("classes", CLASS_DEFINITIONS) if project else CLASS_DEFINITIONS
    normalized = normalize_name(name)
    for cls in classes:
        if normalize_name(cls["name"]) == normalized or normalize_name(cls.get("display_name", "")) == normalized:
            return int(cls["id"])
    for cls in classes:
        if normalized in {normalize_name(s) for s in cls.get("source_folders", [])}:
            return int(cls["id"])
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def average_perceptual_hash(path: Path) -> str:
    with Image.open(path) as im:
        pixels = list(im.convert("L").resize((8, 8)).getdata())
    threshold = sum(pixels) / len(pixels)
    bits = "".join("1" if px >= threshold else "0" for px in pixels)
    return f"{int(bits, 2):016x}"


def iter_source_images(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def detect_source_class(source_root: Path, image_path: Path) -> tuple[int | None, str | None]:
    rel_parts = image_path.relative_to(source_root).parts[:-1]
    folder_lookup = source_folder_map()
    for part in rel_parts:
        cls = folder_lookup.get(normalize_name(part))
        if cls:
            return int(cls["id"]), part
    return None, None


def default_project(project_name: str, source_dir: Path | str, project_dir: Path | str) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": "1.0",
        "project_name": project_name,
        "segmentation_definition": SEGMENTATION_DEFINITION,
        "class_id_policy": "fixed_manual_order",
        "classes": CLASS_DEFINITIONS,
        "source_root": str(Path(source_dir).expanduser().resolve()),
        "project_dir": str(Path(project_dir).expanduser().resolve()),
        "created_at": now,
        "updated_at": now,
        "last_image_id": None,
        "split": {
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "seed": 42,
            "manifest_json": None,
            "manifest_csv": None,
        },
        "images": [],
        "annotations": [],
        "next_ids": {"image": 1, "annotation": 1},
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        if path.exists():
            backup_dir = path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            shutil.copy2(path, backup_dir / f"{path.stem}-{stamp}{path.suffix}.bak")
        os.replace(tmp_name, path)
    finally:
        tmp = Path(tmp_name)
        if tmp.exists():
            tmp.unlink()


def load_project(project_dir: Path | str) -> dict[str, Any]:
    path = project_path(project_dir)
    if not path.exists():
        raise FileNotFoundError(f"Project file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    migrate_project(data, path.parent)
    return data


def save_project(project: dict[str, Any], project_dir: Path | str | None = None) -> None:
    out_dir = Path(project_dir or project.get("project_dir") or ".").expanduser().resolve()
    project["project_dir"] = str(out_dir)
    project["updated_at"] = utc_now()
    atomic_write_json(project_path(out_dir), project)


def migrate_project(project: dict[str, Any], project_dir: Path) -> None:
    project.setdefault("segmentation_definition", SEGMENTATION_DEFINITION)
    project.setdefault("classes", CLASS_DEFINITIONS)
    project.setdefault("annotations", [])
    project.setdefault("images", [])
    project.setdefault("project_dir", str(project_dir.resolve()))
    project.setdefault("next_ids", {})
    next_ids = project["next_ids"]
    next_ids.setdefault("image", max([int(img["id"]) for img in project["images"]] or [0]) + 1)
    next_ids.setdefault("annotation", max([int(ann["id"]) for ann in project["annotations"]] or [0]) + 1)
    project.setdefault(
        "split",
        {"ratios": {"train": 0.70, "val": 0.15, "test": 0.15}, "seed": 42, "manifest_json": None, "manifest_csv": None},
    )


def create_or_update_project(
    source_dir: Path | str,
    project_dir: Path | str,
    project_name: str | None = None,
) -> dict[str, Any]:
    source_root = Path(source_dir).expanduser().resolve()
    out_dir = Path(project_dir).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source image directory does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_root}")

    path = project_path(out_dir)
    if path.exists():
        project = load_project(out_dir)
        project["source_root"] = str(source_root)
    else:
        project = default_project(project_name or source_root.name, source_root, out_dir)

    existing = {img["relative_path"].replace("\\", "/"): img for img in project["images"]}
    imported = 0
    skipped = 0
    warnings: list[str] = []

    for image_path in iter_source_images(source_root):
        rel = image_path.relative_to(source_root).as_posix()
        if rel in existing:
            skipped += 1
            continue
        try:
            width, height = image_size(image_path)
            digest = sha256_file(image_path)
            perceptual_hash = average_perceptual_hash(image_path)
        except Exception as exc:  # pragma: no cover - exact decoder errors differ by platform
            warnings.append(f"Skipped unreadable image {rel}: {exc}")
            continue
        class_id, source_folder = detect_source_class(source_root, image_path)
        image_id = int(project["next_ids"]["image"])
        project["next_ids"]["image"] = image_id + 1
        now = utc_now()
        project["images"].append(
            {
                "id": image_id,
                "relative_path": rel,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "sha256": digest,
                "perceptual_hash": perceptual_hash,
                "source_class_id": class_id,
                "source_class": class_by_id(class_id)["name"] if class_id is not None else None,
                "source_folder": source_folder,
                "status": "unlabelled",
                "reviewed": False,
                "excluded": False,
                "exclusion_reason": "",
                "capture_group": infer_capture_group(rel),
                "tree_id": "",
                "fruit_id": "",
                "session_id": "",
                "imported_at": now,
                "updated_at": now,
                "split": None,
            }
        )
        imported += 1
        if class_id is None:
            warnings.append(f"Image is not inside a recognized class folder: {rel}")

    project["import_summary"] = {"imported": imported, "skipped_existing": skipped, "warnings": warnings}
    save_project(project, out_dir)
    return project


def image_abs_path(project: dict[str, Any], image: dict[str, Any]) -> Path:
    return Path(project["source_root"]).expanduser().resolve() / image["relative_path"]


def annotations_for_image(project: dict[str, Any], image_id: int) -> list[dict[str, Any]]:
    return [ann for ann in project["annotations"] if int(ann["image_id"]) == int(image_id)]


def find_image(project: dict[str, Any], image_id: int) -> dict[str, Any]:
    for image in project["images"]:
        if int(image["id"]) == int(image_id):
            return image
    raise KeyError(f"Unknown image id: {image_id}")


def next_annotation_id(project: dict[str, Any]) -> int:
    annotation_id = int(project["next_ids"]["annotation"])
    project["next_ids"]["annotation"] = annotation_id + 1
    return annotation_id


def normalize_annotation(project: dict[str, Any], image_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    image = find_image(project, image_id)
    points = coerce_points(payload.get("points") or payload.get("polygon") or payload.get("segmentation") or [])
    category_id = int(payload.get("category_id", image.get("source_class_id") or 0))
    now = utc_now()
    annotation_id = int(payload.get("id") or next_annotation_id(project))
    area = polygon_area(points)
    return {
        "id": annotation_id,
        "image_id": int(image_id),
        "category_id": category_id,
        "segmentation": [flatten_points(points)],
        "points": [[float(x), float(y)] for x, y in points],
        "bbox": polygon_bbox(points),
        "area": float(area),
        "iscrowd": int(payload.get("iscrowd", 0)),
        "source": payload.get("source", "manual"),
        "notes": payload.get("notes", ""),
        "created_at": payload.get("created_at", now),
        "updated_at": now,
    }


def replace_image_annotations(
    project: dict[str, Any],
    image_id: int,
    annotations: list[dict[str, Any]],
    image_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image = find_image(project, image_id)
    project["annotations"] = [ann for ann in project["annotations"] if int(ann["image_id"]) != int(image_id)]
    normalized = [normalize_annotation(project, image_id, ann) for ann in annotations]
    project["annotations"].extend(normalized)

    if image_updates:
        for key in (
            "status",
            "reviewed",
            "excluded",
            "exclusion_reason",
            "capture_group",
            "tree_id",
            "fruit_id",
            "session_id",
        ):
            if key in image_updates:
                image[key] = image_updates[key]
    if image.get("excluded"):
        image["status"] = "excluded"
    elif image.get("reviewed"):
        image["status"] = "reviewed"
    elif normalized and image.get("status") == "unlabelled":
        image["status"] = "labelled"
    image["updated_at"] = utc_now()
    project["last_image_id"] = int(image_id)
    return image


def infer_capture_group(relative_path: str) -> str:
    stem = Path(relative_path).stem.lower()
    stem = re.sub(r"(?i)(copy|aug|augment|rot|flip|crop|hsv|rgb)[-_]?\d*$", "", stem)
    stem = re.sub(r"[-_\s]?(img|image|frame|photo)?[-_\s]?\d{1,6}$", "", stem)
    stem = re.sub(r"[-_\s]+", "_", stem).strip("_")
    parent = Path(relative_path).parent.as_posix().lower().replace("/", "_")
    return f"{parent}::{stem or Path(relative_path).stem.lower()}"


def project_summary(project: dict[str, Any]) -> dict[str, Any]:
    annotations_by_image: dict[int, int] = {}
    for ann in project["annotations"]:
        annotations_by_image[int(ann["image_id"])] = annotations_by_image.get(int(ann["image_id"]), 0) + 1
    class_counts = {str(cls["id"]): 0 for cls in project["classes"]}
    for ann in project["annotations"]:
        class_counts[str(int(ann["category_id"]))] = class_counts.get(str(int(ann["category_id"])), 0) + 1
    return {
        "image_count": len(project["images"]),
        "annotation_count": len(project["annotations"]),
        "reviewed_count": sum(1 for img in project["images"] if img.get("reviewed")),
        "excluded_count": sum(1 for img in project["images"] if img.get("excluded")),
        "class_counts": class_counts,
        "unlabelled_count": sum(
            1
            for img in project["images"]
            if not img.get("excluded") and annotations_by_image.get(int(img["id"]), 0) == 0
        ),
    }
