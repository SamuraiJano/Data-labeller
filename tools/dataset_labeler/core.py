from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Collection
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

SCHEMA_VERSION = "3.2"
SEGMENTATION_DEFINITION = "lossless_per_instance_binary_masks"
PROJECT_FILENAME = "jano-project.json"
LEGACY_PROJECT_FILENAME = "project.json"
MASTER_COCO_FILENAME = "coco_instances.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROJECT_STORAGE_DIRS = {"annotations", "backups", "exports", "manifests", "masks", "thumbnails"}
ANNOTATION_KINDS = {"fruit_box", "defect_mask", "foreground_mask"}
IMAGE_STATUSES = {"unlabelled", "draft", "review", "approved", "excluded"}

# This mapping is intentionally explicit. Displayed grades are one-based while
# both training consumers require zero-based class IDs.
CLASS_DEFINITIONS: list[dict[str, Any]] = [
    {"uuid": "6eeab972-75fd-5df3-9935-d8660067b978", "id": 0, "export_index": 0, "name": "class_1", "display_name": "Class 1", "meaning": None, "color": "#42b883"},
    {"uuid": "43eaaf29-d43b-5f3c-b8ae-64925453e87d", "id": 1, "export_index": 1, "name": "class_2", "display_name": "Class 2", "meaning": None, "color": "#f2b84b"},
    {"uuid": "7c0b37a7-44af-5107-b766-880c8f85cd0d", "id": 2, "export_index": 2, "name": "class_3", "display_name": "Class 3", "meaning": None, "color": "#ef6a6a"},
]

SEGMENTATION_CLASS_DEFINITIONS: list[dict[str, Any]] = [
    {"uuid": "9f11bc3e-d93e-4b5a-8382-ed3b0285dcca", "id": 0, "export_index": 0, "name": "skin_wrinkling", "display_name": "Skin wrinkling", "meaning": None, "color": "#42b883"},
    {"uuid": "997fe368-8f63-4122-8ee0-0935c72080f7", "id": 1, "export_index": 1, "name": "dark_marking", "display_name": "Dark marking", "meaning": None, "color": "#5aa7d8"},
    {"uuid": "23dc870a-ac8c-4ba5-ac7f-60510ddbd267", "id": 2, "export_index": 2, "name": "dark_scarring", "display_name": "Dark scarring", "meaning": None, "color": "#f2b84b"},
    {"uuid": "d92373e3-4f67-4349-9ae7-394b06d5213a", "id": 3, "export_index": 3, "name": "wind_scarring", "display_name": "Wind scarring", "meaning": None, "color": "#ef6a6a"},
    {"uuid": "fd3469ce-35c1-4454-929b-42c38728c4cd", "id": 4, "export_index": 4, "name": "light_brown_discolouration", "display_name": "Light-brown discolouration", "meaning": None, "color": "#a77bf3"},
    {"uuid": "d7042129-4a16-4e69-b3ee-1ee3cfa05966", "id": 5, "export_index": 5, "name": "rough_skin", "display_name": "Rough skin", "meaning": None, "color": "#4ecdc4"},
    {"uuid": "e92e2abd-3b09-4a04-973c-05c9448dc06d", "id": 6, "export_index": 6, "name": "open_skin", "display_name": "Open skin", "meaning": None, "color": "#ff8f70"},
]

DEFAULT_SEGMENTATION_CLASS_NAMES = [item["display_name"] for item in SEGMENTATION_CLASS_DEFINITIONS]
LEGACY_GENERIC_SEGMENTATION_CLASS_UUID = "3486f1d8-1412-5ff0-82c6-f8e2c1a32fbe"

CLASSIFICATION_CLASS_DEFINITIONS: list[dict[str, Any]] = [
    {"uuid": "314a19c5-ea7c-5f86-a59a-755ebbbfca0c", "id": 0, "export_index": 0, "name": "good", "display_name": "Good", "meaning": None, "color": "#42b883"},
    {"uuid": "fa957668-42f5-5b47-a257-08cd375f4388", "id": 1, "export_index": 1, "name": "defective", "display_name": "Defective", "meaning": None, "color": "#ef6a6a"},
]

CANONICAL_CLASS_NAMES = ("class_1", "class_2", "class_3")


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_class_definitions(classes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a portable, contiguous class mapping without conflating task types."""
    if not classes:
        raise ValueError("At least one class is required.")
    display_names = [str(item.get("display_name") or "").strip() for item in classes]
    if any(not value for value in display_names) or len(set(display_names)) != len(display_names):
        raise ValueError("Class display names must be non-empty and unique.")
    names = [str(item.get("name") or "").strip() for item in classes]
    if any(not value for value in names) or len(set(names)) != len(names):
        raise ValueError("Class names must be non-empty and unique.")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(classes):
        raw_uuid = str(item.get("uuid") or uuid.uuid5(uuid.NAMESPACE_URL, f"data-labeller-class:{names[index]}"))
        try:
            class_uuid = str(uuid.UUID(raw_uuid))
        except ValueError as exc:
            raise ValueError(f"Class {names[index]!r} has an invalid UUID.") from exc
        normalized.append(
            {
                "uuid": class_uuid,
                "id": index,
                "export_index": index,
                "name": names[index],
                "display_name": display_names[index],
                "meaning": str(item["meaning"]).strip() if item.get("meaning") else None,
                "color": str(item.get("color") or "#8b95a1"),
            }
        )
    if len({item["uuid"] for item in normalized}) != len(normalized):
        raise ValueError("Class UUIDs must be unique.")
    return normalized


def class_schema_payload(project: dict[str, Any]) -> dict[str, Any]:
    detection = validate_class_definitions(list(project.get("detection_classes") or project.get("classes") or CLASS_DEFINITIONS))
    segmentation = validate_class_definitions(list(project.get("segmentation_classes") or SEGMENTATION_CLASS_DEFINITIONS))
    classification = validate_class_definitions(list(project.get("classification_classes") or CLASSIFICATION_CLASS_DEFINITIONS))
    payload = {
        "schema_version": "2.1",
        "authority": "data-labeller",
        "detection_classes": detection,
        "segmentation_classes": segmentation,
        "classification_classes": classification,
    }
    return {**payload, "class_schema_hash": canonical_json_hash(payload)}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def project_path(project_dir: Path | str) -> Path:
    root = Path(project_dir).expanduser().resolve()
    current = root / PROJECT_FILENAME
    legacy = root / LEGACY_PROJECT_FILENAME
    return current if current.exists() or not legacy.exists() else legacy


def annotations_dir(project_dir: Path | str) -> Path:
    return Path(project_dir).expanduser().resolve() / "annotations"


def masks_dir(project_dir: Path | str) -> Path:
    return Path(project_dir).expanduser().resolve() / "masks"


def annotation_path(project_dir: Path | str, image_id: str | int) -> Path:
    return annotations_dir(project_dir) / f"{image_id}.json"


def mask_path(project_dir: Path | str, image_id: str | int, instance_id: str) -> Path:
    return masks_dir(project_dir) / str(image_id) / f"{instance_id}.png"


def safe_resolve(root: Path | str, relative: Path | str) -> Path:
    root_path = Path(root).expanduser().resolve()
    candidate = (root_path / relative).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Path escapes configured root: {relative}") from exc
    return candidate


def class_by_id(class_id: int, project: dict[str, Any] | None = None) -> dict[str, Any]:
    classes = project.get("detection_classes", project.get("classes", CLASS_DEFINITIONS)) if project else CLASS_DEFINITIONS
    for cls in classes:
        if int(cls["id"]) == int(class_id):
            return cls
    raise KeyError(f"Unknown training class id: {class_id}")


def class_id_by_name(name: str, project: dict[str, Any] | None = None) -> int | None:
    value = name.strip().lower().replace(" ", "_")
    classes = project.get("classes", CLASS_DEFINITIONS) if project else CLASS_DEFINITIONS
    for cls in classes:
        candidates = {str(cls.get("name", "")).lower(), str(cls.get("display_name", "")).lower().replace(" ", "_")}
        if value in candidates:
            return int(cls["id"])
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image)
        return int(oriented.width), int(oriented.height)


def average_perceptual_hash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((8, 8)).getdata())
    threshold = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= threshold else "0" for pixel in pixels)
    return f"{int(bits, 2):016x}"


def iter_source_images(source_root: Path) -> list[Path]:
    return sorted(
        path for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _is_project_storage_image(path: Path, source_root: Path, project_root: Path) -> bool:
    """Keep app-generated files out of the source-image inventory."""
    try:
        relative_to_source = path.relative_to(source_root)
    except ValueError:
        return False
    project_is_inside_source = False
    if project_root != source_root:
        try:
            project_root.relative_to(source_root)
            project_is_inside_source = True
        except ValueError:
            pass
    if project_is_inside_source:
        try:
            path.relative_to(project_root)
            return True
        except ValueError:
            return False
    if project_root != source_root:
        return False
    return bool(relative_to_source.parts and relative_to_source.parts[0].casefold() in PROJECT_STORAGE_DIRS)


def stable_image_id(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").casefold()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"data-labeller-image:{normalized}"))


def default_project(project_name: str, source_dir: Path | str, project_dir: Path | str) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "project_name": project_name,
        "source_root": str(Path(source_dir).expanduser().resolve()),
        "project_dir": str(Path(project_dir).expanduser().resolve()),
        "created_at": now,
        "updated_at": now,
        "last_image_id": None,
        "annotation_mode": "mixed",
        "source_kind": "images",
        "detection_classes": [dict(item) for item in CLASS_DEFINITIONS],
        "segmentation_classes": [dict(item) for item in SEGMENTATION_CLASS_DEFINITIONS],
        "classification_classes": [dict(item) for item in CLASSIFICATION_CLASS_DEFINITIONS],
        "classes": [dict(item) for item in CLASS_DEFINITIONS],
        "export_settings": {
            "polygon_simplification_pixels": 0.5,
            "minimum_iou": 0.98,
            "numeric_precision": 6,
            "path_style": "relative",
            "path_base": "",
        },
        "split": {
            "ratios": {"train": 0.70, "val": 0.20, "test": 0.10},
            "seed": 42,
            "manifest_json": None,
            "manifest_csv": None,
            "test_frozen": False,
            "dataset_version": "0.1.0",
        },
        "images": [],
        "annotations": [],
    }


def _prune_backups(backup_dir: Path, prefix: str, keep: int = 5) -> None:
    matches = sorted(backup_dir.glob(f"{prefix}-*.bak"), key=lambda path: path.stat().st_mtime, reverse=True)
    for stale in matches[keep:]:
        stale.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, payload: bytes, *, backup: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if backup and path.exists():
            lowered = [part.lower() for part in path.parts]
            storage_indices = [index for index, part in enumerate(lowered) if part in {"annotations", "masks"}]
            project_root = Path(*path.parts[: storage_indices[-1]]) if storage_indices else path.parent
            backup_dir = project_root / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            prefix = f"{path.parent.name}-{path.stem}"
            shutil.copy2(path, backup_dir / f"{prefix}-{stamp}.bak")
            _prune_backups(backup_dir, prefix)
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any], *, backup: bool = True) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    atomic_write_bytes(path, encoded, backup=backup)


def atomic_write_text(path: Path, text: str, *, backup: bool = False) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), backup=backup)


def _manifest_payload(project: dict[str, Any]) -> dict[str, Any]:
    omitted = {"annotations", "_dirty_annotation_image_ids", "_legacy_source"}
    return {key: value for key, value in project.items() if key not in omitted}


def _default_annotation_record(image: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "image_id": image["id"],
        "original_relative_image_path": image["relative_path"],
        "width": int(image["width"]),
        "height": int(image["height"]),
        "checksum": image.get("sha256"),
        "fruit_id": image.get("fruit_id", ""),
        "view_id": image.get("view_id", ""),
        "status": image.get("status", "unlabelled"),
        "split": image.get("split"),
        "classification_class_uuid": image.get("classification_class_uuid"),
        "classification_class_name": image.get("classification_class_name"),
        "classification_export_index": image.get("classification_export_index"),
        "revision": int(image.get("revision", 0)),
        "created_at": image.get("imported_at", now),
        "updated_at": image.get("updated_at", now),
        "instances": [],
    }


def _legacy_points(annotation: dict[str, Any]) -> list[list[float]]:
    if annotation.get("points"):
        return [[float(point[0]), float(point[1])] for point in annotation["points"]]
    flat = (annotation.get("segmentation") or [[]])[0]
    return [[float(flat[index]), float(flat[index + 1])] for index in range(0, len(flat) - 1, 2)]


def _migrate_legacy_manifest(data: dict[str, Any], root: Path) -> dict[str, Any]:
    migrated = default_project(data.get("project_name", root.name), data.get("source_root", root), root)
    migrated["created_at"] = data.get("created_at", migrated["created_at"])
    migrated["last_image_id"] = data.get("last_image_id")
    legacy_annotations = data.get("annotations", [])
    legacy_classes = {int(item["id"]): item for item in data.get("classes", []) if "id" in item}
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in legacy_annotations:
        by_image[str(annotation.get("image_id"))].append(annotation)
    for old_image in data.get("images", []):
        image = dict(old_image)
        image["id"] = str(old_image["id"])
        old_status = str(image.get("status", "unlabelled"))
        image["status"] = {"labelled": "draft", "reviewed": "approved"}.get(old_status, old_status)
        if image["status"] not in IMAGE_STATUSES:
            image["status"] = "draft" if by_image.get(str(old_image["id"])) else "unlabelled"
        image.setdefault("fruit_id", "")
        image.setdefault("view_id", Path(image.get("relative_path", "")).stem)
        image.setdefault("has_warning", False)
        image.setdefault("revision", 0)
        migrated["images"].append(image)
        for old_annotation in by_image.get(str(old_image["id"]), []):
            category_id = int(old_annotation.get("category_id", 0))
            legacy_class = legacy_classes.get(category_id, {})
            legacy_label = str(legacy_class.get("display_name") or legacy_class.get("name") or f"Legacy class {category_id}")
            migrated["annotations"].append(
                {
                    "id": str(uuid.uuid4()),
                    "image_id": image["id"],
                    "annotation_kind": "fruit_box",
                    "training_class_id": None,
                    "category_id": None,
                    "display_label": f"Unmapped: {legacy_label}",
                    "name": f"Unmapped: {legacy_label}",
                    "color": str(legacy_class.get("color") or "#8b95a1"),
                    "mask_path": "",
                    "parent_fruit_instance_id": None,
                    "provenance": "manual",
                    "prompt_metadata": {
                        "migrated_from": LEGACY_PROJECT_FILENAME,
                        "legacy_category_id": category_id,
                        "legacy_label": legacy_label,
                        "requires_grade_confirmation": True,
                    },
                    "review_status": "approved" if image["status"] == "approved" else "draft",
                    "visible": True,
                    "locked": False,
                    "notes": old_annotation.get("notes", ""),
                    "created_at": old_annotation.get("created_at", utc_now()),
                    "updated_at": utc_now(),
                    "points": _legacy_points(old_annotation),
                }
            )
    migrated["_dirty_annotation_image_ids"] = [str(image["id"]) for image in migrated["images"]]
    migrated["_legacy_source"] = str(root / LEGACY_PROJECT_FILENAME)
    return migrated


def migrate_project(project: dict[str, Any], project_dir: Path) -> None:
    project.setdefault("schema_version", SCHEMA_VERSION)
    project.setdefault("project_dir", str(project_dir.resolve()))
    project.setdefault("annotation_mode", "mixed")
    project.setdefault("source_kind", "images")
    existing_detection = project.get("detection_classes") or project.get("classes") or CLASS_DEFINITIONS
    project["detection_classes"] = validate_class_definitions(list(existing_detection))
    existing_segmentation = project.get("segmentation_classes")
    if not existing_segmentation:
        defect_labels = [str(value).strip() for value in project.get("defect_labels", []) if str(value).strip()]
        existing_segmentation = [
            {
                "name": value.lower().replace(" ", "_"),
                "display_name": value,
                "color": "#a77bf3",
            }
            for value in defect_labels
        ] or SEGMENTATION_CLASS_DEFINITIONS
    project["segmentation_classes"] = validate_class_definitions(list(existing_segmentation))
    existing_classification = project.get("classification_classes") or CLASSIFICATION_CLASS_DEFINITIONS
    project["classification_classes"] = validate_class_definitions(list(existing_classification))
    # Kept as a read-only compatibility alias for pre-v3 frontends.
    project["classes"] = [dict(item) for item in project["detection_classes"]]
    project.setdefault("export_settings", default_project("", project.get("source_root", project_dir), project_dir)["export_settings"])
    project.setdefault("split", default_project("", project.get("source_root", project_dir), project_dir)["split"])
    project["split"].setdefault("ratios", {"train": 0.70, "val": 0.20, "test": 0.10})
    project.setdefault("images", [])
    for image in project["images"]:
        image["id"] = str(image["id"])
        image.setdefault("status", "unlabelled")
        image.setdefault("fruit_id", "")
        image.setdefault("view_id", Path(image.get("relative_path", "")).stem)
        image.setdefault("has_warning", False)
        image.setdefault("revision", 0)
        image.setdefault("split", None)
        image.setdefault("source_video_id", None)
        image.setdefault("source_video_name", None)
        image.setdefault("sample_time_ms", None)
        image.setdefault("actual_time_ms", None)
        image.setdefault("source_frame_index", None)
        image.setdefault("classification_class_uuid", None)
        image.setdefault("classification_class_name", None)
        image.setdefault("classification_export_index", None)
        if image.get("classification_class_uuid"):
            classification_class = next(
                (
                    item
                    for item in project["classification_classes"]
                    if str(item["uuid"]) == str(image["classification_class_uuid"])
                ),
                None,
            )
            if classification_class:
                image["classification_class_name"] = str(classification_class["name"])
                image["classification_export_index"] = int(classification_class["export_index"])


def _read_instances(project_dir: Path, image: dict[str, Any]) -> list[dict[str, Any]]:
    path = annotation_path(project_dir, image["id"])
    if not path.exists():
        return []
    record = json.loads(path.read_text(encoding="utf-8"))
    # Per-image records are written together with their annotations and are
    # therefore the authority when an interrupted or concurrent manifest write
    # left the project-level image entry stale.
    for field in (
        "fruit_id",
        "view_id",
        "status",
        "split",
        "classification_class_uuid",
        "classification_class_name",
        "classification_export_index",
        "updated_at",
    ):
        if field in record:
            image[field] = record[field]
    image["revision"] = int(record.get("revision", image.get("revision", 0)))
    image["reviewed"] = image.get("status") == "approved"
    if image.get("status") == "excluded":
        image["excluded"] = True
    instances: list[dict[str, Any]] = []
    for stored in record.get("instances", []):
        instance = dict(stored, image_id=str(image["id"]))
        legacy_kind = str(instance.get("annotation_kind") or instance.get("type") or "")
        instance["annotation_kind"] = {
            "fruit_grade": "fruit_box",
            "defect_region": "defect_mask",
        }.get(legacy_kind, legacy_kind)
        instance["type"] = instance["annotation_kind"]
        instance.setdefault("uuid", str(instance.get("id")))
        instance.setdefault("source", instance.get("provenance", "manual"))
        instances.append(instance)
    return instances


def load_project(
    project_dir: Path | str,
    annotation_image_ids: Collection[str | int] | None = None,
) -> dict[str, Any]:
    """Load a project and only the annotation records the caller needs.

    ``None`` preserves the full-project behavior required by validation and
    export.  Interactive image and thumbnail endpoints pass a small set so a
    large dataset does not perform thousands of filesystem probes per click.
    """
    root = Path(project_dir).expanduser().resolve()
    path = project_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Project file does not exist: {root / PROJECT_FILENAME}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if path.name == LEGACY_PROJECT_FILENAME and str(data.get("schema_version")) != SCHEMA_VERSION:
        return _migrate_legacy_manifest(data, root)
    migrate_project(data, root)
    requested_ids = None if annotation_image_ids is None else {str(image_id) for image_id in annotation_image_ids}
    if requested_ids is None:
        record_root = annotations_dir(root)
        available_ids = {record.stem for record in record_root.glob("*.json")} if record_root.is_dir() else set()
    else:
        # The targeted path probes at most the explicitly requested records;
        # an empty set performs no annotation-directory I/O at all.
        available_ids = requested_ids
    data["annotations"] = [
        instance
        for image in data["images"]
        if str(image["id"]) in available_ids
        and (requested_ids is None or str(image["id"]) in requested_ids)
        for instance in _read_instances(root, image)
    ]
    legacy_classes = data.get("segmentation_classes", [])
    unused_legacy_default = (
        requested_ids is None
        and len(legacy_classes) == 1
        and str(legacy_classes[0].get("uuid")) == LEGACY_GENERIC_SEGMENTATION_CLASS_UUID
        and not any(item.get("annotation_kind") == "defect_mask" for item in data["annotations"])
    )
    if unused_legacy_default:
        data["segmentation_classes"] = [dict(item) for item in SEGMENTATION_CLASS_DEFINITIONS]
        save_project(data, root)
    return data


def _decode_mask_data(value: str) -> bytes:
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Mask data is not valid base64 PNG data.") from exc


def _polygon_mask(points: list[list[float]], width: int, height: int) -> bytes:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([(float(x), float(y)) for x, y in points], fill=255)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def validate_mask_bytes(payload: bytes, width: int, height: int) -> bytes:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.size != (width, height):
                raise ValueError(f"Mask dimensions {image.width}x{image.height} do not match image {width}x{height}.")
            if "A" in image.getbands():
                array = np.asarray(image.getchannel("A"), dtype=np.uint8)
            else:
                array = np.asarray(image.convert("L"), dtype=np.uint8)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Mask is not a readable PNG image.") from exc
    binary = Image.fromarray(np.where(array > 0, 255, 0).astype(np.uint8), mode="L")
    output = io.BytesIO()
    binary.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _normalise_instance(project: dict[str, Any], image: dict[str, Any], payload: dict[str, Any]) -> tuple[dict[str, Any], bytes | None]:
    now = utc_now()
    raw_id = str(payload.get("id") or payload.get("uuid") or uuid.uuid4())
    try:
        instance_id = str(uuid.UUID(raw_id))
    except ValueError:
        instance_id = str(uuid.uuid4())
    kind = str(payload.get("type") or payload.get("annotation_kind") or "defect_mask")
    kind = {"fruit_grade": "fruit_box", "defect_region": "defect_mask"}.get(kind, kind)
    if kind not in ANNOTATION_KINDS:
        raise ValueError(f"Unknown annotation kind: {kind}")
    class_key = "detection_classes" if kind == "fruit_box" else "segmentation_classes"
    available_classes = project.get(class_key, []) if kind != "foreground_mask" else []
    training_id = payload.get("export_index", payload.get("training_class_id", payload.get("category_id")))
    training_id = int(training_id) if training_id is not None else None
    class_uuid = payload.get("class_uuid")
    if class_uuid is not None:
        selected = next((item for item in available_classes if str(item["uuid"]) == str(class_uuid)), None)
        if selected is None:
            raise ValueError(f"{kind} references unknown class UUID {class_uuid}.")
        training_id = int(selected["export_index"])
    else:
        selected = next((item for item in available_classes if int(item["export_index"]) == training_id), None)
    if kind != "foreground_mask" and selected is None:
        raise ValueError(f"{kind} requires a configured {class_key.replace('_', ' ')} entry.")
    if selected is not None:
        cls = selected
        display_label = cls["display_name"]
        color = cls["color"]
    else:
        display_label = "Synthetic foreground"
        color = str(payload.get("color") or "#45c2e8")
    bbox = payload.get("bbox")
    if kind == "fruit_box":
        if bbox is None and (payload.get("points") or payload.get("segmentation")):
            points = payload.get("points") or _legacy_points(payload)
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("fruit_box requires pixel bbox geometry [x, y, width, height].")
        x, y, width, height = [float(value) for value in bbox]
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > int(image["width"]) or y + height > int(image["height"]):
            raise ValueError("fruit_box bbox must have positive size and stay within image bounds.")
        bbox = [x, y, width, height]
    relative_mask = f"masks/{image['id']}/{instance_id}.png"
    instance = {
        "id": instance_id,
        "uuid": instance_id,
        "type": kind,
        "annotation_kind": kind,
        "class_uuid": str(selected["uuid"]) if selected else None,
        "class_name": str(selected["name"]) if selected else None,
        "export_index": training_id,
        "training_class_id": training_id,
        "category_id": training_id,
        "display_label": display_label,
        "name": str(payload.get("name") or display_label),
        "color": color,
        "mask_path": relative_mask if kind != "fruit_box" else None,
        "bbox": bbox if kind == "fruit_box" else None,
        "parent_fruit_uuid": payload.get("parent_fruit_uuid") or payload.get("parent_fruit_instance_id") or None,
        "parent_fruit_instance_id": payload.get("parent_fruit_uuid") or payload.get("parent_fruit_instance_id") or None,
        "source": str(payload.get("source") or payload.get("provenance") or "manual"),
        "provenance": str(payload.get("source") or payload.get("provenance") or "manual"),
        "prompt_metadata": payload.get("prompt_metadata") or {},
        "review_status": str(payload.get("review_status") or "draft"),
        "visible": bool(payload.get("visible", True)),
        "locked": bool(payload.get("locked", False)),
        "notes": str(payload.get("notes") or ""),
        "created_at": payload.get("created_at", now),
        "updated_at": now,
    }
    mask_bytes: bytes | None = None
    if kind != "fruit_box" and payload.get("mask_data"):
        mask_bytes = _decode_mask_data(str(payload["mask_data"]))
    elif kind != "fruit_box" and (payload.get("points") or payload.get("segmentation")):
        points = payload.get("points") or _legacy_points(payload)
        mask_bytes = _polygon_mask(points, int(image["width"]), int(image["height"]))
    return instance, mask_bytes


def _write_image_record(project: dict[str, Any], image: dict[str, Any], instances: list[dict[str, Any]]) -> None:
    root = Path(project["project_dir"]).resolve()
    previous_path = annotation_path(root, image["id"])
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else _default_annotation_record(image)
    previous_instances = {str(item["id"]): item for item in previous.get("instances", [])}
    normalised: list[dict[str, Any]] = []
    active_ids: set[str] = set()
    for payload in instances:
        instance, mask_bytes = _normalise_instance(project, image, payload)
        active_ids.add(instance["id"])
        destination = mask_path(root, image["id"], instance["id"])
        if instance["annotation_kind"] == "fruit_box":
            pass
        elif mask_bytes is not None:
            atomic_write_bytes(destination, validate_mask_bytes(mask_bytes, int(image["width"]), int(image["height"])))
        elif not destination.exists():
            old = previous_instances.get(str(payload.get("id")))
            if old:
                old_path = safe_resolve(root, old["mask_path"])
                if old_path.exists() and old_path != destination:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old_path, destination)
            if not destination.exists():
                raise ValueError(f"Mask instance {instance['id']} has no mask data.")
        normalised.append(instance)

    removed = [item for key, item in previous_instances.items() if key not in active_ids]
    if removed:
        deleted_dir = root / "backups" / "deleted_masks"
        deleted_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        for item in removed:
            if not item.get("mask_path"):
                continue
            old_path = safe_resolve(root, item["mask_path"])
            if old_path.exists():
                shutil.move(str(old_path), str(deleted_dir / f"{image['id']}-{item['id']}-{stamp}.png"))

    revision = int(previous.get("revision", 0)) + 1
    image["revision"] = revision
    image["updated_at"] = utc_now()
    record = _default_annotation_record(image)
    record.update(
        {
            "revision": revision,
            "fruit_id": image.get("fruit_id", ""),
            "view_id": image.get("view_id", ""),
            "status": image.get("status", "unlabelled"),
            "split": image.get("split"),
            "classification_class_uuid": image.get("classification_class_uuid"),
            "classification_class_name": image.get("classification_class_name"),
            "classification_export_index": image.get("classification_export_index"),
            "updated_at": image["updated_at"],
            "instances": normalised,
        }
    )
    atomic_write_json(previous_path, record)


def save_project(project: dict[str, Any], project_dir: Path | str | None = None) -> None:
    root = Path(project_dir or project.get("project_dir") or ".").expanduser().resolve()
    project["project_dir"] = str(root)
    dirty_ids = {str(item) for item in project.get("_dirty_annotation_image_ids", [])}
    if dirty_ids:
        by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in project.get("annotations", []):
            by_image[str(annotation["image_id"])].append(annotation)
        images = {str(image["id"]): image for image in project.get("images", [])}
        for image_id in sorted(dirty_ids):
            if image_id in images:
                _write_image_record(project, images[image_id], by_image.get(image_id, []))
    project["schema_version"] = SCHEMA_VERSION
    project["updated_at"] = utc_now()
    atomic_write_json(root / PROJECT_FILENAME, _manifest_payload(project))
    project.pop("_dirty_annotation_image_ids", None)


def create_or_update_project(source_dir: Path | str, project_dir: Path | str, project_name: str | None = None) -> dict[str, Any]:
    source_root = Path(source_dir).expanduser().resolve()
    root = Path(project_dir).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source image directory does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_root}")

    if project_path(root).exists():
        project = load_project(root)
        project["source_root"] = str(source_root)
    else:
        project = default_project(project_name or source_root.name, source_root, root)
    project["project_dir"] = str(root)
    retained_images: list[dict[str, Any]] = []
    removed_ids: set[str] = set()
    for image in project["images"]:
        candidate = safe_resolve(source_root, image["relative_path"])
        if _is_project_storage_image(candidate, source_root, root):
            removed_ids.add(str(image["id"]))
        else:
            retained_images.append(image)
    if removed_ids:
        project["images"] = retained_images
        project["annotations"] = [
            item for item in project.get("annotations", [])
            if str(item.get("image_id")) not in removed_ids
        ]
        if str(project.get("last_image_id")) in removed_ids:
            project["last_image_id"] = retained_images[0]["id"] if retained_images else None
    existing = {image["relative_path"].replace("\\", "/"): image for image in project["images"]}
    ids = {str(image["id"]) for image in project["images"]}
    imported = 0
    skipped = 0
    warnings: list[str] = []
    for path in iter_source_images(source_root):
        if _is_project_storage_image(path, source_root, root):
            continue
        relative = path.relative_to(source_root).as_posix()
        if relative in existing:
            skipped += 1
            continue
        try:
            width, height = image_size(path)
            checksum = sha256_file(path)
            perceptual_hash = average_perceptual_hash(path)
        except Exception as exc:  # pragma: no cover - decoder messages vary by platform
            warnings.append(f"Skipped unreadable image {relative}: {exc}")
            continue
        image_id = stable_image_id(relative)
        if image_id in ids:
            image_id = uuid.uuid4().hex[:24]
        ids.add(image_id)
        now = utc_now()
        project["images"].append(
            {
                "id": image_id,
                "relative_path": relative,
                "file_name": path.name,
                "width": width,
                "height": height,
                "sha256": checksum,
                "perceptual_hash": perceptual_hash,
                "status": "unlabelled",
                "reviewed": False,
                "excluded": False,
                "exclusion_reason": "",
                "fruit_id": "",
                "view_id": path.stem,
                "imported_at": now,
                "updated_at": now,
                "split": None,
                "has_warning": False,
                "revision": 0,
                "classification_class_uuid": None,
                "classification_class_name": None,
                "classification_export_index": None,
            }
        )
        imported += 1
    project["images"].sort(key=lambda item: item["relative_path"].casefold())
    project["import_summary"] = {
        "imported": imported,
        "skipped_existing": skipped,
        "removed_project_artifacts": len(removed_ids),
        "warnings": warnings,
    }
    save_project(project, root)
    return load_project(root)


def image_abs_path(project: dict[str, Any], image: dict[str, Any]) -> Path:
    return safe_resolve(Path(project["source_root"]), image["relative_path"])


def annotations_for_image(project: dict[str, Any], image_id: str | int) -> list[dict[str, Any]]:
    return [annotation for annotation in project.get("annotations", []) if str(annotation["image_id"]) == str(image_id)]


def find_image(project: dict[str, Any], image_id: str | int) -> dict[str, Any]:
    for image in project["images"]:
        if str(image["id"]) == str(image_id):
            return image
    raise KeyError(f"Unknown image id: {image_id}")


def replace_image_annotations(
    project: dict[str, Any],
    image_id: str | int,
    annotations: list[dict[str, Any]],
    image_updates: dict[str, Any] | None = None,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    image = find_image(project, image_id)
    if expected_revision is not None and int(image.get("revision", 0)) != int(expected_revision):
        raise RuntimeError(
            f"Annotation revision conflict: expected {expected_revision}, current {image.get('revision', 0)}. Reload before saving."
        )
    project["annotations"] = [item for item in project.get("annotations", []) if str(item["image_id"]) != str(image_id)]
    project["annotations"].extend([{**item, "image_id": str(image_id)} for item in annotations])
    if image_updates:
        for key in (
            "status",
            "reviewed",
            "excluded",
            "exclusion_reason",
            "fruit_id",
            "view_id",
            "split",
            "classification_class_uuid",
        ):
            if key in image_updates:
                image[key] = image_updates[key]
        if "classification_class_uuid" in image_updates:
            class_uuid = image_updates.get("classification_class_uuid")
            if class_uuid:
                selected_class = next(
                    (
                        item
                        for item in project.get("classification_classes", [])
                        if str(item["uuid"]) == str(class_uuid)
                    ),
                    None,
                )
                if selected_class is None:
                    raise ValueError(f"Image references unknown classification class UUID {class_uuid}.")
                image["classification_class_uuid"] = str(selected_class["uuid"])
                image["classification_class_name"] = str(selected_class["name"])
                image["classification_export_index"] = int(selected_class["export_index"])
            else:
                image["classification_class_uuid"] = None
                image["classification_class_name"] = None
                image["classification_export_index"] = None
    if image.get("excluded") or image.get("status") == "excluded":
        image["excluded"] = True
        image["status"] = "excluded"
    elif annotations and image.get("status") == "unlabelled":
        image["status"] = "draft"
    elif not annotations and image.get("status") != "excluded" and not (image_updates and "status" in image_updates):
        image["status"] = "unlabelled"
    image["reviewed"] = image.get("status") == "approved"
    image["grade_ids"] = sorted(
        {
            int(item.get("training_class_id", item.get("category_id")))
            for item in annotations
            if item.get("annotation_kind") == "fruit_box"
            and item.get("training_class_id", item.get("category_id")) is not None
        }
    )
    image["updated_at"] = utc_now()
    project["last_image_id"] = str(image_id)
    dirty = {str(item) for item in project.get("_dirty_annotation_image_ids", [])}
    dirty.add(str(image_id))
    project["_dirty_annotation_image_ids"] = sorted(dirty)
    return image


def project_summary(project: dict[str, Any]) -> dict[str, Any]:
    counts = defaultdict(int)
    classification_counts = defaultdict(int)
    for annotation in project.get("annotations", []):
        if annotation.get("annotation_kind") == "fruit_box" and annotation.get("training_class_id") is not None:
            counts[str(int(annotation["training_class_id"]))] += 1
    for image in project.get("images", []):
        if image.get("classification_export_index") is not None:
            classification_counts[str(int(image["classification_export_index"]))] += 1
    return {
        "image_count": len(project.get("images", [])),
        "annotation_count": len(project.get("annotations", [])),
        "approved_count": sum(1 for image in project.get("images", []) if image.get("status") == "approved"),
        "reviewed_count": sum(1 for image in project.get("images", []) if image.get("status") in {"review", "approved"}),
        "excluded_count": sum(1 for image in project.get("images", []) if image.get("excluded")),
        "warning_count": sum(1 for image in project.get("images", []) if image.get("has_warning")),
        "detection_class_counts": {str(item["export_index"]): counts[str(item["export_index"])] for item in project.get("detection_classes", [])},
        "class_counts": {str(item["export_index"]): counts[str(item["export_index"])] for item in project.get("detection_classes", [])},
        "classification_class_counts": {
            str(item["export_index"]): classification_counts[str(item["export_index"])]
            for item in project.get("classification_classes", [])
        },
        "unlabelled_count": sum(1 for image in project.get("images", []) if image.get("status") == "unlabelled"),
    }
