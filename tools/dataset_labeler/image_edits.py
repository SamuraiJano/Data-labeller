from __future__ import annotations

import base64
import copy
import io
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps

from .core import (
    atomic_write_bytes,
    average_perceptual_hash,
    find_image,
    image_abs_path,
    load_project,
    mask_path,
    replace_image_annotations,
    safe_resolve,
    save_project,
    sha256_file,
    utc_now,
)

ImageEditOperation = Literal["rotate_left", "rotate_right", "flip_horizontal", "flip_vertical"]

_ALPHA_IMAGE_FORMATS = {"PNG", "TIFF", "WEBP"}

_TRANSPOSE = {
    "rotate_left": Image.Transpose.ROTATE_90,
    "rotate_right": Image.Transpose.ROTATE_270,
    "flip_horizontal": Image.Transpose.FLIP_LEFT_RIGHT,
    "flip_vertical": Image.Transpose.FLIP_TOP_BOTTOM,
}


def _transformed_size(operation: ImageEditOperation, width: int, height: int) -> tuple[int, int]:
    return (height, width) if operation in {"rotate_left", "rotate_right"} else (width, height)


def _transform_point(
    point: tuple[float, float],
    operation: ImageEditOperation,
    width: int,
    height: int,
) -> tuple[float, float]:
    x, y = point
    if operation == "rotate_left":
        return y, width - 1 - x
    if operation == "rotate_right":
        return height - 1 - y, x
    if operation == "flip_horizontal":
        return width - 1 - x, y
    return x, height - 1 - y


def _transform_bbox(
    bbox: list[float],
    operation: ImageEditOperation,
    width: int,
    height: int,
) -> list[float]:
    x, y, box_width, box_height = (float(value) for value in bbox)
    if operation == "rotate_left":
        return [y, width - x - box_width, box_height, box_width]
    if operation == "rotate_right":
        return [height - y - box_height, x, box_height, box_width]
    if operation == "flip_horizontal":
        return [width - x - box_width, y, box_width, box_height]
    return [x, height - y - box_height, box_width, box_height]


def _transform_prompt_metadata(
    metadata: dict[str, Any],
    operation: ImageEditOperation,
    width: int,
    height: int,
) -> dict[str, Any]:
    transformed = copy.deepcopy(metadata)
    points = transformed.get("points")
    if isinstance(points, list):
        next_points: list[Any] = []
        for point in points:
            if isinstance(point, dict) and "x" in point and "y" in point:
                x, y = _transform_point((float(point["x"]), float(point["y"])), operation, width, height)
                next_points.append({**point, "x": x, "y": y})
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = _transform_point((float(point[0]), float(point[1])), operation, width, height)
                next_points.append([x, y, *point[2:]])
            else:
                next_points.append(point)
        transformed["points"] = next_points
    box = transformed.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x1, y1, x2, y2 = (float(value) for value in box)
        corners = [
            _transform_point((x1, y1), operation, width, height),
            _transform_point((x2, y1), operation, width, height),
            _transform_point((x1, y2), operation, width, height),
            _transform_point((x2, y2), operation, width, height),
        ]
        transformed["box"] = [
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        ]
    return transformed


def _encode_source_image(image: Image.Image, source_path: Path, exif: Image.Exif, icc_profile: bytes | None) -> bytes:
    image_format = Image.registered_extensions().get(source_path.suffix.lower())
    if not image_format:
        raise ValueError(f"Editing {source_path.suffix or 'this image format'} is not supported.")
    output = io.BytesIO()
    options: dict[str, Any] = {}
    if icc_profile:
        options["icc_profile"] = icc_profile
    if image_format in {"JPEG", "TIFF"} and exif:
        options["exif"] = exif.tobytes()
    if image_format == "JPEG":
        if image.mode not in {"RGB", "L", "CMYK"}:
            image = image.convert("RGB")
        options.update({"quality": 95, "subsampling": 0, "optimize": True})
    elif image_format == "PNG":
        options["optimize"] = False
    image.save(output, format=image_format, **options)
    return output.getvalue()


def _encode_mask(mask: Image.Image, operation: ImageEditOperation) -> bytes:
    transformed = mask.convert("L").transpose(_TRANSPOSE[operation])
    binary = Image.fromarray(
        np.where(np.asarray(transformed, dtype=np.uint8) > 0, 255, 0).astype(np.uint8),
        mode="L",
    )
    output = io.BytesIO()
    binary.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _decode_erase_mask(value: str, width: int, height: int) -> np.ndarray:
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            if opened.size != (width, height):
                raise ValueError(
                    f"Erase mask dimensions {opened.width}x{opened.height} do not match image {width}x{height}."
                )
            channel = opened.getchannel("A") if "A" in opened.getbands() else opened.convert("L")
            return np.asarray(channel, dtype=np.uint8) > 0
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Erase mask is not valid PNG data.") from exc


def _preserved_image_updates(image: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": image.get("status", "unlabelled"),
        "reviewed": image.get("status") == "approved",
        "excluded": bool(image.get("excluded")),
        "exclusion_reason": image.get("exclusion_reason", ""),
        "fruit_id": image.get("fruit_id", ""),
        "view_id": image.get("view_id", ""),
        "split": image.get("split"),
        "classification_class_uuid": image.get("classification_class_uuid"),
    }


def _append_edit_history(
    image: dict[str, Any],
    operation: str,
    backup_path: Path,
    project_root: Path,
    previous_size: tuple[int, int],
    new_size: tuple[int, int],
) -> None:
    history = list(image.get("image_edit_history") or [])
    history.append(
        {
            "operation": operation,
            "edited_at": utc_now(),
            "previous_size": list(previous_size),
            "new_size": list(new_size),
            "backup_path": backup_path.relative_to(project_root).as_posix(),
        }
    )
    image["image_edit_history"] = history[-100:]


def edit_project_image(
    project_dir: Path | str,
    image_id: str,
    operation: ImageEditOperation,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Persist a geometric source-image edit while keeping annotations aligned."""
    if operation not in _TRANSPOSE:
        raise ValueError(f"Unknown image edit operation: {operation}")

    root = Path(project_dir).expanduser().resolve()
    project = load_project(root, annotation_image_ids={image_id})
    image = find_image(project, image_id)
    if expected_revision is not None and int(image.get("revision", 0)) != int(expected_revision):
        raise RuntimeError(
            f"Image revision conflict: expected {expected_revision}, current {image.get('revision', 0)}. Reload before editing."
        )

    source_path = image_abs_path(project, image)
    if not source_path.is_file():
        raise FileNotFoundError(f"Image file is missing: {source_path}")
    original_source = source_path.read_bytes()
    old_width, old_height = int(image["width"]), int(image["height"])

    with Image.open(io.BytesIO(original_source)) as opened:
        opened.load()
        oriented = ImageOps.exif_transpose(opened)
        if oriented.size != (old_width, old_height):
            raise ValueError(
                f"Stored image dimensions {old_width}x{old_height} do not match the source file "
                f"{oriented.width}x{oriented.height}. Rescan the project before editing."
            )
        exif = opened.getexif()
        if exif:
            exif[274] = 1
        icc_profile = opened.info.get("icc_profile")
        transformed_source = oriented.transpose(_TRANSPOSE[operation])
        new_width, new_height = transformed_source.size
        if exif:
            exif[40962] = new_width
            exif[40963] = new_height
        source_payload = _encode_source_image(transformed_source, source_path, exif, icc_profile)

    expected_size = _transformed_size(operation, old_width, old_height)
    if (new_width, new_height) != expected_size:  # pragma: no cover - protects against Pillow regressions
        raise RuntimeError("The edited image dimensions were not what the requested operation should produce.")

    instances = [item for item in project.get("annotations", []) if str(item["image_id"]) == str(image_id)]
    transformed_instances: list[dict[str, Any]] = []
    mask_payloads: dict[Path, bytes] = {}
    original_masks: dict[Path, bytes] = {}
    for stored in instances:
        instance = copy.deepcopy(stored)
        if instance.get("annotation_kind") == "fruit_box" and instance.get("bbox"):
            instance["bbox"] = _transform_bbox(instance["bbox"], operation, old_width, old_height)
        elif instance.get("mask_path"):
            source_mask_path = safe_resolve(root, instance["mask_path"])
            original_mask = source_mask_path.read_bytes()
            with Image.open(io.BytesIO(original_mask)) as opened_mask:
                opened_mask.load()
                if opened_mask.size != (old_width, old_height):
                    raise ValueError(
                        f"Mask {instance['id']} dimensions {opened_mask.width}x{opened_mask.height} do not match "
                        f"image {old_width}x{old_height}."
                    )
                destination = mask_path(root, image_id, str(instance["id"]))
                mask_payloads[destination] = _encode_mask(opened_mask, operation)
                original_masks[destination] = original_mask
        if isinstance(instance.get("prompt_metadata"), dict):
            instance["prompt_metadata"] = _transform_prompt_metadata(
                instance["prompt_metadata"], operation, old_width, old_height
            )
        transformed_instances.append(instance)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_path = root / "backups" / "source_images" / str(image_id) / f"{stamp}-{source_path.name}"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, backup_path)

    try:
        atomic_write_bytes(source_path, source_payload, backup=False)
        for destination, payload in mask_payloads.items():
            atomic_write_bytes(destination, payload)

        image["width"] = new_width
        image["height"] = new_height
        image["sha256"] = sha256_file(source_path)
        image["perceptual_hash"] = average_perceptual_hash(source_path)
        _append_edit_history(
            image,
            operation,
            backup_path,
            root,
            (old_width, old_height),
            (new_width, new_height),
        )
        replace_image_annotations(
            project,
            image_id,
            transformed_instances,
            _preserved_image_updates(image),
            expected_revision,
        )
        save_project(project, root)
    except Exception:
        atomic_write_bytes(source_path, original_source, backup=False)
        for destination, payload in original_masks.items():
            atomic_write_bytes(destination, payload, backup=False)
        raise

    refreshed = load_project(root, annotation_image_ids={image_id})
    refreshed_image = find_image(refreshed, image_id)
    refreshed_instances = [
        item for item in refreshed.get("annotations", []) if str(item["image_id"]) == str(image_id)
    ]
    return {
        "image": refreshed_image,
        "annotations": refreshed_instances,
        "backup_path": backup_path.relative_to(root).as_posix(),
    }


def erase_project_image_pixels(
    project_dir: Path | str,
    image_id: str,
    erase_mask_data: str,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Make painted source pixels transparent while preserving image geometry and annotations."""
    root = Path(project_dir).expanduser().resolve()
    project = load_project(root, annotation_image_ids={image_id})
    image = find_image(project, image_id)
    if expected_revision is not None and int(image.get("revision", 0)) != int(expected_revision):
        raise RuntimeError(
            f"Image revision conflict: expected {expected_revision}, current {image.get('revision', 0)}. Reload before editing."
        )

    source_path = image_abs_path(project, image)
    if not source_path.is_file():
        raise FileNotFoundError(f"Image file is missing: {source_path}")
    image_format = Image.registered_extensions().get(source_path.suffix.lower())
    if not image_format:
        raise ValueError(f"Editing {source_path.suffix or 'this image format'} is not supported.")
    destination_path = source_path if image_format in _ALPHA_IMAGE_FORMATS else source_path.with_suffix(".png")
    if destination_path != source_path and destination_path.exists():
        raise ValueError(
            f"Cannot convert {source_path.name} to a transparency-preserving PNG because {destination_path.name} already exists."
        )

    original_source = source_path.read_bytes()
    width, height = int(image["width"]), int(image["height"])
    erase_mask = _decode_erase_mask(erase_mask_data, width, height)
    if not np.any(erase_mask):
        raise ValueError("Paint at least one pixel before applying the eraser.")

    with Image.open(io.BytesIO(original_source)) as opened:
        opened.load()
        oriented = ImageOps.exif_transpose(opened)
        if oriented.size != (width, height):
            raise ValueError(
                f"Stored image dimensions {width}x{height} do not match the source file "
                f"{oriented.width}x{oriented.height}. Rescan the project before editing."
            )
        exif = opened.getexif()
        if exif:
            exif[274] = 1
            exif[40962] = width
            exif[40963] = height
        icc_profile = opened.info.get("icc_profile")
        pixels = np.asarray(oriented.convert("RGBA")).copy()
        changed_pixels = erase_mask & (pixels[:, :, 3] > 0)
        if not np.any(changed_pixels):
            raise ValueError("The painted area is already transparent.")
        pixels[erase_mask, 3] = 0
        edited_source = Image.fromarray(pixels, mode="RGBA")
        source_payload = _encode_source_image(edited_source, destination_path, exif, icc_profile)

    instances = [item for item in project.get("annotations", []) if str(item["image_id"]) == str(image_id)]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_path = root / "backups" / "source_images" / str(image_id) / f"{stamp}-{source_path.name}"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, backup_path)

    try:
        atomic_write_bytes(destination_path, source_payload, backup=False)
        if destination_path != source_path:
            source_path.unlink()
            image["relative_path"] = destination_path.relative_to(Path(project["source_root"]).resolve()).as_posix()
            image["file_name"] = destination_path.name
        image["sha256"] = sha256_file(destination_path)
        image["perceptual_hash"] = average_perceptual_hash(destination_path)
        _append_edit_history(image, "erase_pixels", backup_path, root, (width, height), (width, height))
        replace_image_annotations(
            project,
            image_id,
            instances,
            _preserved_image_updates(image),
            expected_revision,
        )
        save_project(project, root)
    except Exception:
        atomic_write_bytes(source_path, original_source, backup=False)
        if destination_path != source_path:
            destination_path.unlink(missing_ok=True)
        raise

    refreshed = load_project(root, annotation_image_ids={image_id})
    refreshed_image = find_image(refreshed, image_id)
    refreshed_instances = [
        item for item in refreshed.get("annotations", []) if str(item["image_id"]) == str(image_id)
    ]
    return {
        "image": refreshed_image,
        "annotations": refreshed_instances,
        "backup_path": backup_path.relative_to(root).as_posix(),
        "erased_pixel_count": int(np.count_nonzero(changed_pixels)),
        "converted_to_png": destination_path != source_path,
    }
