from __future__ import annotations

import base64
import io
import json
import random
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from PIL import Image, ImageOps

from .core import (
    IMAGE_EXTENSIONS,
    PROJECT_FILENAME,
    atomic_write_json,
    atomic_write_text,
    create_or_update_project,
    load_project,
    replace_image_annotations,
    save_project,
    sha256_file,
    stable_image_id,
    utc_now,
    validate_class_definitions,
)
from .splitting import create_split_manifest, deterministic_split_assignments

SYNTHETIC_SCHEMA_VERSION = "1.5"
SYNTHETIC_NAMESPACE = uuid.UUID("778c0a76-aa41-4ccf-b2d4-89de2ba8b24c")
ProgressCallback = Callable[[dict[str, Any]], None]


def _report_progress(
    callback: ProgressCallback | None,
    *,
    completed: int,
    total: int,
    percent: int,
    stage: str,
    message: str,
) -> None:
    if callback:
        callback({
            "completed": int(completed),
            "total": int(total),
            "percent": max(0, min(100, int(percent))),
            "stage": stage,
            "message": message,
        })


def _source_files(root: Path, *, foreground: bool) -> list[Path]:
    extensions = {".png", ".webp", ".tif", ".tiff"} if foreground else IMAGE_EXTENSIONS
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )


def _asset_metadata(foreground_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = foreground_root.parent / "assets_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Synthetic asset manifest is not valid JSON: {manifest_path}: {exc}") from exc
    metadata: dict[str, dict[str, Any]] = {}
    for item in payload.get("foregrounds", []):
        relative = str(item.get("path") or "").replace("\\", "/")
        if not relative:
            continue
        candidate = (foreground_root.parent / relative).resolve()
        try:
            candidate.relative_to(foreground_root.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"Foreground manifest path escapes the asset folder: {relative}") from exc
        metadata[str(candidate)] = item
    return metadata


def _category(item: dict[str, Any], fallback_name: str) -> dict[str, Any]:
    raw = item.get("category") if isinstance(item.get("category"), dict) else {}
    name = str(raw.get("name") or item.get("class_name") or fallback_name or "synthetic_object").strip()
    display_name = str(raw.get("display_name") or item.get("display_label") or name.replace("_", " ").title()).strip()
    class_uuid = str(raw.get("uuid") or item.get("class_uuid") or uuid.uuid5(SYNTHETIC_NAMESPACE, f"class:{name.casefold()}"))
    try:
        class_uuid = str(uuid.UUID(class_uuid))
    except ValueError:
        class_uuid = str(uuid.uuid5(SYNTHETIC_NAMESPACE, f"class:{name.casefold()}"))
    return {
        "uuid": class_uuid,
        "name": name.lower().replace(" ", "_"),
        "display_name": display_name,
        "color": str(raw.get("color") or item.get("color") or "#42b883"),
        "meaning": raw.get("meaning"),
    }


def _load_foreground(path: Path) -> tuple[Image.Image, tuple[int, int]]:
    try:
        with Image.open(path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGBA")
    except Exception as exc:
        raise ValueError(f"Could not read foreground {path.name}: {exc}") from exc
    original_size = source.size
    alpha = source.getchannel("A")
    bounds = alpha.getbbox()
    if bounds is None:
        raise ValueError(f"Foreground {path.name} has an empty alpha mask.")
    return source.crop(bounds), original_size


def _transform_foreground(source: Image.Image, scale: float, rotation_degrees: float) -> Image.Image:
    width = max(1, round(source.width * scale))
    height = max(1, round(source.height * scale))
    resized = source.resize((width, height), Image.Resampling.LANCZOS)
    rotated = resized.rotate(rotation_degrees, resample=Image.Resampling.BICUBIC, expand=True)
    bounds = rotated.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("A transformed foreground unexpectedly became empty.")
    return rotated.crop(bounds)


def _mask_png_data(mask: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(output, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _overlap_slices(
    first_x: int,
    first_y: int,
    first_shape: tuple[int, int],
    second_x: int,
    second_y: int,
    second_shape: tuple[int, int],
) -> tuple[slice, slice, slice, slice] | None:
    """Return aligned y/x slices for the intersection of two positioned masks."""
    first_height, first_width = first_shape
    second_height, second_width = second_shape
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first_width, second_x + second_width)
    bottom = min(first_y + first_height, second_y + second_height)
    if left >= right or top >= bottom:
        return None
    return (
        slice(top - first_y, bottom - first_y),
        slice(left - first_x, right - first_x),
        slice(top - second_y, bottom - second_y),
        slice(left - second_x, right - second_x),
    )


def _sample_location(
    rng: random.Random,
    canvas_width: int,
    canvas_height: int,
    alpha: np.ndarray,
    overlap_target: dict[str, Any] | None = None,
) -> tuple[int, int]:
    max_x = canvas_width - alpha.shape[1]
    max_y = canvas_height - alpha.shape[0]
    if overlap_target is None:
        return rng.randint(0, max_x), rng.randint(0, max_y)

    target_alpha = overlap_target["_alpha"]
    min_x = max(0, int(overlap_target["x"]) - alpha.shape[1] + 1)
    max_target_x = min(max_x, int(overlap_target["x"]) + target_alpha.shape[1] - 1)
    min_y = max(0, int(overlap_target["y"]) - alpha.shape[0] + 1)
    max_target_y = min(max_y, int(overlap_target["y"]) + target_alpha.shape[0] - 1)
    if min_x > max_target_x or min_y > max_target_y:
        return rng.randint(0, max_x), rng.randint(0, max_y)
    return rng.randint(min_x, max_target_x), rng.randint(min_y, max_target_y)


def _placement_allowed(
    alpha: np.ndarray,
    alpha_pixels: int,
    x: int,
    y: int,
    placements: list[dict[str, Any]],
    coverage: np.ndarray,
    max_overlap_fraction: float,
    require_overlap: bool,
) -> bool:
    candidate_overlap = alpha & (coverage[y : y + alpha.shape[0], x : x + alpha.shape[1]] > 0)
    candidate_overlap_pixels = int(np.count_nonzero(candidate_overlap))
    if require_overlap != bool(candidate_overlap_pixels):
        return False
    if candidate_overlap_pixels / alpha_pixels > max_overlap_fraction:
        return False

    for existing in placements:
        slices = _overlap_slices(
            int(existing["x"]),
            int(existing["y"]),
            existing["_alpha"].shape,
            x,
            y,
            alpha.shape,
        )
        if slices is None:
            continue
        existing_y, existing_x, candidate_y, candidate_x = slices
        intersection = existing["_alpha"][existing_y, existing_x] & alpha[candidate_y, candidate_x]
        if not np.any(intersection):
            continue
        previous_overlap = existing["_overlap_alpha"][existing_y, existing_x]
        additional_overlap = int(np.count_nonzero(intersection & ~previous_overlap))
        prospective_overlap = int(existing["_overlap_pixels"]) + additional_overlap
        if prospective_overlap / int(existing["_full_pixels"]) > max_overlap_fraction:
            return False
    return True


def _apply_occlusion(
    alpha: np.ndarray,
    x: int,
    y: int,
    placements: list[dict[str, Any]],
) -> np.ndarray:
    """Clip lower objects to visible pixels and return this object's overlap mask."""
    candidate_overlap = np.zeros_like(alpha)
    for existing in placements:
        slices = _overlap_slices(
            int(existing["x"]),
            int(existing["y"]),
            existing["_alpha"].shape,
            x,
            y,
            alpha.shape,
        )
        if slices is None:
            continue
        existing_y, existing_x, candidate_y, candidate_x = slices
        intersection = existing["_alpha"][existing_y, existing_x] & alpha[candidate_y, candidate_x]
        if not np.any(intersection):
            continue
        existing_overlap = existing["_overlap_alpha"][existing_y, existing_x]
        new_existing_overlap = intersection & ~existing_overlap
        existing_overlap |= intersection
        existing["_overlap_pixels"] += int(np.count_nonzero(new_existing_overlap))
        existing["_visible_alpha"][existing_y, existing_x] &= ~alpha[candidate_y, candidate_x]
        candidate_overlap[candidate_y, candidate_x] |= intersection
    return candidate_overlap


def _positioned_bbox(mask: np.ndarray, x: int, y: int) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("Configured overlap fully hid a synthetic foreground.")
    return [
        int(x + xs.min()),
        int(y + ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def _yolo_bbox_row(class_index: int, bbox: list[float], width: int, height: int) -> str:
    x, y, box_width, box_height = [float(value) for value in bbox]
    values = (
        (x + box_width / 2.0) / width,
        (y + box_height / 2.0) / height,
        box_width / width,
        box_height / height,
    )
    return " ".join([str(class_index), *(f"{value:.6f}" for value in values)])


def _write_yolo_detection(project: dict[str, Any], staging: Path) -> dict[str, Any]:
    yolo_root = staging / "yolo_detection"
    generated_images_root = staging / "images"
    classes = sorted(project["detection_classes"], key=lambda item: int(item["export_index"]))
    class_indexes = {str(item["uuid"]): int(item["export_index"]) for item in classes}
    annotations_by_image: dict[str, list[dict[str, Any]]] = {}
    for annotation in project.get("annotations", []):
        if annotation.get("annotation_kind") == "fruit_box":
            annotations_by_image.setdefault(str(annotation["image_id"]), []).append(annotation)
    split_counts = {"train": 0, "val": 0}
    label_count = 0
    for image in sorted(project["images"], key=lambda item: str(item["id"])):
        split = str(image.get("split") or "")
        if split not in split_counts:
            raise ValueError(f"Synthetic image {image['id']} has no train/validation split.")
        source = generated_images_root / image["relative_path"]
        image_destination = yolo_root / "images" / split / image["file_name"]
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(image_destination)
        image["relative_path"] = f"{split}/{image['file_name']}"
        rows: list[str] = []
        for annotation in sorted(annotations_by_image.get(str(image["id"]), []), key=lambda item: str(item["uuid"])):
            class_uuid = str(annotation.get("class_uuid") or "")
            if class_uuid not in class_indexes:
                raise ValueError(f"Synthetic box {annotation['uuid']} has an unknown detection class UUID.")
            rows.append(_yolo_bbox_row(
                class_indexes[class_uuid],
                annotation["bbox"],
                int(image["width"]),
                int(image["height"]),
            ))
        label_text = "\n".join(rows) + ("\n" if rows else "")
        atomic_write_text(yolo_root / "labels" / split / f"{Path(image['file_name']).stem}.txt", label_text)
        split_counts[split] += 1
        label_count += len(rows)
    generated_images_root.rmdir()
    data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {int(item["export_index"]): str(item["name"]) for item in classes},
    }
    atomic_write_text(yolo_root / "data.yaml", yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return {"images": split_counts, "labels": label_count}


def _commit(staging: Path, final: Path) -> Path | None:
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
    return backup


def _plan_object_counts(
    total_images: int,
    max_objects_per_image: int,
    required_instances: int,
    rng: random.Random,
) -> list[int]:
    """Create a seeded variable count plan with at least one negative image."""
    maximum_capacity = (total_images - 1) * max_objects_per_image
    if required_instances > maximum_capacity:
        raise ValueError(
            f"Cannot place {required_instances} foreground sources in {total_images} images with a maximum of "
            f"{max_objects_per_image} objects per image while reserving a plain background image."
        )

    plain_background_index = rng.randrange(total_images)
    counts = [rng.randint(0, max_objects_per_image) for _ in range(total_images)]
    counts[plain_background_index] = 0

    missing = max(0, required_instances - sum(counts))
    while missing:
        candidates = [
            index
            for index, count in enumerate(counts)
            if index != plain_background_index and count < max_objects_per_image
        ]
        if not candidates:
            raise RuntimeError("Object-count planning exhausted the available non-background slots.")
        index = rng.choice(candidates)
        increment = rng.randint(1, min(missing, max_objects_per_image - counts[index]))
        counts[index] += increment
        missing -= increment
    return counts


def _partition_foregrounds(
    asset_count: int,
    available_splits: list[str],
    ratios: dict[str, float],
    seed: int,
) -> tuple[dict[int, str], dict[str, int]]:
    """Create a seeded, disjoint foreground partition for the available image splits."""
    splits = [split for split in ("train", "val") if split in available_splits and ratios.get(split, 0) > 0]
    if not splits:
        raise ValueError("Synthetic generation requires at least one train or validation image split.")

    ratio_total = sum(ratios[split] for split in splits)
    raw = {split: asset_count * ratios[split] / ratio_total for split in splits}
    counts = {split: int(raw[split]) for split in splits}
    remainder = asset_count - sum(counts.values())
    order = sorted(splits, key=lambda split: (-(raw[split] - counts[split]), splits.index(split)))
    for split in order[:remainder]:
        counts[split] += 1

    # A useful validation set needs its own positive sources. When the asset set
    # is large enough, keep every available split non-empty even if rounding a
    # very small dataset would otherwise assign all sources to training.
    if asset_count >= len(splits):
        for empty_split in (split for split in splits if counts[split] == 0):
            donor = max(splits, key=lambda split: (counts[split], ratios[split], -splits.index(split)))
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[empty_split] += 1

    shuffled = list(range(asset_count))
    random.Random(int(seed)).shuffle(shuffled)
    assignments: dict[int, str] = {}
    cursor = 0
    for split in splits:
        for asset_index in shuffled[cursor : cursor + counts[split]]:
            assignments[asset_index] = split
        cursor += counts[split]
    return assignments, {split: counts.get(split, 0) for split in ("train", "val")}


def _plan_split_object_counts(
    composite_splits: list[str],
    max_objects_per_image: int,
    required_by_split: dict[str, int],
    rng: random.Random,
) -> list[int]:
    """Plan counts while reserving a negative and preserving source isolation."""
    counts = [
        rng.randint(0, max_objects_per_image) if required_by_split.get(split, 0) else 0
        for split in composite_splits
    ]
    indices_by_split = {
        split: [index for index, image_split in enumerate(composite_splits) if image_split == split]
        for split in ("train", "val")
    }
    for split, required in required_by_split.items():
        capacity = len(indices_by_split.get(split, [])) * max_objects_per_image
        if required > capacity:
            raise ValueError(
                f"The {split} split has capacity for {capacity} foreground instances but its disjoint "
                f"foreground pool contains {required}. Increase total images or maximum objects per image."
            )

    negative_candidates = [
        index
        for index, split in enumerate(composite_splits)
        if required_by_split.get(split, 0)
        <= (len(indices_by_split[split]) - 1) * max_objects_per_image
    ]
    if not negative_candidates:
        raise ValueError(
            "The 80/20 split does not have enough per-split capacity to use every foreground and retain an "
            "untouched background image. Increase total images or maximum objects per image."
        )
    negative_index = rng.choice(negative_candidates)
    counts[negative_index] = 0

    for split, required in required_by_split.items():
        missing = max(0, required - sum(counts[index] for index in indices_by_split.get(split, [])))
        while missing:
            candidates = [
                index
                for index in indices_by_split.get(split, [])
                if index != negative_index and counts[index] < max_objects_per_image
            ]
            if not candidates:
                raise RuntimeError(f"Object-count planning exhausted the available {split} slots.")
            index = rng.choice(candidates)
            increment = rng.randint(1, min(missing, max_objects_per_image - counts[index]))
            counts[index] += increment
            missing -= increment
    return counts


def generate_synthetic_project(
    foreground_dir: Path | str,
    background_dir: Path | str,
    output_dir: Path | str,
    *,
    objects_per_image: int = 6,
    total_images: int = 100,
    seed: int = 42,
    min_scale: float = 0.10,
    max_scale: float = 0.17,
    max_rotation_degrees: float = 360.0,
    overlap_probability: float = 0.30,
    max_overlap_fraction: float = 0.20,
    placement_attempts: int = 500,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Create reproducible composites with bounded occlusion and an openable project."""
    _report_progress(
        progress_callback,
        completed=0,
        total=total_images,
        percent=0,
        stage="preparing",
        message="Checking foregrounds and backgrounds…",
    )
    foreground_root = Path(foreground_dir).expanduser().resolve()
    background_root = Path(background_dir).expanduser().resolve()
    output_parent = Path(output_dir).expanduser().resolve()
    for label, root in (("Foreground", foreground_root), ("Background", background_root)):
        if not root.is_dir():
            raise NotADirectoryError(f"{label} folder does not exist: {root}")
    if objects_per_image < 1 or total_images < 2:
        raise ValueError(
            "Maximum objects per image must be positive and total images must be at least 2 for "
            "train/validation output."
        )
    if not 0 < min_scale <= max_scale:
        raise ValueError("Foreground scales must satisfy 0 < minimum <= maximum.")
    if not 0 <= max_rotation_degrees <= 360:
        raise ValueError("Maximum rotation must be between 0 and 360 degrees.")
    if not 0 <= overlap_probability <= 1:
        raise ValueError("Overlap probability must be between 0 and 1.")
    if not 0 < max_overlap_fraction <= 0.80:
        raise ValueError("Maximum overlap fraction must be greater than 0 and no more than 0.80.")
    if placement_attempts < 1:
        raise ValueError("Placement attempts must be positive.")

    foreground_paths = _source_files(foreground_root, foreground=True)
    background_paths = _source_files(background_root, foreground=False)
    if not foreground_paths:
        raise ValueError(f"No transparent foreground images were found in {foreground_root}.")
    if not background_paths:
        raise ValueError(f"No supported background images were found in {background_root}.")
    if total_images < len(background_paths):
        raise ValueError(
            f"Total images must be at least {len(background_paths)} so every background can be used."
        )
    minimum_for_foregrounds = 1 + (len(foreground_paths) + objects_per_image - 1) // objects_per_image
    if total_images < minimum_for_foregrounds:
        raise ValueError(
            f"Total images must be at least {minimum_for_foregrounds} with a maximum of {objects_per_image} "
            f"objects per image so all {len(foreground_paths)} foregrounds can be used and at least one plain "
            "background image can be included."
        )

    metadata = _asset_metadata(foreground_root)
    assets: list[dict[str, Any]] = []
    categories_by_uuid: dict[str, dict[str, Any]] = {}
    for path in foreground_paths:
        rgba, original_size = _load_foreground(path)
        item_metadata = metadata.get(str(path.resolve()), {})
        category = _category(item_metadata, "synthetic_object")
        categories_by_uuid.setdefault(category["uuid"], category)
        assets.append({
            "path": path,
            "relative_path": path.relative_to(foreground_root).as_posix(),
            "image": rgba,
            "original_size": original_size,
            "metadata": item_metadata,
            "category": category,
        })

    _report_progress(
        progress_callback,
        completed=0,
        total=total_images,
        percent=3,
        stage="preparing",
        message=f"Loaded {len(assets)} foregrounds and {len(background_paths)} backgrounds.",
    )

    split_ratios = {"train": 0.5, "val": 0.5, "test": 0.0} if total_images == 2 else {"train": 0.8, "val": 0.2, "test": 0.0}
    composite_count = int(total_images)
    image_uuids = [
        str(uuid.uuid5(SYNTHETIC_NAMESPACE, f"image:{seed}:{composite_index}"))
        for composite_index in range(composite_count)
    ]
    split_group_ids = [f"image-{stable_image_id(f'{image_uuid}.png')}" for image_uuid in image_uuids]
    split_assignments = deterministic_split_assignments(split_group_ids, split_ratios, seed)
    composite_splits = [split_assignments[group_id] for group_id in split_group_ids]

    available_splits = [split for split in ("train", "val") if split in composite_splits]
    foreground_split_by_index, foreground_split_counts = _partition_foregrounds(
        len(assets), available_splits, split_ratios, seed
    )
    for asset_index, asset in enumerate(assets):
        asset["split"] = foreground_split_by_index[asset_index]

    rng = random.Random(int(seed))
    background_order = list(range(len(background_paths)))
    rng.shuffle(background_order)
    object_counts = _plan_split_object_counts(
        composite_splits,
        objects_per_image,
        foreground_split_counts,
        rng,
    )
    foreground_decks: dict[str, list[int]] = {}
    deck_indexes = {"train": 0, "val": 0}
    for split in ("train", "val"):
        order = [index for index, asset_split in foreground_split_by_index.items() if asset_split == split]
        rng.shuffle(order)
        slot_count = sum(
            count for count, composite_split in zip(object_counts, composite_splits) if composite_split == split
        )
        foreground_decks[split] = [order[index % len(order)] for index in range(slot_count)] if order else []

    output_parent.mkdir(parents=True, exist_ok=True)
    final = output_parent / "synthetic-data"
    staging = output_parent / f".synthetic-data.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    warnings: list[str] = []
    generated: list[dict[str, Any]] = []
    capacity_shortfalls: list[dict[str, Any]] = []
    source_usage: dict[str, int] = {asset["relative_path"]: 0 for asset in assets}
    background_usage: dict[str, int] = {path.relative_to(background_root).as_posix(): 0 for path in background_paths}
    try:
        for composite_index in range(composite_count):
            composite_split = composite_splits[composite_index]
            background_index = background_order[composite_index % len(background_order)]
            background_path = background_paths[background_index]
            background_relative = background_path.relative_to(background_root).as_posix()
            background_usage[background_relative] += 1
            try:
                with Image.open(background_path) as opened:
                    canvas = ImageOps.exif_transpose(opened).convert("RGBA")
            except Exception as exc:
                raise ValueError(f"Could not read background {background_relative}: {exc}") from exc
            coverage = np.zeros((canvas.height, canvas.width), dtype=np.uint16)
            placements: list[dict[str, Any]] = []
            for slot in range(object_counts[composite_index]):
                asset = assets[foreground_decks[composite_split][deck_indexes[composite_split]]]
                scale = rng.uniform(min_scale, max_scale)
                rotation = rng.uniform(0.0, max_rotation_degrees)
                sprite = _transform_foreground(asset["image"], scale, rotation)
                alpha = np.asarray(sprite.getchannel("A"), dtype=np.uint8) > 0
                alpha_pixels = int(np.count_nonzero(alpha))
                shortfall_reason: str | None = None
                location: tuple[int, int] | None = None
                if sprite.width > canvas.width or sprite.height > canvas.height:
                    wants_overlap = False
                    shortfall_reason = "transformed foreground is larger than this background"
                else:
                    wants_overlap = bool(placements) and overlap_probability > 0 and rng.random() < overlap_probability
                    for _ in range(placement_attempts):
                        target = rng.choice(placements) if wants_overlap else None
                        x, y = _sample_location(rng, canvas.width, canvas.height, alpha, target)
                        if _placement_allowed(
                            alpha,
                            alpha_pixels,
                            x,
                            y,
                            placements,
                            coverage,
                            max_overlap_fraction,
                            require_overlap=wants_overlap,
                        ):
                            location = (x, y)
                            break
                    if location is None and wants_overlap:
                        for _ in range(placement_attempts):
                            x, y = _sample_location(rng, canvas.width, canvas.height, alpha)
                            if _placement_allowed(
                                alpha,
                                alpha_pixels,
                                x,
                                y,
                                placements,
                                coverage,
                                max_overlap_fraction,
                                require_overlap=False,
                            ):
                                location = (x, y)
                                break
                    if location is None:
                        shortfall_reason = "no position remained within the configured overlap limit"
                if location is None:
                    capacity_shortfalls.append({
                        "image_index": composite_index,
                        "background_relative_path": background_relative,
                        "requested_objects": object_counts[composite_index],
                        "placed_objects": len(placements),
                        "deferred_foreground_relative_path": asset["relative_path"],
                        "reason": shortfall_reason,
                    })
                    break
                x, y = location
                overlap_alpha = _apply_occlusion(alpha, x, y, placements)
                canvas.alpha_composite(sprite, (x, y))
                coverage[y : y + sprite.height, x : x + sprite.width] += alpha.astype(np.uint16)
                placement_key = f"{seed}:{composite_index}:{slot}:{asset['relative_path']}"
                box_uuid = str(uuid.uuid5(SYNTHETIC_NAMESPACE, f"box:{placement_key}"))
                mask_uuid = str(uuid.uuid5(SYNTHETIC_NAMESPACE, f"mask:{placement_key}"))
                placements.append({
                    "box_uuid": box_uuid,
                    "mask_uuid": mask_uuid,
                    "foreground_relative_path": asset["relative_path"],
                    "foreground_sha256": sha256_file(asset["path"]),
                    "foreground_split": asset["split"],
                    "category": asset["category"],
                    "source_annotation_uuid": asset["metadata"].get("annotation_uuid"),
                    "source_size": list(asset["original_size"]),
                    "scale": scale,
                    "rotation_degrees": rotation,
                    "x": x,
                    "y": y,
                    "overlap_requested": wants_overlap,
                    "_alpha": alpha,
                    "_visible_alpha": alpha.copy(),
                    "_overlap_alpha": overlap_alpha,
                    "_overlap_pixels": int(np.count_nonzero(overlap_alpha)),
                    "_full_pixels": alpha_pixels,
                })
                deck_indexes[composite_split] += 1
                source_usage[asset["relative_path"]] += 1

            for placement in placements:
                visible_alpha = placement["_visible_alpha"]
                full_pixels = int(placement["_full_pixels"])
                visible_pixels = int(np.count_nonzero(visible_alpha))
                overlap_pixels = int(placement["_overlap_pixels"])
                full_mask = np.zeros(coverage.shape, dtype=bool)
                x = int(placement["x"])
                y = int(placement["y"])
                full_mask[y : y + visible_alpha.shape[0], x : x + visible_alpha.shape[1]] = visible_alpha
                placement.update({
                    "bbox": _positioned_bbox(visible_alpha, x, y),
                    "amodal_bbox": _positioned_bbox(placement["_alpha"], x, y),
                    "overlap_achieved": overlap_pixels > 0,
                    "overlap_pixels": overlap_pixels,
                    "overlap_fraction": overlap_pixels / full_pixels,
                    "occluded_pixels": full_pixels - visible_pixels,
                    "occlusion_fraction": (full_pixels - visible_pixels) / full_pixels,
                    "visible_pixels": visible_pixels,
                    "visible_fraction": visible_pixels / full_pixels,
                    "mask_data": _mask_png_data(full_mask),
                })
                for internal_key in (
                    "_alpha",
                    "_visible_alpha",
                    "_overlap_alpha",
                    "_overlap_pixels",
                    "_full_pixels",
                ):
                    placement.pop(internal_key)

            image_uuid = image_uuids[composite_index]
            image_name = f"{image_uuid}.png"
            destination = staging / "images" / image_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            canvas.convert("RGB").save(destination, format="PNG", optimize=False)
            generated.append({
                "image_uuid": image_uuid,
                "file_name": image_name,
                "planned_split": composite_split,
                "background_relative_path": background_relative,
                "background_sha256": sha256_file(background_path),
                "width": canvas.width,
                "height": canvas.height,
                "requested_object_count": object_counts[composite_index],
                "object_count": len(placements),
                "capacity_shortfall": object_counts[composite_index] - len(placements),
                "placements": placements,
            })
            _report_progress(
                progress_callback,
                completed=composite_index + 1,
                total=composite_count,
                percent=3 + round(((composite_index + 1) / composite_count) * 82),
                stage="compositing",
                message=f"Generated image {composite_index + 1} of {composite_count}.",
            )

        unused = [name for name, count in source_usage.items() if count == 0]
        if unused:
            raise RuntimeError("Generation could not place every foreground: " + ", ".join(unused))
        if capacity_shortfalls:
            first = capacity_shortfalls[0]
            warnings.append(
                f"{len(capacity_shortfalls)} image(s) reached physical placement capacity and received fewer than "
                f"their sampled maximum. The deferred foregrounds continued on later images. First example: image "
                f"{first['image_index'] + 1} placed {first['placed_objects']} of {first['requested_objects']} because "
                f"{first['reason']}."
            )

        project_root = staging / "project"
        project = create_or_update_project(staging / "images", project_root, "Synthetic Data")
        class_definitions = []
        for index, category in enumerate(sorted(categories_by_uuid.values(), key=lambda item: (item["name"].casefold(), item["uuid"]))):
            class_definitions.append({**category, "id": index, "export_index": index})
        project["detection_classes"] = validate_class_definitions(class_definitions)
        project["classes"] = [dict(item) for item in project["detection_classes"]]
        classes = {item["uuid"]: item for item in project["detection_classes"]}
        generated_by_name = {item["file_name"]: item for item in generated}
        for image in project["images"]:
            record = generated_by_name[image["file_name"]]
            annotations: list[dict[str, Any]] = []
            for placement in record["placements"]:
                class_item = classes[placement["category"]["uuid"]]
                provenance = {
                    "generator": "data-labeller",
                    "generator_schema_version": SYNTHETIC_SCHEMA_VERSION,
                    "seed": seed,
                    "foreground_relative_path": placement["foreground_relative_path"],
                    "foreground_split": placement["foreground_split"],
                    "background_relative_path": record["background_relative_path"],
                    "scale": placement["scale"],
                    "rotation_degrees": placement["rotation_degrees"],
                    "x": placement["x"],
                    "y": placement["y"],
                    "overlap_fraction": placement["overlap_fraction"],
                    "occlusion_fraction": placement["occlusion_fraction"],
                    "visible_fraction": placement["visible_fraction"],
                }
                annotations.extend([
                    {
                        "id": placement["box_uuid"],
                        "type": "fruit_box",
                        "class_uuid": class_item["uuid"],
                        "export_index": class_item["export_index"],
                        "bbox": [float(value) for value in placement["bbox"]],
                        "source": "synthetic",
                        "prompt_metadata": provenance,
                    },
                    {
                        "id": placement["mask_uuid"],
                        "type": "foreground_mask",
                        "parent_fruit_instance_id": placement["box_uuid"],
                        "mask_data": placement.pop("mask_data"),
                        "source": "synthetic",
                        "prompt_metadata": provenance,
                    },
                ])
            replace_image_annotations(project, image["id"], annotations, {"status": "draft", "view_id": image["file_name"]})
        save_project(project, project_root)

        _report_progress(
            progress_callback,
            completed=composite_count,
            total=composite_count,
            percent=90,
            stage="annotations",
            message="Saved masks and bounding boxes.",
        )

        create_split_manifest(project_root, split_ratios, seed=seed)
        project = load_project(project_root)
        _report_progress(
            progress_callback,
            completed=composite_count,
            total=composite_count,
            percent=94,
            stage="splitting",
            message="Created deterministic train and validation splits.",
        )
        yolo_counts = _write_yolo_detection(project, staging)
        project["source_root"] = str(final / "yolo_detection" / "images")
        project["_dirty_annotation_image_ids"] = [str(image["id"]) for image in project["images"]]
        save_project(project, project_root)
        create_split_manifest(project_root, split_ratios, seed=seed)
        project = load_project(project_root)
        split_by_file = {image["file_name"]: image["split"] for image in project["images"]}
        for record in generated:
            record["split"] = split_by_file[record["file_name"]]
            if record["split"] != record.pop("planned_split"):
                raise RuntimeError(f"Synthetic split planning drifted for {record['file_name']}.")
            if any(
                placement["foreground_split"] != record["split"]
                for placement in record["placements"]
            ):
                raise RuntimeError(f"A foreground crossed the train/validation boundary in {record['file_name']}.")
            record["image_relative_path"] = (
                f"yolo_detection/images/{record['split']}/{record['file_name']}"
            )
        _report_progress(
            progress_callback,
            completed=composite_count,
            total=composite_count,
            percent=98,
            stage="exporting",
            message="Wrote YOLO images, labels, and data.yaml.",
        )

        manifest = {
            "schema_version": SYNTHETIC_SCHEMA_VERSION,
            "created_at": utc_now(),
            "generator": "data-labeller",
            "seed": int(seed),
            "settings": {
                "max_objects_per_image": objects_per_image,
                "total_images": total_images,
                "min_scale": min_scale,
                "max_scale": max_scale,
                "rotation_range_degrees": [0.0, max_rotation_degrees],
                "overlap_policy": "bounded_alpha_overlap_with_visible_masks",
                "foreground_split_policy": "seeded_disjoint_train_validation_pools",
                "overlap_probability": overlap_probability,
                "max_overlap_fraction": max_overlap_fraction,
                "placement_attempts": placement_attempts,
            },
            "counts": {
                "foreground_sources": len(assets),
                "background_sources": len(background_paths),
                "generated_images": len(generated),
                "placed_instances": sum(len(item["placements"]) for item in generated),
                "overlapping_instances": sum(
                    bool(placement["overlap_achieved"])
                    for item in generated
                    for placement in item["placements"]
                ),
                "occluded_instances": sum(
                    placement["occluded_pixels"] > 0
                    for item in generated
                    for placement in item["placements"]
                ),
                "plain_background_images": sum(not item["placements"] for item in generated),
                "capacity_limited_images": len(capacity_shortfalls),
                "capacity_limited_instances": sum(item["capacity_shortfall"] for item in generated),
            },
            "object_count_distribution": {
                str(count): sum(len(item["placements"]) == count for item in generated)
                for count in sorted({len(item["placements"]) for item in generated})
            },
            "planned_object_count_distribution": {
                str(count): object_counts.count(count)
                for count in sorted(set(object_counts))
            },
            "split": {
                "seed": int(seed),
                "ratios": split_ratios,
                "counts": yolo_counts["images"],
            },
            "foreground_usage": source_usage,
            "foreground_split_counts": foreground_split_counts,
            "foreground_splits": {
                asset["relative_path"]: asset["split"]
                for asset in assets
            },
            "background_usage": background_usage,
            "paths": {
                "images": "yolo_detection/images",
                "project": "project",
                "project_manifest": f"project/{PROJECT_FILENAME}",
                "yolo_detection": "yolo_detection",
                "yolo_detection_yaml": "yolo_detection/data.yaml",
            },
            "yolo_detection": {
                "format": "class x_center y_center width height (normalized)",
                "images": yolo_counts["images"],
                "labels": yolo_counts["labels"],
                "classes": [
                    {"uuid": item["uuid"], "name": item["name"], "export_index": item["export_index"]}
                    for item in project["detection_classes"]
                ],
            },
            "images": generated,
            "capacity_shortfalls": capacity_shortfalls,
            "warnings": warnings,
        }
        atomic_write_json(staging / "synthetic_generation_manifest.json", manifest, backup=False)

        project_manifest_path = project_root / PROJECT_FILENAME
        project_manifest = json.loads(project_manifest_path.read_text(encoding="utf-8"))
        project_manifest["source_root"] = str(final / "yolo_detection" / "images")
        project_manifest["project_dir"] = str(final / "project")
        project_manifest["split"]["manifest_json"] = str(final / "project" / "manifests" / "split_manifest.json")
        project_manifest["split"]["manifest_csv"] = str(final / "project" / "manifests" / "split_manifest.csv")
        atomic_write_json(project_manifest_path, project_manifest, backup=False)
        backup = _commit(staging, final)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    _report_progress(
        progress_callback,
        completed=composite_count,
        total=composite_count,
        percent=100,
        stage="complete",
        message="Synthetic dataset is ready.",
    )

    return {
        "dataset_root": str(final),
        "project_dir": str(final / "project"),
        "manifest": str(final / "synthetic_generation_manifest.json"),
        "yolo_detection_dir": str(final / "yolo_detection"),
        "detection_yaml": str(final / "yolo_detection" / "data.yaml"),
        "counts": manifest["counts"],
        "warnings": warnings,
        "previous_output": str(backup) if backup else None,
    }
