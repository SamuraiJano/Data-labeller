from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import annotation_path, atomic_write_json, atomic_write_text, load_project, save_project, utc_now

SPLITS = ("train", "val", "test")
INFERRED_FRUIT_RE = re.compile(r"^(.+_\d{3,})_\d+$", re.IGNORECASE)


def _manifest_paths(project_dir: Path) -> tuple[Path, Path]:
    root = project_dir / "manifests"
    return root / "split_manifest.json", root / "split_manifest.csv"


def _validate_ratios(ratios: dict[str, float]) -> dict[str, float]:
    values = {split: float(ratios.get(split, 0.0)) for split in SPLITS}
    if any(value < 0 for value in values.values()):
        raise ValueError("Split ratios cannot be negative.")
    total = sum(values.values())
    if total <= 0 or abs(total - 1.0) > 1e-6:
        raise ValueError("Train, validation and test ratios must sum to 1.0.")
    return values


def _target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    enabled = [split for split in SPLITS if ratios[split] > 0]
    if total < len(enabled):
        # Prefer the most useful splits in workflow order when there are not
        # enough independent groups to populate every requested split.
        return {split: int(split in enabled[:total]) for split in SPLITS}
    raw = {split: total * ratios[split] for split in SPLITS}
    counts = {split: int(raw[split]) for split in SPLITS}
    remainder = total - sum(counts.values())
    order = sorted(SPLITS, key=lambda split: (-(raw[split] - counts[split]), SPLITS.index(split)))
    for split in order[:remainder]:
        counts[split] += 1
    for empty_split in [split for split in enabled if counts[split] == 0]:
        donors = sorted(
            (split for split in enabled if counts[split] > 1),
            key=lambda split: (counts[split], ratios[split], -SPLITS.index(split)),
            reverse=True,
        )
        if donors:
            counts[donors[0]] -= 1
            counts[empty_split] += 1
    return counts


def _assign_fruits(fruit_ids: list[str], ratios: dict[str, float], seed: int) -> dict[str, str]:
    shuffled = sorted(fruit_ids, key=str.casefold)
    random.Random(seed).shuffle(shuffled)
    targets = _target_counts(len(shuffled), ratios)
    assignments: dict[str, str] = {}
    cursor = 0
    for split in SPLITS:
        for fruit_id in shuffled[cursor : cursor + targets[split]]:
            assignments[fruit_id] = split
        cursor += targets[split]
    return assignments


def deterministic_split_assignments(
    item_ids: list[str],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Return the same seeded assignments used by the split manifest writer."""
    return _assign_fruits(item_ids, _validate_ratios(ratios), int(seed))


def _source_class(project: dict[str, Any], image: dict[str, Any]) -> str:
    explicit = str(image.get("source_class") or "").strip()
    if explicit:
        return explicit
    if project.get("source_kind") == "images":
        parts = Path(str(image.get("relative_path") or "")).parts
        if len(parts) > 1 and parts[0].casefold() not in SPLITS:
            return str(parts[0])
    classification = str(image.get("classification_class_name") or "").strip()
    return classification or "All images"


def _group_identity(image: dict[str, Any], source_class: str) -> tuple[str, str, str]:
    for field, kind in (
        ("source_video_id", "source_video"),
        ("capture_group", "capture_group"),
        ("fruit_id", "fruit_id"),
    ):
        value = str(image.get(field) or "").strip()
        if value:
            return f"{source_class.casefold()}::{kind}:{value.casefold()}", kind, value
    match = INFERRED_FRUIT_RE.match(Path(str(image.get("file_name") or image.get("relative_path") or "")).stem)
    if match:
        inferred = match.group(1)
        return f"{source_class.casefold()}::inferred:{inferred.casefold()}", "inferred_fruit_id", inferred
    image_id = str(image["id"])
    return f"{source_class.casefold()}::image:{image_id}", "image", ""


def _stratum_seed(seed: int, source_class: str) -> int:
    if source_class == "All images":
        return int(seed)
    digest = hashlib.sha256(f"{seed}:{source_class.casefold()}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rows_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    fields = list(rows[0].keys()) if rows else ["image_id", "relative_path", "fruit_id", "view_id", "split"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def create_split_manifest(
    project_dir: Path | str,
    ratios: dict[str, float] | None = None,
    seed: int | None = None,
    *,
    pilot_mode: bool = False,
    freeze_test: bool = False,
    confirm_frozen_change: bool = False,
    dataset_version: str | None = None,
) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    project = load_project(root)
    split_settings = project.get("split", {})
    ratios = _validate_ratios(ratios or split_settings.get("ratios") or {"train": 0.70, "val": 0.20, "test": 0.10})
    seed = int(seed if seed is not None else split_settings.get("seed", 42))
    images = [image for image in project["images"] if not image.get("excluded")]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_metadata: dict[str, dict[str, str]] = {}
    groups_by_class: dict[str, list[str]] = defaultdict(list)
    for image in images:
        source_class = _source_class(project, image)
        group_id, group_kind, display_group_id = _group_identity(image, source_class)
        if group_id not in grouped:
            groups_by_class[source_class].append(group_id)
            group_metadata[group_id] = {
                "source_class": source_class,
                "group_kind": group_kind,
                "display_group_id": display_group_id,
            }
        grouped[group_id].append(image)
    assignments: dict[str, str] = {}
    for source_class in sorted(groups_by_class, key=str.casefold):
        assignments.update(
            deterministic_split_assignments(
                groups_by_class[source_class],
                ratios,
                _stratum_seed(seed, source_class),
            )
        )
    rows: list[dict[str, Any]] = []
    for group_id in sorted(grouped, key=str.casefold):
        split = assignments[group_id]
        metadata = group_metadata[group_id]
        for image in sorted(grouped[group_id], key=lambda item: item["relative_path"].casefold()):
            image["split"] = split
            rows.append(
                {
                    "image_id": str(image["id"]),
                    "relative_path": image["relative_path"],
                    "source_class": metadata["source_class"],
                    "group_id": group_id if metadata["display_group_id"] else "",
                    "group_label": metadata["display_group_id"],
                    "group_kind": metadata["group_kind"],
                    "source_video_id": image.get("source_video_id", ""),
                    "fruit_id": image.get("fruit_id", ""),
                    "view_id": image.get("view_id", ""),
                    "split": split,
                    "sha256": image.get("sha256", ""),
                    "annotation_status": image.get("status", ""),
                    "pilot_mode": str(bool(pilot_mode)).lower(),
                }
            )

    json_path, csv_path = _manifest_paths(root)
    previous = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else None
    previous_frozen = bool(split_settings.get("test_frozen") or (previous or {}).get("test_frozen"))
    previous_test = {row["image_id"] for row in (previous or {}).get("rows", []) if row.get("split") == "test"}
    next_test = {row["image_id"] for row in rows if row["split"] == "test"}
    old_version = str(split_settings.get("dataset_version") or (previous or {}).get("dataset_version") or "0.1.0")
    next_version = str(dataset_version or old_version)
    if previous_frozen and previous_test != next_test:
        if not confirm_frozen_change:
            raise ValueError("The test split is frozen. Explicit confirmation and a new dataset version are required to change it.")
        if next_version == old_version:
            raise ValueError(f"Changing a frozen test split requires a new dataset version (current: {old_version}).")

    warnings: list[str] = []
    if pilot_mode:
        warnings.append("Pilot mode is active; this flag is retained for compatibility but image-level splitting is supported by default.")
    if len(grouped) < 10:
        warnings.append(f"Only {len(grouped)} independent capture group(s) are available; split statistics will be weak.")
    class_counts: dict[str, dict[str, dict[str, int]]] = {}
    for source_class in sorted(groups_by_class, key=str.casefold):
        class_rows = [row for row in rows if row["source_class"] == source_class]
        class_counts[source_class] = split_counts(class_rows)
    payload = {
        "schema_version": "1.1",
        "created_at": utc_now(),
        "seed": seed,
        "ratios": ratios,
        "stratified_by": "source_class",
        "counts": split_counts(rows),
        "class_counts": class_counts,
        "dataset_version": next_version,
        "test_frozen": bool(previous_frozen or freeze_test),
        "pilot_mode": bool(pilot_mode),
        "rows": rows,
        "warnings": warnings,
    }
    atomic_write_json(json_path, payload)
    atomic_write_text(csv_path, _rows_csv(rows))
    project["split"] = {
        "ratios": ratios,
        "seed": seed,
        "manifest_json": str(json_path),
        "manifest_csv": str(csv_path),
        "created_at": payload["created_at"],
        "test_frozen": payload["test_frozen"],
        "dataset_version": next_version,
        "pilot_mode": bool(pilot_mode),
    }
    project["_dirty_annotation_image_ids"] = [
        str(image["id"]) for image in images
        if annotation_path(root, image["id"]).is_file()
    ]
    save_project(project, root)
    payload["manifest_json"] = str(json_path)
    payload["manifest_csv"] = str(csv_path)
    return payload


def ensure_split_manifest(project_dir: Path | str, force: bool = False, **kwargs: Any) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    json_path, _ = _manifest_paths(root)
    if json_path.exists() and not force:
        return json.loads(json_path.read_text(encoding="utf-8"))
    return create_split_manifest(root, **kwargs)


def split_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    image_counts = Counter(row["split"] for row in rows)
    fruit_sets: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        fruit_sets[row["split"]].add(row.get("group_id") or row.get("fruit_id") or f"pilot:{row['image_id']}")
    return {
        split: {"images": image_counts[split], "physical_fruits": len(fruit_sets[split])}
        for split in SPLITS
    }
