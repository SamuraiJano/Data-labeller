from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import infer_capture_group, load_project, save_project

SPLITS = ("train", "val", "test")


def _manifest_paths(project_dir: Path) -> tuple[Path, Path]:
    manifest_dir = project_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / "split_manifest.json", manifest_dir / "split_manifest.csv"


def _group_key(image: dict[str, Any], duplicate_hashes: set[str]) -> str:
    digest = image.get("sha256")
    if digest and digest in duplicate_hashes:
        return f"sha256:{digest}"
    for field in ("fruit_id", "tree_id", "session_id", "capture_group"):
        value = str(image.get(field) or "").strip()
        if value:
            return f"{field}:{value.lower()}"
    return f"inferred:{infer_capture_group(image['relative_path'])}"


def _target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    train = round(total * float(ratios.get("train", 0.70)))
    val = round(total * float(ratios.get("val", 0.15)))
    test = total - train - val
    if total >= 3:
        if val == 0:
            val = 1
            train = max(1, train - 1)
        if test == 0:
            test = 1
            train = max(1, train - 1)
    return {"train": train, "val": val, "test": test}


def _assign_groups(groups: list[dict[str, Any]], ratios: dict[str, float], seed: int) -> dict[str, str]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_class[str(group["primary_class"])].append(group)
    assignments: dict[str, str] = {}
    for cls, cls_groups in sorted(by_class.items()):
        rng = random.Random(f"{seed}:{cls}")
        shuffled = cls_groups[:]
        rng.shuffle(shuffled)
        targets = _target_counts(len(shuffled), ratios)
        order: list[str] = []
        for split in SPLITS:
            order.extend([split] * targets[split])
        while len(order) < len(shuffled):
            order.append("train")
        for group, split in zip(shuffled, order, strict=False):
            assignments[group["group_key"]] = split
    return assignments


def create_split_manifest(
    project_dir: Path | str,
    ratios: dict[str, float] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    project_dir = Path(project_dir).expanduser().resolve()
    project = load_project(project_dir)
    ratios = ratios or project.get("split", {}).get("ratios") or {"train": 0.70, "val": 0.15, "test": 0.15}
    seed = int(seed if seed is not None else project.get("split", {}).get("seed", 42))
    non_excluded = [img for img in project["images"] if not img.get("excluded")]
    hash_counts = Counter(img.get("sha256") for img in non_excluded if img.get("sha256"))
    duplicate_hashes = {digest for digest, count in hash_counts.items() if count > 1}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in non_excluded:
        grouped[_group_key(image, duplicate_hashes)].append(image)

    groups: list[dict[str, Any]] = []
    warnings: list[str] = []
    for key, images in grouped.items():
        labels = [img.get("source_class") or "unknown" for img in images]
        primary_class = Counter(labels).most_common(1)[0][0]
        if len(set(labels)) > 1:
            warnings.append(f"Group {key} contains multiple source classes: {sorted(set(labels))}")
        groups.append({"group_key": key, "images": images, "primary_class": primary_class})

    per_class = Counter(group["primary_class"] for group in groups)
    for cls, count in sorted(per_class.items()):
        if count < 3:
            warnings.append(f"Class {cls} has only {count} grouped example(s); a reliable train/val/test split is not possible.")

    assignments = _assign_groups(groups, ratios, seed)
    rows: list[dict[str, Any]] = []
    for key, images in sorted(grouped.items()):
        split = assignments[key]
        for image in sorted(images, key=lambda img: int(img["id"])):
            image["split"] = split
            rows.append(
                {
                    "image_id": int(image["id"]),
                    "relative_path": image["relative_path"],
                    "split": split,
                    "group_key": key,
                    "sha256": image.get("sha256", ""),
                    "source_class": image.get("source_class") or "",
                    "annotation_status": image.get("status", ""),
                }
            )

    json_path, csv_path = _manifest_paths(project_dir)
    payload = {
        "seed": seed,
        "ratios": ratios,
        "segmentation_definition": project["segmentation_definition"],
        "rows": rows,
        "warnings": warnings,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["image_id"])
        writer.writeheader()
        writer.writerows(rows)

    project["split"] = {
        "ratios": ratios,
        "seed": seed,
        "manifest_json": str(json_path),
        "manifest_csv": str(csv_path),
        "created_at": payload.get("created_at"),
    }
    save_project(project, project_dir)
    payload["manifest_json"] = str(json_path)
    payload["manifest_csv"] = str(csv_path)
    return payload


def ensure_split_manifest(project_dir: Path | str, force: bool = False) -> dict[str, Any]:
    project_dir = Path(project_dir).expanduser().resolve()
    json_path, _ = _manifest_paths(project_dir)
    if json_path.exists() and not force:
        return json.loads(json_path.read_text(encoding="utf-8"))
    return create_split_manifest(project_dir)


def split_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for row in rows:
        split = row["split"]
        cls = row.get("source_class") or "unknown"
        counts[split][cls] = counts[split].get(cls, 0) + 1
    return counts
