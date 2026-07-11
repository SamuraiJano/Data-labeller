from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import yaml

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def convert_rgb_file_to_raw_hsv(src: Path, dst: Path) -> None:
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read RGB image: {src}")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), hsv):
        raise IOError(f"Could not write HSV image: {dst}")


def write_dataset_yaml(path: Path, root: Path, classes: list[dict[str, Any]]) -> None:
    data = {
        "path": root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(cls["id"]): cls["name"] for cls in sorted(classes, key=lambda c: int(c["id"]))},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def create_hsv_from_yolo_rgb(
    rgb_root: Path | str,
    hsv_root: Path | str,
    classes: list[dict[str, Any]],
    manifest_dir: Path | str | None = None,
) -> dict[str, Any]:
    rgb_root = Path(rgb_root).expanduser().resolve()
    hsv_root = Path(hsv_root).expanduser().resolve()
    manifest_root = Path(manifest_dir).expanduser().resolve() if manifest_dir else hsv_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    summary: dict[str, Any] = {"rgb_root": str(rgb_root), "hsv_root": str(hsv_root), "splits": {}}

    for split in SPLITS:
        src_images = rgb_root / "images" / split
        src_labels = rgb_root / "labels" / split
        dst_images = hsv_root / "images" / split
        dst_labels = hsv_root / "labels" / split
        if not src_images.exists():
            raise FileNotFoundError(f"Missing RGB image split: {src_images}")
        if not src_labels.exists():
            raise FileNotFoundError(f"Missing RGB label split: {src_labels}")
        image_count = 0
        for src in sorted(p for p in src_images.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            rel = src.relative_to(src_images)
            dst = dst_images / rel
            convert_rgb_file_to_raw_hsv(src, dst)
            rows.append(
                {
                    "split": split,
                    "rgb_image": str(src.relative_to(rgb_root)).replace("\\", "/"),
                    "hsv_image": str(dst.relative_to(hsv_root)).replace("\\", "/"),
                    "label": str((dst_labels / rel.with_suffix(".txt")).relative_to(hsv_root)).replace("\\", "/"),
                    "hsv_storage": "OpenCV raw HSV channels written as image bytes; H range 0..179, S/V range 0..255.",
                }
            )
            image_count += 1
        label_count = 0
        for label in sorted(p for p in src_labels.rglob("*.txt") if p.is_file()):
            rel = label.relative_to(src_labels)
            out = dst_labels / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label, out)
            label_count += 1
        summary["splits"][split] = {"images": image_count, "labels": label_count}

    write_dataset_yaml(hsv_root / "data.yaml", hsv_root, classes)
    csv_path = manifest_root / "rgb_hsv_mapping.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["split", "rgb_image", "hsv_image", "label"])
        writer.writeheader()
        writer.writerows(rows)
    json_path = manifest_root / "rgb_hsv_mapping.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    summary["mapping_csv"] = str(csv_path)
    summary["mapping_json"] = str(json_path)
    summary["data_yaml"] = str(hsv_root / "data.yaml")
    return summary
