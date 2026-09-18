from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .core import atomic_write_json, image_abs_path, load_project
from .mask_ops import load_binary_mask


def _rgb(value: str) -> tuple[int, int, int]:
    raw = value.lstrip("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def draw_overlay(project: dict[str, Any], image: dict[str, Any], out_path: Path, opacity: float = 0.35) -> None:
    with Image.open(image_abs_path(project, image)) as opened:
        base = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.float32).copy()
    boxes: list[tuple[list[float], tuple[int, int, int]]] = []
    for instance in project.get("annotations", []):
        if str(instance["image_id"]) != str(image["id"]) or not instance.get("visible", True):
            continue
        color = np.asarray(_rgb(instance.get("color", "#5aa7d8")), dtype=np.float32)
        if instance.get("annotation_kind") == "fruit_box":
            boxes.append((instance["bbox"], tuple(int(value) for value in color)))
            continue
        mask = load_binary_mask(project, instance)
        base[mask] = base[mask] * (1.0 - opacity) + color * opacity
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(rendered)
    for (x, y, width, height), color in boxes:
        draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
    rendered.save(out_path, quality=92)


def generate_previews(project_dir: Path | str, output_dir: Path | str, count: int = 25, seed: int = 42) -> dict[str, Any]:
    project = load_project(project_dir)
    destination = Path(output_dir).expanduser().resolve()
    candidates = [image for image in project["images"] if not image.get("excluded")]
    random.Random(seed).shuffle(candidates)
    rows = []
    for image in candidates[: max(0, count)]:
        output = destination / f"preview_{image['id']}_{Path(image['relative_path']).stem}.jpg"
        draw_overlay(project, image, output)
        rows.append({"image_id": image["id"], "source": image["relative_path"], "preview": str(output)})
    manifest = destination / "preview_manifest.json"
    atomic_write_json(manifest, {"rows": rows})
    return {"count": len(rows), "manifest": str(manifest), "output_dir": str(destination)}
