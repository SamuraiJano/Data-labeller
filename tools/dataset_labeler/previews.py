from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .core import annotations_for_image, class_by_id, image_abs_path, load_project
from .geometry import coerce_points


def hex_to_rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), alpha


def draw_overlay(project: dict[str, Any], image: dict[str, Any], out_path: Path, opacity: float = 0.35) -> None:
    src = image_abs_path(project, image)
    with Image.open(src) as im:
        base = im.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke = ImageDraw.Draw(base)
    alpha = max(0, min(255, int(255 * opacity)))
    for ann in annotations_for_image(project, int(image["id"])):
        cls = class_by_id(int(ann["category_id"]), project)
        points = coerce_points(ann.get("points") or ann["segmentation"][0])
        color = hex_to_rgba(cls["color"], alpha)
        line_color = hex_to_rgba(cls["color"], 255)
        draw.polygon(points, fill=color)
        stroke.line(points + [points[0]], fill=line_color, width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).convert("RGB").save(out_path, quality=92)


def generate_previews(
    project_dir: Path | str,
    output_dir: Path | str,
    count: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    project = load_project(project_dir)
    out_dir = Path(output_dir).expanduser().resolve()
    candidates = [img for img in project["images"] if not img.get("excluded")]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[: max(0, count)]
    rows = []
    for image in selected:
        out = out_dir / f"preview_{int(image['id']):06d}_{Path(image['relative_path']).stem}.jpg"
        draw_overlay(project, image, out)
        rows.append({"image_id": int(image["id"]), "source": image["relative_path"], "preview": str(out)})
    manifest = out_dir / "preview_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    return {"count": len(rows), "manifest": str(manifest), "output_dir": str(out_dir)}
