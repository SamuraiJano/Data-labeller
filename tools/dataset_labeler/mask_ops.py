from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .core import safe_resolve
from .geometry import is_self_intersecting


def load_binary_mask(project: dict[str, Any], instance: dict[str, Any]) -> np.ndarray:
    path = safe_resolve(Path(project["project_dir"]), instance["mask_path"])
    if not path.exists():
        raise FileNotFoundError(f"Mask file is missing: {instance['mask_path']}")
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def rasterize_polygon(points: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        contour = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(output, [contour], 1)
    return output.astype(bool)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if not union:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def _deduplicate(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for point in points:
        if not result or point != result[-1]:
            result.append(point)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return result


def canonicalize_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    points = _deduplicate(points)
    if len(points) < 3:
        return points
    signed_twice_area = sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if signed_twice_area < 0:
        points = list(reversed(points))
    start = min(range(len(points)), key=lambda index: (points[index][1], points[index][0]))
    return points[start:] + points[:start]


def mask_to_polygon(
    mask: np.ndarray,
    simplification_pixels: float = 0.5,
) -> dict[str, Any]:
    binary = np.where(mask, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Mask is empty.")
    hierarchy_rows = hierarchy[0] if hierarchy is not None else np.full((len(contours), 4), -1)
    external_indices = [index for index, row in enumerate(hierarchy_rows) if int(row[3]) == -1]
    hole_count = sum(1 for row in hierarchy_rows if int(row[3]) != -1)
    external_indices.sort(key=lambda index: (-abs(cv2.contourArea(contours[index])), index))
    selected = contours[external_indices[0]]
    epsilon = max(0.0, float(simplification_pixels))
    simplified = cv2.approxPolyDP(selected, epsilon, True) if epsilon else selected
    points = canonicalize_polygon([(float(item[0][0]), float(item[0][1])) for item in simplified])
    if len(set(points)) < 3:
        points = canonicalize_polygon([(float(item[0][0]), float(item[0][1])) for item in selected])
    if len(set(points)) < 3:
        raise ValueError("Mask contour has fewer than three unique points.")
    height, width = mask.shape
    rasterized = rasterize_polygon(points, width, height)
    warnings: list[str] = []
    if len(external_indices) > 1:
        warnings.append(
            f"mask has {len(external_indices)} disconnected components; YOLO export keeps the largest component"
        )
    if hole_count:
        warnings.append(f"mask has {hole_count} hole(s); YOLO export fills holes")
    if bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()):
        warnings.append("mask touches the image boundary")
    area = int(mask.sum())
    if area < 16:
        warnings.append(f"mask is tiny ({area} pixels)")
    if is_self_intersecting(points):
        warnings.append("exported polygon is self-intersecting")
    iou = mask_iou(mask, rasterized)
    return {
        "points": points,
        "iou": iou,
        "warnings": warnings,
        "source_area": area,
        "exported_area": int(rasterized.sum()),
        "components": len(external_indices),
        "holes": hole_count,
    }


def mask_to_polygons(mask: np.ndarray, simplification_pixels: float = 0.5) -> list[dict[str, Any]]:
    """Convert every disconnected foreground component to its own polygon.

    YOLO segmentation has one instance per row. Returning one polygon per
    component prevents disconnected defects from being joined into an invalid
    bridge or silently discarded.
    """
    binary = np.where(mask, 255, 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    results: list[dict[str, Any]] = []
    for label in range(1, count):
        component = labels == label
        if component.any():
            results.append(mask_to_polygon(component, simplification_pixels))
    results.sort(key=lambda item: (-int(item["source_area"]), item["points"][0]))
    return results


def normalized_yolo_row(
    training_class_id: int,
    points: list[tuple[float, float]],
    width: int,
    height: int,
    precision: int = 6,
) -> str:
    if training_class_id < 0:
        raise ValueError("Training class ID cannot be negative.")
    if len(set(points)) < 3:
        raise ValueError("Polygon has fewer than three unique points.")
    values: list[str] = []
    previous: tuple[float, float] | None = None
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("Polygon contains a non-finite coordinate.")
        normalized = (
            max(0.0, min(1.0, x / float(width))),
            max(0.0, min(1.0, y / float(height))),
        )
        if previous == normalized:
            continue
        previous = normalized
        values.extend([f"{normalized[0]:.{precision}f}", f"{normalized[1]:.{precision}f}"])
    if len(values) < 6:
        raise ValueError("Polygon has fewer than three unique normalized vertices.")
    return " ".join([str(int(training_class_id)), *values])
