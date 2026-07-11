from __future__ import annotations

import math
from collections.abc import Iterable

Point = tuple[float, float]


def coerce_points(points: Iterable[object]) -> list[Point]:
    out: list[Point] = []
    for point in points:
        if isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")
        else:
            pair = list(point)  # type: ignore[arg-type]
            if len(pair) != 2:
                raise ValueError("Each point must contain exactly two coordinates.")
            x, y = pair
        out.append((float(x), float(y)))
    return out


def flatten_points(points: Iterable[Point]) -> list[float]:
    flat: list[float] = []
    for x, y in points:
        flat.extend([float(x), float(y)])
    return flat


def unflatten_points(values: Iterable[float]) -> list[Point]:
    nums = [float(v) for v in values]
    if len(nums) % 2 != 0:
        raise ValueError("Polygon coordinate list has an odd number of values.")
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def distinct_points(points: Iterable[Point], precision: int = 6) -> set[Point]:
    return {(round(x, precision), round(y, precision)) for x, y in points}


def polygon_area(points: Iterable[Point]) -> float:
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    acc = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


def polygon_bbox(points: Iterable[Point]) -> list[float]:
    pts = list(points)
    if not pts:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = min(xs)
    y1 = min(ys)
    return [x1, y1, max(xs) - x1, max(ys) - y1]


def normalize_polygon(points: Iterable[Point], width: int, height: int) -> list[float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    values: list[float] = []
    for x, y in points:
        values.append(max(0.0, min(1.0, float(x) / float(width))))
        values.append(max(0.0, min(1.0, float(y) / float(height))))
    return values


def denormalize_polygon(values: Iterable[float], width: int, height: int) -> list[Point]:
    nums = [float(v) for v in values]
    if len(nums) % 2 != 0:
        raise ValueError("YOLO polygon row has an odd number of coordinates.")
    return [(nums[i] * width, nums[i + 1] * height) for i in range(0, len(nums), 2)]


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _on_segment(a1, b1, a2))
        or (o2 == 0 and _on_segment(a1, b2, a2))
        or (o3 == 0 and _on_segment(b1, a1, b2))
        or (o4 == 0 and _on_segment(b1, a2, b2))
    )


def is_self_intersecting(points: Iterable[Point]) -> bool:
    pts = list(points)
    n = len(pts)
    if n < 4:
        return False
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i, (a1, a2) in enumerate(edges):
        for j, (b1, b2) in enumerate(edges):
            if abs(i - j) <= 1 or {i, j} == {0, n - 1}:
                continue
            if segments_intersect(a1, a2, b1, b2):
                return True
    return False


def validate_polygon(points: Iterable[Point], width: int, height: int, min_area: float = 4.0) -> list[str]:
    errors: list[str] = []
    pts = list(points)
    if len(distinct_points(pts)) < 3:
        errors.append("polygon has fewer than three distinct vertices")
    for x, y in pts:
        if not (math.isfinite(x) and math.isfinite(y)):
            errors.append("polygon contains non-finite coordinates")
            break
        if x < 0 or y < 0 or x > width or y > height:
            errors.append("polygon has points outside image boundaries")
            break
    if polygon_area(pts) <= 0:
        errors.append("polygon has zero area")
    elif polygon_area(pts) < min_area:
        errors.append("polygon area is extremely small")
    if is_self_intersecting(pts):
        errors.append("polygon is self-intersecting")
    return errors


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    x, y = point
    pts = list(polygon)
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside
