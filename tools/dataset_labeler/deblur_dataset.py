"""Paired synthetic motion-blur dataset generation.

The blur synthesis is original project code informed by common camera-motion PSF
simulation.  It does not vendor an external deblurring implementation.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .core import IMAGE_EXTENSIONS, atomic_write_bytes, atomic_write_json, atomic_write_text, average_perceptual_hash, sha256_file, utc_now

DEBLUR_SCHEMA_VERSION = "1.0"
SPLITS = ("train", "val", "test")
BLUR_TYPES = ("horizontal", "diagonal", "tangential", "spatially_varying")
ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class SeverityRange:
    path_length: tuple[int, int]
    curvature: tuple[float, float]
    exposure: tuple[float, float]
    brightness: tuple[float, float]
    contrast: tuple[float, float]
    noise_sigma: tuple[float, float]


@dataclass(frozen=True)
class DeblurGenerationConfig:
    size: int = 256
    variants: int = 4
    seed: int = 42
    ratios: dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.2, "test": 0.1})
    enabled_blur_types: tuple[str, ...] = BLUR_TYPES
    blur_probabilities: dict[str, float] = field(default_factory=lambda: {
        "horizontal": 0.35,
        "diagonal": 0.25,
        "tangential": 0.20,
        "spatially_varying": 0.20,
    })
    moderate_probability: float = 0.65
    moderate: SeverityRange = field(default_factory=lambda: SeverityRange(
        path_length=(5, 13), curvature=(0.02, 0.16), exposure=(-0.025, 0.025),
        brightness=(-0.018, 0.018), contrast=(0.97, 1.03), noise_sigma=(0.001, 0.005),
    ))
    severe: SeverityRange = field(default_factory=lambda: SeverityRange(
        path_length=(15, 31), curvature=(0.12, 0.42), exposure=(-0.04, 0.04),
        brightness=(-0.025, 0.025), contrast=(0.95, 1.05), noise_sigma=(0.002, 0.009),
    ))
    resize_mode: str = "cover"

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None = None) -> "DeblurGenerationConfig":
        value = dict(value or {})
        moderate = _severity_from_mapping(value.pop("moderate", None), cls().moderate)
        severe = _severity_from_mapping(value.pop("severe", None), cls().severe)
        enabled = tuple(value.pop("enabled_blur_types", BLUR_TYPES))
        probabilities = dict(cls().blur_probabilities)
        probabilities.update(value.pop("blur_probabilities", {}) or {})
        ratios = dict(cls().ratios)
        ratios.update(value.pop("ratios", {}) or {})
        return cls(
            size=int(value.pop("size", 256)), variants=int(value.pop("variants", 4)),
            seed=int(value.pop("seed", 42)), ratios={key: float(ratios.get(key, 0)) for key in SPLITS},
            enabled_blur_types=enabled, blur_probabilities={key: float(probabilities.get(key, 0)) for key in BLUR_TYPES},
            moderate_probability=float(value.pop("moderate_probability", 0.65)), moderate=moderate, severe=severe,
            resize_mode=str(value.pop("resize_mode", "cover")),
        )

    def validate(self) -> None:
        if self.size < 32 or self.size > 4096:
            raise ValueError("Image size must be between 32 and 4096 pixels.")
        if self.variants < 1 or self.variants > 1000:
            raise ValueError("Variants must be between 1 and 1000.")
        if abs(sum(self.ratios.values()) - 1.0) > 1e-6 or any(value < 0 for value in self.ratios.values()):
            raise ValueError("Train, validation and test ratios must be non-negative and sum to 1.0.")
        unknown = set(self.enabled_blur_types) - set(BLUR_TYPES)
        if unknown or not self.enabled_blur_types:
            raise ValueError(f"At least one supported blur type must be enabled; unknown: {sorted(unknown)}")
        weights = [self.blur_probabilities.get(name, 0.0) for name in self.enabled_blur_types]
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Enabled blur probabilities must be non-negative and have a positive sum.")
        if not 0 <= self.moderate_probability <= 1:
            raise ValueError("Moderate probability must be between 0 and 1.")
        for name, ranges in (("moderate", self.moderate), ("severe", self.severe)):
            if ranges.path_length[0] < 1 or ranges.path_length[0] > ranges.path_length[1]:
                raise ValueError(f"Invalid {name} path-length range.")
            if ranges.path_length[1] >= self.size:
                raise ValueError(f"{name.title()} path length must be smaller than the image size.")
            for range_name in ("curvature", "exposure", "brightness", "contrast", "noise_sigma"):
                low, high = getattr(ranges, range_name)
                if low > high:
                    raise ValueError(f"Invalid {name} {range_name} range.")
        if self.resize_mode not in {"cover", "contain"}:
            raise ValueError("Resize mode must be 'cover' or 'contain'.")


def _severity_from_mapping(value: Any, default: SeverityRange) -> SeverityRange:
    raw = dict(value or {})
    return SeverityRange(**{
        name: tuple(raw.get(name, getattr(default, name)))
        for name in ("path_length", "curvature", "exposure", "brightness", "contrast", "noise_sigma")
    })


def config_payload(config: DeblurGenerationConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["enabled_blur_types"] = list(config.enabled_blur_types)
    return payload


def _config_fingerprint(config: DeblurGenerationConfig, source_dir: Path) -> str:
    payload = {"schema_version": DEBLUR_SCHEMA_VERSION, "source_dir": str(source_dir), "config": config_payload(config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _metadata_rows(source_root: Path) -> list[dict[str, Any]]:
    candidates = [
        source_root / "source_groups.json", source_root / "source_groups.csv",
        source_root / "manifests" / "split_manifest.json", source_root.parent / "manifests" / "split_manifest.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if path.suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    return list(csv.DictReader(handle))
            data = json.loads(path.read_text(encoding="utf-8"))
            return list(data if isinstance(data, list) else data.get("rows", []))
        except Exception as exc:
            raise ValueError(f"Could not read grouping metadata {path}: {exc}") from exc
    return []


def _source_records(source_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not source_root.is_dir():
        raise NotADirectoryError(f"Sharp-image source folder does not exist: {source_root}")
    paths = sorted(
        (path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(source_root).as_posix().casefold(),
    )
    metadata = _metadata_rows(source_root)
    by_path = {str(row.get("relative_path") or row.get("source_path") or "").replace("\\", "/").casefold(): row for row in metadata}
    by_hash = {str(row.get("sha256") or row.get("source_hash") or ""): row for row in metadata if row.get("sha256") or row.get("source_hash")}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        relative = path.relative_to(source_root).as_posix()
        try:
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
            digest = sha256_file(path)
            row = by_path.get(relative.casefold()) or by_hash.get(digest) or {}
            group_fields = {
                key: str(row.get(key) or "").strip()
                for key in ("capture_group", "fruit_id", "tree_id", "session_id")
            }
            records.append({
                "path": path, "relative_path": relative, "sha256": digest,
                "perceptual_hash": average_perceptual_hash(path), **group_fields,
            })
        except Exception as exc:
            failures.append({"source_path": str(path), "error": str(exc)})
    if not records:
        detail = f" ({len(failures)} unreadable file(s))" if failures else ""
        raise ValueError(f"No readable source images were found in {source_root}{detail}.")
    return records, failures


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _leakage_groups(records: list[dict[str, Any]]) -> list[list[int]]:
    union = _UnionFind(len(records))
    indexes: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        identities = [("sha256", record["sha256"]), ("perceptual_hash", record["perceptual_hash"])]
        identities.extend((key, record[key].casefold()) for key in ("capture_group", "fruit_id", "tree_id", "session_id") if record[key])
        for identity in identities:
            if identity in indexes:
                union.union(index, indexes[identity])
            else:
                indexes[identity] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union.find(index)].append(index)
    return sorted(groups.values(), key=lambda group: min(records[index]["relative_path"].casefold() for index in group))


def _target_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratios[name] for name in SPLITS}
    result = {name: int(raw[name]) for name in SPLITS}
    for name in sorted(SPLITS, key=lambda item: (-(raw[item] - result[item]), SPLITS.index(item)))[: total - sum(result.values())]:
        result[name] += 1
    return result


def _assign_splits(records: list[dict[str, Any]], config: DeblurGenerationConfig) -> dict[int, str]:
    groups = _leakage_groups(records)
    rng = np.random.default_rng(config.seed)
    order = list(rng.permutation(len(groups)))
    shuffled_rank = {group_index: rank for rank, group_index in enumerate(order)}
    order.sort(key=lambda index: (-len(groups[index]), shuffled_rank[index]))
    targets = _target_counts(len(records), config.ratios)
    counts = {name: 0 for name in SPLITS}
    assignments: dict[int, str] = {}
    for group_index in order:
        group = groups[group_index]
        split = max(SPLITS, key=lambda name: (targets[name] - counts[name], config.ratios[name], -SPLITS.index(name)))
        for record_index in group:
            assignments[record_index] = split
        counts[split] += len(group)
    return assignments


def _prepare_sharp(path: Path, size: int, mode: str) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if mode == "cover":
            image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        else:
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            background = Image.new("RGB", (size, size), tuple(int(value) for value in np.asarray(image).reshape(-1, 3).mean(axis=0)))
            background.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
            image = background
    return np.asarray(image, dtype=np.float32) / 255.0


def _splat(kernel: np.ndarray, x: float, y: float, weight: float = 1.0) -> None:
    x0, y0 = math.floor(x), math.floor(y)
    for px, py, amount in (
        (x0, y0, (x0 + 1 - x) * (y0 + 1 - y)), (x0 + 1, y0, (x - x0) * (y0 + 1 - y)),
        (x0, y0 + 1, (x0 + 1 - x) * (y - y0)), (x0 + 1, y0 + 1, (x - x0) * (y - y0)),
    ):
        if 0 <= py < kernel.shape[0] and 0 <= px < kernel.shape[1]:
            kernel[py, px] += float(weight * amount)


def motion_kernel(path_length: int, angle_degrees: float, curvature: float = 0.0) -> tuple[np.ndarray, list[list[float]]]:
    size = max(3, int(path_length) | 1)
    centre = (size - 1) / 2.0
    steps = max(16, int(path_length) * 6)
    distance = np.linspace(-path_length / 2.0, path_length / 2.0, steps)
    radians = math.radians(angle_degrees)
    normal = np.asarray([-math.sin(radians), math.cos(radians)])
    tangent = np.asarray([math.cos(radians), math.sin(radians)])
    curve = curvature * ((distance / max(path_length / 2.0, 1.0)) ** 2 - 0.35) * path_length * 0.30
    points = np.asarray([centre, centre]) + distance[:, None] * tangent + curve[:, None] * normal
    kernel = np.zeros((size, size), dtype=np.float32)
    for x, y in points:
        _splat(kernel, float(x), float(y))
    total = float(kernel.sum())
    if total <= 0:
        kernel[size // 2, size // 2] = 1.0
    else:
        kernel /= total
    trajectory = [[round(float(x - centre), 4), round(float(y - centre), 4)] for x, y in points[:: max(1, steps // 12)]]
    return kernel, trajectory


def _filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def _blur_parameters(rng: np.random.Generator, config: DeblurGenerationConfig) -> dict[str, Any]:
    names = list(config.enabled_blur_types)
    probabilities = np.asarray([config.blur_probabilities[name] for name in names], dtype=np.float64)
    probabilities /= probabilities.sum()
    blur_type = str(rng.choice(names, p=probabilities))
    severity = "moderate" if rng.random() < config.moderate_probability else "severe"
    ranges = config.moderate if severity == "moderate" else config.severe
    length = int(rng.integers(ranges.path_length[0], ranges.path_length[1] + 1))
    if length % 2 == 0:
        length += 1 if length < ranges.path_length[1] else -1
    if blur_type == "horizontal":
        angle = float(rng.uniform(-2.5, 2.5))
        curvature = 0.0
    elif blur_type == "diagonal":
        angle = float(rng.choice([-1.0, 1.0]) * rng.uniform(7.0, 22.0))
        curvature = 0.0
    else:
        angle = float(rng.uniform(-35.0, 35.0))
        curvature = float(rng.choice([-1.0, 1.0]) * rng.uniform(*ranges.curvature))
    return {
        "blur_type": blur_type, "severity": severity, "path_length": length,
        "angle_degrees": angle, "curvature": curvature,
        "exposure_ev": float(rng.uniform(*ranges.exposure)),
        "brightness": float(rng.uniform(*ranges.brightness)),
        "contrast": float(rng.uniform(*ranges.contrast)),
        "noise_sigma": float(rng.uniform(*ranges.noise_sigma)),
    }


def apply_motion_blur(sharp: np.ndarray, parameters: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    kernel, trajectory = motion_kernel(parameters["path_length"], parameters["angle_degrees"], parameters["curvature"])
    spatial: dict[str, Any] = {"enabled": False}
    if parameters["blur_type"] == "spatially_varying":
        alternate_length = max(3, int(round(parameters["path_length"] * rng.uniform(0.65, 1.20))) | 1)
        alternate_angle = float(parameters["angle_degrees"] + rng.uniform(-18, 18))
        alternate_curvature = float(parameters["curvature"] * rng.uniform(0.55, 1.35))
        alternate_kernel, alternate_trajectory = motion_kernel(alternate_length, alternate_angle, alternate_curvature)
        first, second = _filter(sharp, kernel), _filter(sharp, alternate_kernel)
        height, width = sharp.shape[:2]
        centre_x, centre_y = rng.uniform(0.25, 0.75, size=2)
        direction = float(rng.uniform(0, math.tau))
        yy, xx = np.mgrid[0:height, 0:width]
        signed = ((xx / width - centre_x) * math.cos(direction) + (yy / height - centre_y) * math.sin(direction))
        softness = float(rng.uniform(0.10, 0.24))
        mask = np.clip(0.5 + signed / softness, 0, 1).astype(np.float32)[..., None]
        blurred = first * mask + second * (1.0 - mask)
        spatial = {
            "enabled": True, "mask_centre": [float(centre_x), float(centre_y)], "mask_direction_radians": direction,
            "mask_softness": softness, "secondary_angle_degrees": alternate_angle,
            "secondary_path_length": alternate_length, "secondary_curvature": alternate_curvature,
            "secondary_trajectory": alternate_trajectory,
        }
    else:
        blurred = _filter(sharp, kernel)
    exposure_scale = 2.0 ** float(parameters["exposure_ev"])
    blurred = (blurred - 0.5) * float(parameters["contrast"]) + 0.5
    blurred = blurred * exposure_scale + float(parameters["brightness"])
    sigma = float(parameters["noise_sigma"])
    if sigma > 0:
        blurred += rng.normal(0.0, sigma, size=blurred.shape).astype(np.float32)
    return np.clip(blurred, 0.0, 1.0), {"trajectory": trajectory, "spatial_variation": spatial}


def laplacian_variance(image: np.ndarray) -> float:
    uint8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(uint8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())


def _png_bytes(image: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8), "RGB").save(output, "PNG", optimize=False)
    return output.getvalue()


def _write_pairs_manifests(root: Path, rows: list[dict[str, Any]]) -> None:
    manifests = root / "manifests"
    atomic_write_json(manifests / "pairs.json", {"schema_version": DEBLUR_SCHEMA_VERSION, "pairs": rows}, backup=False)
    fields = [
        "pair_id", "file_name", "source_path", "source_hash", "split", "capture_group", "fruit_id", "tree_id", "session_id",
        "random_seed", "blur_type", "severity", "motion_angle_degrees", "trajectory", "path_length", "spatial_variation",
        "exposure_settings", "brightness_contrast_settings", "noise_settings", "sharp_laplacian_variance",
        "blurred_laplacian_variance", "sharp_output_hash", "blurred_output_hash",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    atomic_write_text(manifests / "pairs.csv", output.getvalue())


def _existing_rows(root: Path, fingerprint: str, config: DeblurGenerationConfig) -> list[dict[str, Any]]:
    config_path = root / "manifests" / "generation_config.json"
    pairs_path = root / "manifests" / "pairs.json"
    if not config_path.is_file() or not pairs_path.is_file():
        raise ValueError("Resume requires existing generation_config.json and pairs.json manifests.")
    existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    if existing_config.get("fingerprint") != fingerprint:
        raise ValueError("Existing dataset configuration or source folder differs; resume would mix incompatible data.")
    rows = list(json.loads(pairs_path.read_text(encoding="utf-8")).get("pairs", []))
    seen: set[str] = set()
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if not pair_id or pair_id in seen:
            raise ValueError("Existing pairs manifest contains a missing or duplicate pair ID.")
        seen.add(pair_id)
        for kind, hash_key in (("blur", "blurred_output_hash"), ("sharp", "sharp_output_hash")):
            path = root / str(row["split"]) / kind / str(row["file_name"])
            if not path.is_file() or sha256_file(path) != row.get(hash_key):
                raise ValueError(f"Existing resume file is missing or has changed: {path}")
            with Image.open(path) as image:
                if image.size != (config.size, config.size) or image.mode != "RGB":
                    raise ValueError(f"Existing resume file has the wrong image format: {path}")
    return rows


def plan_deblur_dataset(source_dir: Path | str, config: DeblurGenerationConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    resolved_config = config if isinstance(config, DeblurGenerationConfig) else DeblurGenerationConfig.from_mapping(config)
    resolved_config.validate()
    source_root = Path(source_dir).expanduser().resolve()
    records, failures = _source_records(source_root)
    assignments = _assign_splits(records, resolved_config)
    source_counts = {name: sum(assignments[index] == name for index in range(len(records))) for name in SPLITS}
    pair_counts = {name: count * resolved_config.variants for name, count in source_counts.items()}
    estimated_bytes = len(records) * resolved_config.variants * resolved_config.size * resolved_config.size * 3 * 1.25
    return {
        "source_count": len(records), "source_failures": failures, "source_split_counts": source_counts,
        "split_counts": pair_counts, "total_pairs": sum(pair_counts.values()), "estimated_storage_bytes": int(estimated_bytes),
    }


def preview_deblur(source_dir: Path | str, config: DeblurGenerationConfig | dict[str, Any] | None = None, source_index: int = 0, variant: int = 0) -> dict[str, Any]:
    resolved_config = config if isinstance(config, DeblurGenerationConfig) else DeblurGenerationConfig.from_mapping(config)
    resolved_config.validate()
    source_root = Path(source_dir).expanduser().resolve()
    records, failures = _source_records(source_root)
    record = records[source_index % len(records)]
    pair_seed = _pair_seed(resolved_config.seed, record["sha256"], variant)
    rng = np.random.default_rng(pair_seed)
    sharp = _prepare_sharp(record["path"], resolved_config.size, resolved_config.resize_mode)
    parameters = _blur_parameters(rng, resolved_config)
    blurred, detail = apply_motion_blur(sharp, parameters, rng)
    difference = np.abs(sharp - blurred)
    scale = max(float(difference.max()), 1e-6)
    difference = np.clip(difference / scale, 0, 1)
    plan = plan_deblur_dataset(source_root, resolved_config)
    encode = lambda image: "data:image/png;base64," + base64.b64encode(_png_bytes(image)).decode("ascii")
    return {
        **plan, "source_path": record["relative_path"], "sharp_image": encode(sharp), "blurred_image": encode(blurred),
        "difference_image": encode(difference), "sharp_laplacian_variance": laplacian_variance(sharp),
        "blurred_laplacian_variance": laplacian_variance(blurred), "blur_parameters": {**parameters, **detail},
        "source_failures": failures,
    }


def _pair_seed(seed: int, source_hash: str, variant: int) -> int:
    digest = hashlib.sha256(f"{seed}:{source_hash}:{variant}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _safe_stem(relative_path: str) -> str:
    stem = Path(relative_path).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._") or "image"
    return cleaned[:80]


def generate_deblur_dataset(
    source_dir: Path | str,
    output_dir: Path | str,
    config: DeblurGenerationConfig | dict[str, Any] | None = None,
    *,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    resolved_config = config if isinstance(config, DeblurGenerationConfig) else DeblurGenerationConfig.from_mapping(config)
    resolved_config.validate()
    source_root = Path(source_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    records, source_failures = _source_records(source_root)
    assignments = _assign_splits(records, resolved_config)
    fingerprint = _config_fingerprint(resolved_config, source_root)
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"Output folder is not empty: {output_root}. Enable resume or choose a new folder.")
    if output_root.exists() and any(output_root.iterdir()) and resume and not (output_root / "manifests" / "generation_config.json").is_file():
        raise ValueError("Resume was requested for a non-empty folder without a generation_config.json manifest.")
    output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for member in ("blur", "sharp"):
            (output_root / split / member).mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    rows = _existing_rows(output_root, fingerprint, resolved_config) if resume and (output_root / "manifests" / "generation_config.json").exists() else []
    completed_ids = {str(row["pair_id"]) for row in rows}
    total = len(records) * resolved_config.variants
    generation_payload = {
        "schema_version": DEBLUR_SCHEMA_VERSION, "created_at": utc_now(), "updated_at": utc_now(),
        "source_dir": str(source_root), "output_dir": str(output_root), "fingerprint": fingerprint,
        "config": config_payload(resolved_config), "status": "running", "source_failures": source_failures,
    }
    atomic_write_json(output_root / "manifests" / "generation_config.json", generation_payload, backup=False)
    if not (output_root / "manifests" / "pairs.json").is_file():
        _write_pairs_manifests(output_root, rows)
    failures = list(source_failures)
    processed = len(rows)
    pending_manifest_rows = 0
    if progress_callback:
        progress_callback({"state": "running", "stage": "generating", "completed": processed, "total": total, "percent": round(processed * 100 / total), "message": "Generating aligned blur/sharp pairs."})
    cancelled = False
    for record_index, record in enumerate(records):
        split = assignments[record_index]
        for variant in range(resolved_config.variants):
            source_identity = hashlib.sha256(record["relative_path"].casefold().encode()).hexdigest()[:8]
            pair_id = f"{record['sha256'][:16]}-{source_identity}-v{variant:03d}"
            if pair_id in completed_ids:
                continue
            if cancel_check and cancel_check():
                cancelled = True
                break
            file_name = f"{_safe_stem(record['relative_path'])}-{pair_id}.png"
            try:
                pair_seed = _pair_seed(resolved_config.seed, record["sha256"], variant)
                rng = np.random.default_rng(pair_seed)
                sharp = _prepare_sharp(record["path"], resolved_config.size, resolved_config.resize_mode)
                parameters = _blur_parameters(rng, resolved_config)
                blurred, detail = apply_motion_blur(sharp, parameters, rng)
                sharp_path = output_root / split / "sharp" / file_name
                blurred_path = output_root / split / "blur" / file_name
                sharp_bytes, blurred_bytes = _png_bytes(sharp), _png_bytes(blurred)
                if sharp_path.exists() or blurred_path.exists():
                    if not resume or not sharp_path.is_file() or not blurred_path.is_file():
                        raise FileExistsError(f"Unmanifested or incomplete output already exists for {pair_id}; refusing to overwrite it.")
                    if sharp_path.read_bytes() != sharp_bytes or blurred_path.read_bytes() != blurred_bytes:
                        raise ValueError(f"Unmanifested resume output does not match deterministic regeneration for {pair_id}.")
                else:
                    atomic_write_bytes(sharp_path, sharp_bytes, backup=False)
                    try:
                        atomic_write_bytes(blurred_path, blurred_bytes, backup=False)
                    except Exception:
                        sharp_path.unlink(missing_ok=True)
                        raise
                row = {
                    "pair_id": pair_id, "file_name": file_name, "source_path": record["relative_path"],
                    "source_hash": record["sha256"], "perceptual_hash": record["perceptual_hash"], "split": split,
                    "capture_group": record["capture_group"], "fruit_id": record["fruit_id"], "tree_id": record["tree_id"],
                    "session_id": record["session_id"], "random_seed": pair_seed, "blur_type": parameters["blur_type"],
                    "severity": parameters["severity"], "motion_angle_degrees": parameters["angle_degrees"],
                    "trajectory": detail["trajectory"], "path_length": parameters["path_length"],
                    "spatial_variation": detail["spatial_variation"], "exposure_settings": {"ev": parameters["exposure_ev"]},
                    "brightness_contrast_settings": {"brightness": parameters["brightness"], "contrast": parameters["contrast"]},
                    "noise_settings": {"distribution": "gaussian_sensor_approximation", "sigma_0_1": parameters["noise_sigma"]},
                    "sharp_laplacian_variance": laplacian_variance(sharp), "blurred_laplacian_variance": laplacian_variance(blurred),
                    "sharp_output_hash": sha256_file(sharp_path), "blurred_output_hash": sha256_file(blurred_path),
                }
                rows.append(row)
                rows.sort(key=lambda item: (SPLITS.index(item["split"]), item["source_path"].casefold(), item["pair_id"]))
                pending_manifest_rows += 1
                if pending_manifest_rows >= 25:
                    _write_pairs_manifests(output_root, rows)
                    pending_manifest_rows = 0
                completed_ids.add(pair_id)
            except Exception as exc:
                failures.append({"source_path": record["relative_path"], "pair_id": pair_id, "error": str(exc)})
            processed += 1
            if progress_callback:
                progress_callback({
                    "state": "running", "stage": "generating", "completed": processed, "total": total,
                    "percent": round(processed * 100 / total), "message": f"Processed {processed} of {total} planned pairs.",
                    "current_file": record["relative_path"], "failure_count": len(failures),
                })
        if cancelled:
            break
    generation_payload.update({
        "updated_at": utc_now(), "status": "cancelled" if cancelled else ("complete_with_errors" if failures else "complete"),
        "generated_pairs": len(rows), "planned_pairs": total, "failures": failures,
    })
    atomic_write_json(output_root / "manifests" / "generation_config.json", generation_payload, backup=False)
    _write_pairs_manifests(output_root, rows)
    counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    return {
        "state": generation_payload["status"], "dataset_dir": str(output_root), "generated_pairs": len(rows),
        "planned_pairs": total, "split_counts": counts, "failure_count": len(failures), "failures": failures,
        "pairs_csv": str(output_root / "manifests" / "pairs.csv"), "pairs_json": str(output_root / "manifests" / "pairs.json"),
        "generation_config": str(output_root / "manifests" / "generation_config.json"),
    }


def validate_deblur_dataset(dataset_dir: Path | str) -> dict[str, Any]:
    root = Path(dataset_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    pairs_path = root / "manifests" / "pairs.json"
    config_path = root / "manifests" / "generation_config.json"
    if not pairs_path.is_file() or not config_path.is_file():
        return {"valid": False, "dataset_dir": str(root), "errors": ["Missing pairs.json or generation_config.json."], "warnings": []}
    try:
        rows = list(json.loads(pairs_path.read_text(encoding="utf-8")).get("pairs", []))
        config = DeblurGenerationConfig.from_mapping(json.loads(config_path.read_text(encoding="utf-8")).get("config", {}))
        config.validate()
    except Exception as exc:
        return {"valid": False, "dataset_dir": str(root), "errors": [f"Invalid manifest: {exc}"], "warnings": []}
    pair_ids: set[str] = set()
    source_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    required = {
        "pair_id", "source_path", "source_hash", "split", "random_seed", "blur_type", "severity", "trajectory",
        "path_length", "spatial_variation", "exposure_settings", "brightness_contrast_settings", "noise_settings",
        "sharp_laplacian_variance", "blurred_laplacian_variance", "sharp_output_hash", "blurred_output_hash",
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            errors.append(f"Pair {row.get('pair_id', '<unknown>')} is missing fields: {sorted(missing)}")
            continue
        pair_id = str(row["pair_id"])
        if pair_id in pair_ids:
            errors.append(f"Duplicate pair ID: {pair_id}")
        pair_ids.add(pair_id)
        split = str(row["split"])
        if split not in SPLITS:
            errors.append(f"Pair {pair_id} has invalid split {split!r}.")
            continue
        source_splits[("sha256", str(row["source_hash"]))].add(split)
        if row.get("perceptual_hash"):
            source_splits[("perceptual_hash", str(row["perceptual_hash"]))].add(split)
        for key in ("capture_group", "fruit_id", "tree_id", "session_id"):
            if row.get(key):
                group_splits[(key, str(row[key]).casefold())].add(split)
        for member, hash_key in (("blur", "blurred_output_hash"), ("sharp", "sharp_output_hash")):
            path = root / split / member / str(row.get("file_name", ""))
            if not path.is_file():
                errors.append(f"Missing {member} member for {pair_id}: {path}")
                continue
            if sha256_file(path) != row[hash_key]:
                errors.append(f"Hash mismatch for {path}")
            try:
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (config.size, config.size):
                        errors.append(f"Invalid RGB dimensions for {path}: {image.mode} {image.size}")
            except Exception as exc:
                errors.append(f"Invalid image {path}: {exc}")
    for identity, splits in {**source_splits, **group_splits}.items():
        if len(splits) > 1:
            errors.append(f"Split leakage for {identity[0]} {identity[1]} across {sorted(splits)}")
    manifest_names = {str(row.get("file_name", "")) for row in rows}
    for split in SPLITS:
        blur_names = {path.name for path in (root / split / "blur").glob("*") if path.is_file()}
        sharp_names = {path.name for path in (root / split / "sharp").glob("*") if path.is_file()}
        if blur_names != sharp_names:
            errors.append(f"Filename mismatch between {split}/blur and {split}/sharp.")
        extras = (blur_names | sharp_names) - manifest_names
        if extras:
            warnings.append(f"Unmanifested files in {split}: {sorted(extras)[:5]}")
    return {
        "valid": not errors, "dataset_dir": str(root), "pair_count": len(rows),
        "split_counts": {split: sum(row.get("split") == split for row in rows) for split in SPLITS},
        "errors": errors, "warnings": warnings,
    }
