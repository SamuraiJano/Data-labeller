from __future__ import annotations

import hashlib
import math
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2

from .core import (
    CLASS_DEFINITIONS,
    DEFAULT_SEGMENTATION_CLASS_NAMES,
    PROJECT_FILENAME,
    SEGMENTATION_CLASS_DEFINITIONS,
    atomic_write_bytes,
    atomic_write_json,
    create_or_update_project,
    load_project,
    project_summary,
    save_project,
    utc_now,
    validate_class_definitions,
)
from .splitting import create_split_manifest, split_counts

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mpeg", ".mpg", ".webm"}
VIDEO_IMPORT_MANIFEST = "video_import_manifest.json"
DEFAULT_FRAME_INTERVAL_SECONDS = 0.5
DEFAULT_JPEG_QUALITY = 95

ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


class VideoImportCancelled(RuntimeError):
    pass


def _cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check and cancel_check():
        raise VideoImportCancelled("Video import was cancelled.")


def _safe_slug(value: str, fallback: str = "video-dataset") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-_")
    return slug or fallback


def _sha256(path: Path, cancel_check: CancelCheck | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            _cancelled(cancel_check)
            digest.update(chunk)
    return digest.hexdigest()


def _open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return capture


def inspect_video(
    video_path: Path | str,
    *,
    interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    path = Path(video_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {path.suffix or '(none)'}")
    if interval_seconds <= 0:
        raise ValueError("Frame interval must be positive.")

    checksum = _sha256(path, cancel_check)
    capture = _open_capture(path)
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video.")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        ok, first_frame = capture.read()
        if not ok or first_frame is None or first_frame.size == 0:
            raise ValueError("The video does not contain a readable frame.")
        height, width = first_frame.shape[:2]
        duration_ms = (frame_count / fps * 1000.0) if fps > 0 and frame_count > 0 else 0.0
        expected_frames = max(1, int(math.ceil(duration_ms / (interval_seconds * 1000.0)))) if duration_ms > 0 else 1
        return {
            "video_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"data-labeller-video:{checksum}")),
            "file_name": path.name,
            "path": str(path),
            "extension": path.suffix.lower(),
            "sha256": checksum,
            "size_bytes": path.stat().st_size,
            "width": int(width),
            "height": int(height),
            "fps": fps,
            "source_frame_count": frame_count,
            "duration_ms": duration_ms,
            "expected_frames": expected_frames,
            "interval_ms": int(round(interval_seconds * 1000)),
        }
    finally:
        capture.release()


def inspect_videos(
    video_paths: list[Path | str],
    *,
    interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    videos: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_checksums: set[str] = set()
    for value in video_paths:
        _cancelled(cancel_check)
        try:
            inspected = inspect_video(value, interval_seconds=interval_seconds, cancel_check=cancel_check)
        except VideoImportCancelled:
            raise
        except Exception as exc:
            warnings.append(f"Skipped {Path(value).name or value}: {exc}")
            continue
        if inspected["sha256"] in seen_checksums:
            warnings.append(f"Skipped duplicate video {inspected['file_name']} ({inspected['sha256'][:12]}).")
            continue
        seen_checksums.add(inspected["sha256"])
        videos.append(inspected)
    return {
        "videos": videos,
        "warnings": warnings,
        "video_count": len(videos),
        "expected_frames": sum(int(item["expected_frames"]) for item in videos),
        "interval_ms": int(round(interval_seconds * 1000)),
    }


def _timestamp_ms(capture: cv2.VideoCapture, frame_index: int, fps: float) -> float:
    timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    fallback = (frame_index / fps * 1000.0) if fps > 0 else float(frame_index)
    if frame_index > 0 and timestamp <= 0:
        return fallback
    return timestamp


def _write_jpeg(path: Path, frame: Any, quality: int) -> None:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError(f"Could not encode extracted frame: {path.name}")
    atomic_write_bytes(path, encoded.tobytes(), backup=False)


def _extract_video(
    inspected: dict[str, Any],
    frames_root: Path,
    *,
    interval_seconds: float,
    jpeg_quality: int,
    completed_before: int,
    total_expected: int,
    progress_callback: ProgressCallback | None,
    cancel_check: CancelCheck | None,
) -> list[dict[str, Any]]:
    path = Path(inspected["path"])
    video_id = str(inspected["video_id"])
    folder = frames_root / f"{_safe_slug(path.stem, 'video')}-{inspected['sha256'][:10]}"
    folder.mkdir(parents=True, exist_ok=True)
    capture = _open_capture(path)
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not reopen this video for extraction.")

    interval_ms = interval_seconds * 1000.0
    duration_ms = float(inspected.get("duration_ms") or 0.0)
    fps = float(inspected.get("fps") or 0.0)
    extracted: list[dict[str, Any]] = []
    previous_frame = None
    previous_timestamp = 0.0
    previous_index = 0
    target_ms = 0.0
    frame_index = 0
    try:
        while True:
            _cancelled(cancel_check)
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp = _timestamp_ms(capture, frame_index, fps)
            if timestamp < previous_timestamp:
                timestamp = (frame_index / fps * 1000.0) if fps > 0 else previous_timestamp
            while target_ms <= timestamp + 1e-6 and (duration_ms <= 0 or target_ms < duration_ms - 1e-6):
                if previous_frame is not None and abs(target_ms - previous_timestamp) <= abs(timestamp - target_ms):
                    selected_frame = previous_frame
                    actual_timestamp = previous_timestamp
                    source_index = previous_index
                else:
                    selected_frame = frame
                    actual_timestamp = timestamp
                    source_index = frame_index
                relative = Path(folder.name) / f"frame-t{int(round(target_ms)):012d}.jpg"
                destination = frames_root / relative
                _write_jpeg(destination, selected_frame, jpeg_quality)
                record = {
                    "relative_path": relative.as_posix(),
                    "source_video_id": video_id,
                    "source_video_name": str(inspected["file_name"]),
                    "sample_time_ms": int(round(target_ms)),
                    "actual_time_ms": round(float(actual_timestamp), 3),
                    "source_frame_index": int(source_index),
                }
                extracted.append(record)
                if progress_callback:
                    completed = completed_before + len(extracted)
                    progress_callback({
                        "state": "running",
                        "stage": "extracting",
                        "completed": completed,
                        "total": max(total_expected, completed),
                        "percent": round(completed * 100 / max(1, total_expected)),
                        "message": f"Extracting {inspected['file_name']} at {target_ms / 1000.0:.1f}s",
                    })
                target_ms += interval_ms
            previous_frame = frame.copy()
            previous_timestamp = timestamp
            previous_index = frame_index
            frame_index += 1

        if not extracted and previous_frame is not None:
            relative = Path(folder.name) / "frame-t000000000000.jpg"
            _write_jpeg(frames_root / relative, previous_frame, jpeg_quality)
            extracted.append({
                "relative_path": relative.as_posix(),
                "source_video_id": video_id,
                "source_video_name": str(inspected["file_name"]),
                "sample_time_ms": 0,
                "actual_time_ms": round(float(previous_timestamp), 3),
                "source_frame_index": int(previous_index),
            })
        if not extracted:
            raise ValueError("No frames could be extracted.")
        return extracted
    finally:
        capture.release()


def create_class_definitions(names: list[str], namespace: str) -> list[dict[str, Any]]:
    cleaned = [str(name).strip() for name in names if str(name).strip()]
    if not cleaned:
        cleaned = ["Object"]
    palette = ["#42b883", "#5aa7d8", "#f2b84b", "#ef6a6a", "#a77bf3", "#4ecdc4", "#ff8f70"]
    values = []
    used_slugs: set[str] = set()
    for index, display_name in enumerate(cleaned):
        base = _safe_slug(display_name.lower(), f"class-{index + 1}").replace("-", "_").replace(".", "_")
        name = base
        suffix = 2
        while name in used_slugs:
            name = f"{base}_{suffix}"
            suffix += 1
        used_slugs.add(name)
        values.append({
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"data-labeller:{namespace}:{name}")),
            "id": index,
            "export_index": index,
            "name": name,
            "display_name": display_name,
            "meaning": None,
            "color": palette[index % len(palette)],
        })
    return validate_class_definitions(values)


def create_video_project(
    video_paths: list[Path | str],
    output_parent: Path | str,
    project_name: str,
    annotation_mode: str,
    class_names: list[str] | None = None,
    *,
    interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    if annotation_mode not in {"detection", "segmentation", "classification"}:
        raise ValueError("New video projects must use detection, segmentation, or classification mode.")
    if abs(float(interval_seconds) - DEFAULT_FRAME_INTERVAL_SECONDS) > 1e-9:
        raise ValueError("Video projects currently sample at the required fixed interval of 0.5 seconds.")
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")
    parent = Path(output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    project_slug = _safe_slug(project_name)
    target = parent / project_slug
    if target.exists():
        if any(target.iterdir()) if target.is_dir() else True:
            raise FileExistsError(f"Project destination is not empty: {target}")
        target.rmdir()
    staging = parent / f".{project_slug}.importing-{uuid.uuid4().hex[:8]}"
    if staging.exists():
        raise FileExistsError(f"Temporary import destination already exists: {staging}")

    warnings: list[str] = []
    try:
        if progress_callback:
            progress_callback({"state": "running", "stage": "inspecting", "percent": 0, "message": "Inspecting selected videos…"})
        inspected_result = inspect_videos(video_paths, interval_seconds=interval_seconds, cancel_check=cancel_check)
        inspected = inspected_result["videos"]
        warnings.extend(inspected_result["warnings"])
        if not inspected:
            raise ValueError("None of the selected videos could be decoded.")
        total_expected = int(inspected_result["expected_frames"])
        frames_root = staging / "frames"
        frames_root.mkdir(parents=True)
        frame_records: list[dict[str, Any]] = []
        imported_videos: list[dict[str, Any]] = []
        for video_number, video in enumerate(inspected, start=1):
            _cancelled(cancel_check)
            try:
                extracted = _extract_video(
                    video,
                    frames_root,
                    interval_seconds=interval_seconds,
                    jpeg_quality=jpeg_quality,
                    completed_before=len(frame_records),
                    total_expected=total_expected,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            except VideoImportCancelled:
                raise
            except Exception as exc:
                failed_folder = frames_root / f"{_safe_slug(Path(video['file_name']).stem, 'video')}-{video['sha256'][:10]}"
                if failed_folder.exists():
                    shutil.rmtree(failed_folder)
                warnings.append(f"Skipped {video['file_name']} during extraction: {exc}")
                continue
            frame_records.extend(extracted)
            imported_videos.append({
                key: video[key]
                for key in (
                    "video_id", "file_name", "extension", "sha256", "size_bytes", "width", "height",
                    "fps", "source_frame_count", "duration_ms", "expected_frames",
                )
            } | {"extracted_frames": len(extracted), "import_order": video_number})
        if not frame_records:
            raise ValueError("No frames could be extracted from the selected videos.")

        if progress_callback:
            progress_callback({"state": "running", "stage": "project", "message": "Creating the annotation project…"})
        project = create_or_update_project(frames_root, staging, project_name.strip() or project_slug)
        project["annotation_mode"] = annotation_mode
        project["source_kind"] = "video_frames"
        default_names = {
            "detection": ["Object"],
            "segmentation": DEFAULT_SEGMENTATION_CLASS_NAMES,
            "classification": ["Good", "Defective"],
        }
        active_classes = create_class_definitions(class_names or default_names[annotation_mode], f"{project_slug}:{annotation_mode}")
        if annotation_mode == "detection":
            project["detection_classes"] = active_classes
            project["classes"] = [dict(item) for item in active_classes]
            project.setdefault("segmentation_classes", [dict(item) for item in SEGMENTATION_CLASS_DEFINITIONS])
        elif annotation_mode == "segmentation":
            project["segmentation_classes"] = active_classes
            project.setdefault("detection_classes", [dict(item) for item in CLASS_DEFINITIONS])
            project["classes"] = [dict(item) for item in project["detection_classes"]]
        else:
            project["classification_classes"] = active_classes
            project.setdefault("detection_classes", [dict(item) for item in CLASS_DEFINITIONS])
            project.setdefault("segmentation_classes", [dict(item) for item in SEGMENTATION_CLASS_DEFINITIONS])
            project["classes"] = [dict(item) for item in project["detection_classes"]]
        records_by_path = {item["relative_path"]: item for item in frame_records}
        for image in project["images"]:
            metadata = records_by_path[str(image["relative_path"]).replace("\\", "/")]
            image.update(metadata)
            image["capture_group"] = metadata["source_video_id"]
            image["view_id"] = f"t{int(metadata['sample_time_ms']):012d}"
        manifest_relative = Path("manifests") / VIDEO_IMPORT_MANIFEST
        project["video_import"] = {
            "manifest": manifest_relative.as_posix(),
            "interval_seconds": interval_seconds,
            "jpeg_quality": jpeg_quality,
            "video_count": len(imported_videos),
            "frame_count": len(frame_records),
        }
        project["_dirty_annotation_image_ids"] = [str(image["id"]) for image in project["images"]]
        save_project(project, staging)
        manifest = {
            "schema_version": "1.0",
            "created_at": utc_now(),
            "project_name": project.get("project_name"),
            "annotation_mode": annotation_mode,
            "settings": {
                "interval_seconds": interval_seconds,
                "jpeg_quality": jpeg_quality,
                "full_resolution": True,
                "resize": None,
                "split_ratios": ratios or {"train": 0.7, "val": 0.2, "test": 0.1},
                "split_seed": int(seed),
            },
            "videos": imported_videos,
            "frames": frame_records,
            "warnings": warnings,
        }
        atomic_write_json(staging / manifest_relative, manifest, backup=False)
        split = create_split_manifest(staging, ratios or {"train": 0.7, "val": 0.2, "test": 0.1}, seed=seed)
        counts = split_counts(split["rows"])
        warnings.extend(message for message in split.get("warnings", []) if message not in warnings)
        if len(imported_videos) == 1:
            warning = "Only one source video is available; all frames are in training. Add two independent videos for validation and test sets."
            warnings.append(warning)
        elif len(imported_videos) == 2:
            warning = "Only two source videos are available; training and validation were created, but the test set is empty. Add another independent video for testing."
            warnings.append(warning)
        manifest["warnings"] = warnings
        atomic_write_json(staging / manifest_relative, manifest, backup=False)

        _cancelled(cancel_check)
        raw_project = load_project(staging)
        raw_project["project_dir"] = str(target)
        raw_project["source_root"] = str(target / "frames")
        raw_project["split"]["manifest_json"] = str(target / "manifests" / "split_manifest.json")
        raw_project["split"]["manifest_csv"] = str(target / "manifests" / "split_manifest.csv")
        payload = {key: value for key, value in raw_project.items() if key not in {"annotations", "_dirty_annotation_image_ids", "_legacy_source"}}
        atomic_write_json(staging / PROJECT_FILENAME, payload, backup=False)
        staging.replace(target)
        created = load_project(target)
        if progress_callback:
            progress_callback({
                "state": "complete", "stage": "complete", "completed": len(frame_records), "total": len(frame_records),
                "percent": 100, "message": "Video dataset is ready.",
            })
        return {
            "project_dir": str(target),
            "project": created,
            "summary": project_summary(created),
            "counts": {"videos": len(imported_videos), "frames": len(frame_records), "splits": counts},
            "warnings": warnings,
            "manifest": str(target / manifest_relative),
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
