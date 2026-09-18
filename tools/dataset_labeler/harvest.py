from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from .core import atomic_write_bytes, atomic_write_json, create_or_update_project, load_project, safe_resolve, save_project, utc_now
from .providers import (
    SAM2_DEFAULT_CHECKPOINT,
    SAM2_DEFAULT_CHECKPOINT_SHA256,
    SAM2_DEFAULT_CONFIG,
    SAM2_PINNED_REVISION,
    file_sha256,
)
from .splitting import create_split_manifest
from .video_import import VIDEO_EXTENSIONS, inspect_videos

HARVEST_INTERVAL_MS = 250
SESSION_FILENAME = "harvest-session.json"
MANIFEST_FILENAME = "harvest-manifest.json"
TRACK_COLORS = ["#65d1ff", "#ff6b73", "#f2b84b", "#a77bf3", "#42b883", "#ff8bd1"]
LEGACY_BOUNDARY_SKIP_REASON = "fruit touches the source-frame boundary"


class HarvestCancelled(RuntimeError):
    pass


def _cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check and cancel_check():
        raise HarvestCancelled("Harvest operation was cancelled.")


def _slug(value: str, fallback: str = "naartjie-harvest") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip(".-_")
    return cleaned or fallback


def snap_time_ms(value: float | int, duration_ms: float | int | None = None) -> int:
    snapped = max(0, int(math.floor((float(value) + HARVEST_INTERVAL_MS / 2) / HARVEST_INTERVAL_MS)) * HARVEST_INTERVAL_MS)
    if duration_ms is not None and float(duration_ms) > 0:
        # Video duration is the exclusive end timestamp. A sample exactly at
        # that value has no frame (for example frame 20 in a 20 FPS, 1 s clip).
        final_sample = max(0, int(math.floor((float(duration_ms) - 1e-6) / HARVEST_INTERVAL_MS)) * HARVEST_INTERVAL_MS)
        snapped = min(snapped, final_sample)
    return snapped


def _session_path(root: Path) -> Path:
    return root / SESSION_FILENAME


def _mask_path(root: Path, track_id: str, time_ms: int) -> Path:
    return root / ".work" / "masks" / track_id / f"t{time_ms:012d}.png"


def _frame_path(root: Path, video_id: str, time_ms: int) -> Path:
    return root / ".work" / "frames" / video_id / f"t{time_ms:012d}.jpg"


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("L"), dtype=np.uint8) > 0


def decode_mask_data(mask_data: str, width: int, height: int) -> np.ndarray:
    try:
        encoded = mask_data.split(",", 1)[1] if "," in mask_data else mask_data
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as opened:
            mask = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
    except Exception as exc:
        raise ValueError("The selected fruit mask is not a valid PNG data URL.") from exc
    if mask.shape != (int(height), int(width)):
        raise ValueError(f"Fruit mask must be {width}x{height}; received {mask.shape[1]}x{mask.shape[0]}.")
    if not mask.any():
        raise ValueError("The selected fruit mask is empty.")
    return mask


def _write_mask(path: Path, mask: np.ndarray) -> None:
    output = io.BytesIO()
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(output, "PNG")
    atomic_write_bytes(path, output.getvalue(), backup=False)


def _video(session: dict[str, Any], video_id: str) -> dict[str, Any]:
    try:
        return next(item for item in session["videos"] if str(item["video_id"]) == str(video_id))
    except StopIteration as exc:
        raise KeyError(f"Unknown harvest video id: {video_id}") from exc


def _track(session: dict[str, Any], track_id: str) -> dict[str, Any]:
    try:
        return next(item for item in session["tracks"] if str(item["id"]) == str(track_id))
    except StopIteration as exc:
        raise KeyError(f"Unknown harvest track id: {track_id}") from exc


def _read_frame(video: dict[str, Any], time_ms: int) -> tuple[np.ndarray, int, float]:
    capture = cv2.VideoCapture(str(video["path"]))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video {video['file_name']}.")
        capture.set(cv2.CAP_PROP_POS_MSEC, float(time_ms))
        ok, frame = capture.read()
        if not ok or frame is None:
            fps = float(video.get("fps") or 0)
            frame_index = int(round(time_ms * fps / 1000.0)) if fps > 0 else 0
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Could not decode {video['file_name']} at {time_ms / 1000:.2f}s.")
        frame_index = max(0, int(round(float(capture.get(cv2.CAP_PROP_POS_FRAMES) or 1))) - 1)
        actual_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or time_ms)
        return frame, frame_index, actual_ms
    finally:
        capture.release()


def cached_frame(session: dict[str, Any], root: Path, video_id: str, time_ms: int) -> tuple[Path, dict[str, Any]]:
    video = _video(session, video_id)
    snapped = snap_time_ms(time_ms, video.get("duration_ms"))
    destination = _frame_path(root, video_id, snapped)
    metadata_path = destination.with_suffix(".json")
    if destination.is_file() and metadata_path.is_file():
        return destination, json.loads(metadata_path.read_text(encoding="utf-8"))
    frame, frame_index, actual_ms = _read_frame(video, snapped)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("Could not encode the selected harvest frame.")
    metadata = {
        "video_id": video_id,
        "time_ms": snapped,
        "actual_time_ms": round(actual_ms, 3),
        "source_frame_index": frame_index,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
    }
    atomic_write_bytes(destination, encoded.tobytes(), backup=False)
    atomic_write_json(metadata_path, metadata, backup=False)
    return destination, metadata


def browser_media_path(session: dict[str, Any], root: Path, video_id: str) -> Path:
    video = _video(session, video_id)
    source = Path(video["path"]).resolve()
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        codec = probe.stdout.strip().casefold() if probe.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        codec = ""
    suffix = source.suffix.lower()
    directly_playable = (
        (suffix in {".mp4", ".m4v", ".mov"} and codec in {"h264", "av1"})
        or (suffix == ".webm" and codec in {"vp8", "vp9", "av1"})
    )
    if directly_playable:
        return source
    destination = root / ".work" / "media" / f"{video_id}.mp4"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}-{uuid.uuid4().hex[:8]}.tmp.mp4")
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=60 * 60,
            check=False,
        )
        if completed.returncode != 0 or not temporary.is_file():
            raise RuntimeError(completed.stderr.strip() or "FFmpeg could not create a browser-compatible preview.")
        os.replace(temporary, destination)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required to preview this video format in the browser.") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _public_session(session: dict[str, Any]) -> dict[str, Any]:
    videos = [
        {key: value for key, value in video.items() if key != "path"}
        | {"media_url": f"/api/harvest/sessions/{session['session_id']}/videos/{video['video_id']}/media"}
        for video in session["videos"]
    ]
    tracks = []
    for item in session.get("tracks", []):
        public = {key: value for key, value in item.items() if key != "current_mask_path"}
        if item.get("current_mask_path"):
            public["mask_url"] = (
                f"/api/harvest/sessions/{session['session_id']}/tracks/{item['id']}/mask"
                f"?t={item.get('current_time_ms', 0)}"
            )
        tracks.append(public)
    return {
        **{
            key: value
            for key, value in session.items()
            if key not in {"session_root", "videos", "tracks", "deleted_tracks", "deleted_masks"}
        },
        "videos": videos,
        "tracks": tracks,
        "frame_url": (
            f"/api/harvest/sessions/{session['session_id']}/videos/{session['current_video_id']}/frame"
            f"?time_ms={session['current_time_ms']}"
        ) if session.get("current_video_id") else None,
    }


def public_session(session: dict[str, Any]) -> dict[str, Any]:
    return _public_session(session)


def _write_session(session: dict[str, Any], root: Path) -> None:
    session["updated_at"] = utc_now()
    atomic_write_json(_session_path(root), session, backup=False)
    manifest = {
        "schema_version": "1.0",
        "session_id": session["session_id"],
        "name": session["name"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "interval_ms": HARVEST_INTERVAL_MS,
        "model": session["model"],
        "videos": [{key: value for key, value in video.items() if key != "path"} for video in session["videos"]],
        "tracks": [{key: value for key, value in item.items() if key != "current_mask_path"} for item in session["tracks"]],
        "deleted_tracks": session.get("deleted_tracks", []),
        "deleted_masks": session.get("deleted_masks", []),
        "samples": session["samples"],
        "counts": {
            "tracks": len(session["tracks"]),
            "deleted_tracks": len(session.get("deleted_tracks", [])),
            "saved": sum(item.get("status") == "saved" for item in session["samples"]),
            "skipped": sum(item.get("status") == "skipped" for item in session["samples"]),
        },
    }
    atomic_write_json(root / MANIFEST_FILENAME, manifest, backup=False)


def create_session(
    video_paths: list[str],
    output_parent: str | Path,
    name: str,
    segmentation_classes: list[dict[str, Any]],
) -> tuple[dict[str, Any], Path]:
    parent = Path(output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / _slug(name)
    if root.exists() and _session_path(root).is_file():
        session = load_session(root)
        requested = inspect_videos(video_paths, interval_seconds=HARVEST_INTERVAL_MS / 1000)["videos"]
        if [item["sha256"] for item in requested] != [item["sha256"] for item in session["videos"]]:
            raise ValueError("This harvest folder belongs to different source videos. Choose another name or resume it directly.")
        return session, root
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Harvest destination is not empty: {root}")
    inspected = inspect_videos(video_paths, interval_seconds=HARVEST_INTERVAL_MS / 1000)
    if not inspected["videos"]:
        raise ValueError("None of the selected videos could be decoded.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "foregrounds").mkdir()
    now = utc_now()
    first_video = inspected["videos"][0]
    session = {
        "schema_version": "1.0",
        "session_id": str(uuid.uuid4()),
        "session_root": str(root),
        "name": str(name).strip() or root.name,
        "state": "active",
        "created_at": now,
        "updated_at": now,
        "interval_ms": HARVEST_INTERVAL_MS,
        "current_video_id": first_video["video_id"],
        "current_time_ms": 0,
        "videos": inspected["videos"],
        "tracks": [],
        "deleted_tracks": [],
        "deleted_masks": [],
        "samples": [],
        "completed_advances": [],
        "segmentation_classes": segmentation_classes,
        "warnings": inspected.get("warnings", []),
        "model": {
            "segmenter": "sam2.1-hiera-small",
            "checkpoint_sha256": SAM2_DEFAULT_CHECKPOINT_SHA256,
            "pinned_revision": SAM2_PINNED_REVISION,
            "optional_detector": "grounding-dino-tiny",
        },
    }
    _write_session(session, root)
    return session, root


def load_session(root: str | Path, *, verify_sources: bool = True) -> dict[str, Any]:
    resolved = Path(root).expanduser().resolve()
    path = _session_path(resolved)
    if not path.is_file():
        raise FileNotFoundError(f"Harvest session was not found: {path}")
    session = json.loads(path.read_text(encoding="utf-8"))
    session.setdefault("deleted_tracks", [])
    session.setdefault("deleted_masks", [])
    for video in session.get("videos", []):
        source = Path(video["path"])
        if not source.is_file():
            raise FileNotFoundError(f"Source video moved or is missing: {source}")
        if verify_sources and file_sha256(source) != video["sha256"]:
            raise ValueError(f"Source video changed since this harvest began: {source.name}")
    session["session_root"] = str(resolved)
    recovered = recover_boundary_samples(session, resolved)
    if recovered and session.get("state") == "finished":
        # A completed project's source folder has already been scanned. Re-run
        # the idempotent finish step so recovered PNGs are included without
        # discarding any defect labels added since the original harvest.
        finish_session(session, resolved)
    return session


def set_position(session: dict[str, Any], root: Path, video_id: str, time_ms: int) -> dict[str, Any]:
    if any(item.get("status") in {"active", "needs_mask"} for item in session.get("tracks", [])):
        raise ValueError("End, repair, or delete the open fruit tracks before seeking to another time or video.")
    video = _video(session, video_id)
    session["current_video_id"] = video_id
    session["current_time_ms"] = snap_time_ms(time_ms, video.get("duration_ms"))
    _write_session(session, root)
    return session


def rewind_session(session: dict[str, Any], root: Path) -> dict[str, Any]:
    """Move back one sample and restore editable track masks at that timestamp."""
    if session.get("state") != "active":
        raise ValueError("Only an active harvest session can move to a previous sample.")
    video_id = str(session.get("current_video_id") or "")
    video = _video(session, video_id)
    current_ms = int(session.get("current_time_ms", 0))
    if current_ms <= 0:
        raise ValueError("This is already the first sample in the video.")
    previous_ms = snap_time_ms(current_ms - HARVEST_INTERVAL_MS, video.get("duration_ms"))

    open_tracks = [
        item for item in session.get("tracks", [])
        if str(item.get("video_id")) == video_id and item.get("status") in {"active", "needs_mask"}
    ]
    newer_tracks = [item for item in open_tracks if int(item.get("seed_time_ms", 0)) > previous_ms]
    if newer_tracks:
        names = ", ".join(str(item.get("id")) for item in newer_tracks)
        raise ValueError(f"Delete or end tracks started on the current sample before going back: {names}.")

    restored: list[tuple[dict[str, Any], Path]] = []
    for track in session.get("tracks", []):
        if str(track.get("video_id")) != video_id or track.get("capture_mode") == "single":
            continue
        existed_at_previous = int(track.get("seed_time_ms", 0)) <= previous_ms
        ended_at_or_after_previous = (
            track.get("status") == "ended"
            and int(track.get("end_time_ms", -1)) >= previous_ms
        )
        if not existed_at_previous or (track not in open_tracks and not ended_at_or_after_previous):
            continue
        mask_path = _mask_path(root, str(track["id"]), previous_ms)
        if not mask_path.is_file():
            raise FileNotFoundError(f"The previous mask for {track['id']} is missing at {previous_ms / 1000:.2f}s.")
        restored.append((track, mask_path))

    restored_ids = {str(track["id"]) for track, _path in restored}
    removed_samples = [
        item for item in session.get("samples", [])
        if str(item.get("video_id")) == video_id
        and str(item.get("track_id")) in restored_ids
        and int(item.get("time_ms", -1)) >= previous_ms
    ]
    removed_keys = {str(item.get("key")) for item in removed_samples}
    session["samples"] = [item for item in session.get("samples", []) if str(item.get("key")) not in removed_keys]

    backup_dir: Path | None = None
    saved_samples = [item for item in removed_samples if item.get("relative_path")]
    if saved_samples:
        backup_dir = safe_resolve(
            root.resolve(),
            Path(".work") / "rewind-backups" / f"t{previous_ms:012d}-{uuid.uuid4().hex[:8]}",
        )
        backup_dir.mkdir(parents=True, exist_ok=True)
        for sample in saved_samples:
            source = safe_resolve(root.resolve(), str(sample["relative_path"]))
            if source.is_file():
                shutil.move(str(source), str(backup_dir / source.name))
        atomic_write_json(backup_dir / "rewound-samples.json", {
            "schema_version": "1.0",
            "video_id": video_id,
            "time_ms": previous_ms,
            "samples": removed_samples,
        }, backup=False)

    for track, mask_path in restored:
        track["status"] = "active"
        track["current_time_ms"] = previous_ms
        track["current_mask_path"] = mask_path.relative_to(root).as_posix()
        track.pop("end_time_ms", None)
        track.pop("end_reason", None)
        track["invalid_count"] = 0
        remaining = [item for item in session["samples"] if str(item.get("track_id")) == str(track["id"])]
        track["saved_count"] = sum(item.get("status") == "saved" for item in remaining)
        track["skipped_count"] = sum(item.get("status") == "skipped" for item in remaining)
        earlier = [item for item in remaining if int(item.get("time_ms", -1)) < previous_ms and item.get("status") == "saved"]
        latest = max(earlier, key=lambda item: int(item.get("time_ms", -1)), default=None)
        latest_area = (latest.get("quality") or {}).get("area") if latest else None
        track["last_valid_area"] = int(latest_area) if latest_area else None

    session["completed_advances"] = [
        item for item in session.get("completed_advances", [])
        if int(item.get("from_time_ms", -1)) < previous_ms
    ]
    session["current_time_ms"] = previous_ms
    _write_session(session, root)
    return {
        "from_time_ms": current_ms,
        "to_time_ms": previous_ms,
        "restored_track_ids": sorted(restored_ids),
        "removed_sample_count": len(removed_samples),
        "backup_dir": str(backup_dir) if backup_dir else None,
    }


def reopen_session(session: dict[str, Any], root: Path) -> dict[str, Any]:
    """Make a completed harvest editable again without changing its saved outputs."""
    if session.get("state") == "active":
        return session
    if session.get("state") != "finished":
        raise ValueError(f"Harvest session cannot be reopened from state {session.get('state')!r}.")
    session["state"] = "active"
    session["reopened_at"] = utc_now()
    _write_session(session, root)
    return session


def create_track(
    session: dict[str, Any],
    root: Path,
    video_id: str,
    time_ms: int,
    mask_data: str,
    prompt_metadata: dict[str, Any] | None = None,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    video = _video(session, video_id)
    snapped = snap_time_ms(time_ms, video.get("duration_ms"))
    if video_id != session["current_video_id"] or snapped != int(session["current_time_ms"]):
        raise ValueError("The track seed must match the currently displayed harvest sample.")
    mask = decode_mask_data(mask_data, int(video["width"]), int(video["height"]))
    known_track_ids = [str(item.get("id", "")) for item in session.get("tracks", [])]
    known_track_ids.extend(str(item.get("id", "")) for item in session.get("deleted_tracks", []))
    known_numbers = [
        int(match.group(1))
        for track_id_value in known_track_ids
        if (match := re.fullmatch(r"fruit-(\d+)", track_id_value))
    ]
    track_number = max(known_numbers, default=0) + 1
    track_id = f"fruit-{track_number:05d}"
    path = _mask_path(root, track_id, snapped)
    _write_mask(path, mask)
    track = {
        "id": track_id,
        "video_id": video_id,
        "color": TRACK_COLORS[(track_number - 1) % len(TRACK_COLORS)],
        "status": "active",
        "seed_time_ms": snapped,
        "current_time_ms": snapped,
        "current_mask_path": path.relative_to(root).as_posix(),
        "last_valid_area": None,
        "invalid_count": 0,
        "saved_count": 0,
        "skipped_count": 0,
        "prompt_metadata": prompt_metadata or {},
    }
    session["tracks"].append(track)
    if persist:
        _write_session(session, root)
    return track


def delete_track(session: dict[str, Any], root: Path, track_id: str) -> dict[str, Any]:
    """Remove a track from the live harvest while retaining recoverable copies."""
    if session.get("state") != "active":
        raise ValueError("Tracks can only be deleted from an active harvest session.")
    track = _track(session, track_id)
    resolved_root = root.resolve()
    deleted_at = utc_now()
    backup_dir = safe_resolve(
        resolved_root,
        Path("backups") / "deleted-tracks" / f"{_slug(track_id)}-{uuid.uuid4().hex[:8]}",
    )
    backup_files = backup_dir / "files"
    track_samples = [item for item in session.get("samples", []) if str(item.get("track_id")) == str(track_id)]
    source_files: list[Path] = []

    mask_dir = safe_resolve(resolved_root, Path(".work") / "masks" / track_id)
    if mask_dir.is_dir():
        mask_backup = safe_resolve(backup_files, Path(".work") / "masks" / track_id)
        mask_backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(mask_dir, mask_backup)

    for sample in track_samples:
        relative_path = sample.get("relative_path")
        if not relative_path:
            continue
        source = safe_resolve(resolved_root, relative_path)
        if not source.is_file():
            continue
        destination = safe_resolve(backup_files, relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_files.append(source)

    backup_record = {
        "schema_version": "1.0",
        "deleted_at": deleted_at,
        "session_id": session["session_id"],
        "session_root": str(resolved_root),
        "track": track,
        "samples": track_samples,
        "restore_files_from": str(backup_files),
    }
    atomic_write_json(backup_dir / "deleted-track.json", backup_record, backup=False)

    removed_keys = {str(item.get("key")) for item in track_samples}
    session["tracks"] = [item for item in session.get("tracks", []) if str(item.get("id")) != str(track_id)]
    session["samples"] = [item for item in session.get("samples", []) if str(item.get("track_id")) != str(track_id)]
    for advance in session.get("completed_advances", []):
        advance["committed_keys"] = [key for key in advance.get("committed_keys", []) if str(key) not in removed_keys]
    session.setdefault("deleted_tracks", []).append({
        "id": track["id"],
        "deleted_at": deleted_at,
        "backup_path": backup_dir.relative_to(resolved_root).as_posix(),
        "saved_count": sum(item.get("status") == "saved" for item in track_samples),
        "skipped_count": sum(item.get("status") == "skipped" for item in track_samples),
    })
    _write_session(session, resolved_root)

    cleanup_warnings: list[str] = []
    try:
        if mask_dir.is_dir():
            shutil.rmtree(mask_dir)
    except OSError as exc:
        cleanup_warnings.append(f"Could not remove the live mask folder: {exc}")
    for source in source_files:
        try:
            source.unlink(missing_ok=True)
        except OSError as exc:
            cleanup_warnings.append(f"Could not remove {source.name}: {exc}")

    return {
        "track_id": track["id"],
        "deleted_sample_count": len(track_samples),
        "deleted_saved_count": sum(item.get("status") == "saved" for item in track_samples),
        "backup_dir": str(backup_dir),
        "cleanup_warnings": cleanup_warnings,
    }


def delete_track_mask(session: dict[str, Any], root: Path, track_id: str) -> dict[str, Any]:
    """Delete the selected track's current mask and pause it for a replacement."""
    if session.get("state") != "active":
        raise ValueError("Masks can only be deleted from an active harvest session.")
    track = _track(session, track_id)
    if track.get("status") != "active":
        raise ValueError("Only the current mask of an active track can be deleted.")
    if str(track.get("video_id")) != str(session.get("current_video_id")):
        raise ValueError("The selected track does not belong to the displayed video.")
    relative_path = track.get("current_mask_path")
    if not relative_path:
        raise FileNotFoundError("The selected track does not have a current mask.")

    resolved_root = root.resolve()
    source = safe_resolve(resolved_root, relative_path)
    if not source.is_file():
        raise FileNotFoundError("The selected track's current mask is missing.")
    deleted_at = utc_now()
    backup_dir = safe_resolve(
        resolved_root,
        Path("backups") / "deleted-masks" / f"{_slug(track_id)}-t{int(track.get('current_time_ms', 0)):012d}-{uuid.uuid4().hex[:8]}",
    )
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_mask = backup_dir / source.name
    shutil.copy2(source, backup_mask)
    atomic_write_json(backup_dir / "deleted-mask.json", {
        "schema_version": "1.0",
        "deleted_at": deleted_at,
        "session_id": session["session_id"],
        "track_id": track["id"],
        "video_id": track["video_id"],
        "time_ms": int(track.get("current_time_ms", session.get("current_time_ms", 0))),
        "original_path": str(relative_path),
        "backup_mask": backup_mask.name,
    }, backup=False)

    track["current_mask_path"] = None
    track["status"] = "needs_mask"
    track["invalid_count"] = 0
    session.setdefault("deleted_masks", []).append({
        "track_id": track["id"],
        "video_id": track["video_id"],
        "time_ms": int(track.get("current_time_ms", session.get("current_time_ms", 0))),
        "deleted_at": deleted_at,
        "backup_path": backup_dir.relative_to(resolved_root).as_posix(),
    })
    _write_session(session, resolved_root)
    source.unlink(missing_ok=True)
    return {"track_id": track["id"], "backup_dir": str(backup_dir)}


def correct_track(session: dict[str, Any], root: Path, track_id: str, mask_data: str) -> dict[str, Any]:
    track = _track(session, track_id)
    if track["status"] not in {"active", "needs_mask"}:
        raise ValueError("Only an open fruit track can be corrected.")
    if str(track.get("video_id")) != str(session.get("current_video_id")):
        raise ValueError("The selected track does not belong to the displayed video.")
    video = _video(session, track["video_id"])
    mask = decode_mask_data(mask_data, int(video["width"]), int(video["height"]))
    path = _mask_path(root, track_id, int(session["current_time_ms"]))
    _write_mask(path, mask)
    track["current_mask_path"] = path.relative_to(root).as_posix()
    track["current_time_ms"] = int(session["current_time_ms"])
    track["invalid_count"] = 0
    track["status"] = "active"
    _write_session(session, root)
    return track


def track_mask_path(session: dict[str, Any], root: Path, track_id: str) -> Path:
    track = _track(session, track_id)
    path = (root / track["current_mask_path"]).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise FileNotFoundError("The current track mask is missing.")
    return path


def _largest_mask_component(mask: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    area = int(mask.sum())
    if not area:
        return None, {"area": 0}
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None, {"area": area}
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = labels == largest_label
    largest_area = int(largest.sum())
    ys, xs = np.where(largest)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    touches_boundary = bool(largest[0, :].any() or largest[-1, :].any() or largest[:, 0].any() or largest[:, -1].any())
    metadata = {
        "area": largest_area,
        "bbox": [int(xs.min()), int(ys.min()), width, height],
        "largest_component_fraction": largest_area / area,
        "touches_boundary": touches_boundary,
    }
    return largest, metadata


def _quality_mask(mask: np.ndarray, previous_area: int | None) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    largest, metadata = _largest_mask_component(mask)
    if largest is None:
        return None, "empty mask", metadata
    _left, _top, width, height = metadata["bbox"]
    if width < 32 or height < 32:
        return None, f"fruit is too small ({width}x{height}px)", metadata
    if float(metadata["largest_component_fraction"]) < 0.80:
        return None, "mask is fragmented", metadata
    if previous_area:
        ratio = int(metadata["area"]) / previous_area
        metadata["area_ratio"] = ratio
        if ratio < 0.4 or ratio > 2.5:
            return None, f"mask area changed implausibly ({ratio:.2f}x)", metadata
    return largest, None, metadata


def _sample_key(video_id: str, track_id: str, time_ms: int) -> str:
    return f"{video_id}:{track_id}:{time_ms}"


def _write_sample_png(
    session: dict[str, Any],
    root: Path,
    track: dict[str, Any],
    frame: np.ndarray,
    valid: np.ndarray,
    record: dict[str, Any],
) -> None:
    video = _video(session, track["video_id"])
    ys, xs = np.where(valid)
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad_x = max(1, int(round((right - left) * 0.05)))
    pad_y = max(1, int(round((bottom - top) * 0.05)))
    crop_left, crop_top = max(0, left - pad_x), max(0, top - pad_y)
    crop_right, crop_bottom = min(frame.shape[1], right + pad_x), min(frame.shape[0], bottom + pad_y)
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    crop = rgba[crop_top:crop_bottom, crop_left:crop_right]
    time_ms = int(record["time_ms"])
    file_name = f"{_slug(Path(video['file_name']).stem)}__{track['id']}__t{time_ms:012d}.png"
    destination = root / "foregrounds" / file_name
    output = io.BytesIO()
    Image.fromarray(crop, mode="RGBA").save(output, "PNG")
    atomic_write_bytes(destination, output.getvalue(), backup=False)
    record.pop("reason", None)
    record.update({
        "status": "saved",
        "file_name": file_name,
        "relative_path": f"foregrounds/{file_name}",
        "crop_bbox": [crop_left, crop_top, crop_right - crop_left, crop_bottom - crop_top],
        "mask_bbox": [left, top, right - left, bottom - top],
    })


def _refresh_track_sample_counters(session: dict[str, Any], track: dict[str, Any]) -> None:
    samples = sorted(
        (item for item in session.get("samples", []) if str(item.get("track_id")) == str(track["id"])),
        key=lambda item: int(item.get("time_ms", -1)),
    )
    track["saved_count"] = sum(item.get("status") == "saved" for item in samples)
    track["skipped_count"] = sum(item.get("status") == "skipped" for item in samples)
    latest_saved = next((item for item in reversed(samples) if item.get("status") == "saved"), None)
    latest_area = (latest_saved.get("quality") or {}).get("area") if latest_saved else None
    track["last_valid_area"] = int(latest_area) if latest_area else None

    consecutive_invalid = 0
    for item in reversed(samples):
        if item.get("status") == "saved":
            break
        if item.get("status") == "skipped" and item.get("reason") != "skipped by user":
            consecutive_invalid += 1
    track["invalid_count"] = consecutive_invalid


def recover_boundary_samples(session: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """Write PNGs for samples skipped by the retired boundary rejection rule."""
    candidates = [
        item for item in session.get("samples", [])
        if item.get("status") == "skipped" and item.get("reason") == LEGACY_BOUNDARY_SKIP_REASON
    ]
    if not candidates:
        return []

    tracks = {str(item["id"]): item for item in session.get("tracks", [])}
    recovered: list[dict[str, Any]] = []
    recovery_warnings: list[str] = []
    (root / "foregrounds").mkdir(parents=True, exist_ok=True)
    for record in candidates:
        track_id = str(record.get("track_id", ""))
        time_ms = int(record.get("time_ms", 0))
        track = tracks.get(track_id)
        try:
            if track is None:
                raise KeyError(f"track {track_id} is no longer present")
            mask_path = _mask_path(root, track_id, time_ms)
            if not mask_path.is_file():
                raise FileNotFoundError(f"retained mask is missing: {mask_path}")
            largest, quality = _largest_mask_component(_read_mask(mask_path))
            if largest is None:
                raise ValueError("retained mask is empty")
            frame_path, _frame_metadata = cached_frame(session, root, str(record["video_id"]), time_ms)
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("source frame could not be reloaded")
            previous_reason = str(record.get("reason") or LEGACY_BOUNDARY_SKIP_REASON)
            record["quality"] = {**quality, **(record.get("quality") or {}), "touches_boundary": True}
            _write_sample_png(session, root, track, frame, largest, record)
            record["recovered_from_reason"] = previous_reason
            record["recovered_at"] = utc_now()
            recovered.append(record)
        except Exception as exc:
            recovery_warnings.append(
                f"Could not recover boundary sample {track_id} at {time_ms / 1000:.2f}s: {exc}"
            )

    for track_id in {str(item["track_id"]) for item in recovered}:
        track = tracks.get(track_id)
        if track is not None:
            _refresh_track_sample_counters(session, track)
    if recovery_warnings:
        warnings = session.setdefault("warnings", [])
        for warning in recovery_warnings:
            if warning not in warnings:
                warnings.append(warning)
    if recovered or recovery_warnings:
        _write_session(session, root)
    return recovered


def _save_sample(
    session: dict[str, Any],
    root: Path,
    track: dict[str, Any],
    frame: np.ndarray,
    frame_metadata: dict[str, Any],
    *,
    manual_skip: bool = False,
) -> dict[str, Any]:
    time_ms = int(session["current_time_ms"])
    key = _sample_key(track["video_id"], track["id"], time_ms)
    existing = next((item for item in session["samples"] if item["key"] == key), None)
    if existing:
        return existing
    mask = _read_mask(root / track["current_mask_path"])
    valid, reason, quality = _quality_mask(mask, track.get("last_valid_area"))
    if manual_skip:
        valid, reason = None, "skipped by user"
    video = _video(session, track["video_id"])
    record: dict[str, Any] = {
        "key": key,
        "video_id": track["video_id"],
        "video_name": video["file_name"],
        "track_id": track["id"],
        "time_ms": time_ms,
        "actual_time_ms": frame_metadata["actual_time_ms"],
        "source_frame_index": frame_metadata["source_frame_index"],
        "quality": quality,
    }
    if valid is None:
        record.update({"status": "skipped", "reason": reason})
        track["skipped_count"] += 1
        if not manual_skip:
            track["invalid_count"] += 1
        session["samples"].append(record)
        return record
    _write_sample_png(session, root, track, frame, valid, record)
    session["samples"].append(record)
    track["last_valid_area"] = int(valid.sum())
    track["invalid_count"] = 0
    track["saved_count"] += 1
    return record


def save_single_mask(
    session: dict[str, Any],
    root: Path,
    video_id: str,
    time_ms: int,
    mask_data: str,
    prompt_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Save one selected fruit at the current frame without starting tracking."""
    track = create_track(
        session,
        root,
        video_id,
        time_ms,
        mask_data,
        prompt_metadata,
        persist=False,
    )
    try:
        frame_path, frame_metadata = cached_frame(session, root, video_id, time_ms)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not reload the selected harvest frame.")
        sample = _save_sample(session, root, track, frame, frame_metadata)
        track.update({
            "status": "ended",
            "capture_mode": "single",
            "end_time_ms": int(session["current_time_ms"]),
            "end_reason": "single-mask capture",
        })
        _write_session(session, root)
        return track, sample
    except Exception:
        created_samples = [item for item in session.get("samples", []) if item.get("track_id") == track["id"]]
        session["samples"] = [item for item in session.get("samples", []) if item.get("track_id") != track["id"]]
        session["tracks"] = [item for item in session.get("tracks", []) if item.get("id") != track["id"]]
        for sample in created_samples:
            if sample.get("relative_path"):
                safe_resolve(root, sample["relative_path"]).unlink(missing_ok=True)
        mask_dir = safe_resolve(root, Path(".work") / "masks" / track["id"])
        if mask_dir.is_dir():
            shutil.rmtree(mask_dir)
        raise


def _decode_window(video: dict[str, Any], start_ms: int, end_ms: int, destination: Path) -> tuple[list[Path], dict[str, Any]]:
    start_frame, _, _ = _read_frame(video, start_ms)
    end_frame, end_index, end_actual = _read_frame(video, end_ms)
    fps = float(video.get("fps") or 0)
    start_index = max(0, int(round(start_ms * fps / 1000.0))) if fps > 0 else 0
    final_index = max(start_index + 1, int(round(end_ms * fps / 1000.0))) if fps > 0 else start_index + 1
    frames = [start_frame]
    capture = cv2.VideoCapture(str(video["path"]))
    try:
        if capture.isOpened() and fps > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_index + 1)
            for _index in range(start_index + 1, final_index):
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
    finally:
        capture.release()
    frames.append(end_frame)
    paths: list[Path] = []
    for index, frame in enumerate(frames):
        path = destination / f"{index:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise ValueError("Could not prepare native frames for SAM 2 tracking.")
        paths.append(path)
    return paths, {
        "time_ms": end_ms,
        "actual_time_ms": round(end_actual, 3),
        "source_frame_index": end_index,
        "width": int(end_frame.shape[1]),
        "height": int(end_frame.shape[0]),
        "frame": end_frame,
    }


class HarvestTracker:
    def __init__(self, mock: bool = False) -> None:
        self.mock = mock
        self.predictor: Any = None
        self.torch: Any = None
        self.device = "cpu" if mock else os.getenv("SAM2_DEVICE", "auto").strip().lower()
        self.lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "provider": "mock" if self.mock else "sam2_video",
            "state": "ready" if self.mock or self.predictor is not None else "idle",
            "device": self.device,
            "checkpoint": None if self.mock else str(SAM2_DEFAULT_CHECKPOINT),
        }

    def _load(self) -> None:
        if self.mock or self.predictor is not None:
            return
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for SAM 2 video tracking.")
        checkpoint = Path(os.getenv("SAM2_CHECKPOINT_PATH", str(SAM2_DEFAULT_CHECKPOINT))).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM 2 checkpoint is missing: {checkpoint}")
        self.torch = torch
        self.device = device
        self.predictor = build_sam2_video_predictor(
            os.getenv("SAM2_MODEL_CONFIG", SAM2_DEFAULT_CONFIG),
            str(checkpoint),
            device=device,
            apply_postprocessing=False,
        )

    def propagate(
        self,
        video: dict[str, Any],
        start_ms: int,
        end_ms: int,
        masks: dict[str, np.ndarray],
        work_root: Path,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _cancelled(cancel_check)
        with tempfile.TemporaryDirectory(prefix="track-window-", dir=str(work_root)) as temporary:
            frame_dir = Path(temporary)
            paths, end_metadata = _decode_window(video, start_ms, end_ms, frame_dir)
            _cancelled(cancel_check)
            if self.mock:
                shifted = {}
                for track_id, mask in masks.items():
                    output = np.zeros_like(mask)
                    output[:, 2:] = mask[:, :-2]
                    shifted[track_id] = output
                return shifted, end_metadata
            with self.lock:
                self._load()
                autocast = self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16) if self.device == "cuda" else nullcontext()
                try:
                    with self.torch.inference_mode(), autocast:
                        state = self.predictor.init_state(
                            video_path=str(frame_dir),
                            offload_video_to_cpu=True,
                            offload_state_to_cpu=False,
                            async_loading_frames=False,
                        )
                        object_tracks: dict[int, str] = {}
                        for index, (track_id, mask) in enumerate(masks.items(), start=1):
                            object_tracks[index] = track_id
                            self.predictor.add_new_mask(state, 0, index, mask)
                        final_ids: list[Any] = []
                        final_logits: Any = None
                        for _frame_index, object_ids, logits in self.predictor.propagate_in_video(state):
                            _cancelled(cancel_check)
                            final_ids, final_logits = list(object_ids), logits
                        result = {}
                        for position, object_id in enumerate(final_ids):
                            track_id = object_tracks.get(int(object_id))
                            if track_id is not None and position < len(final_logits):
                                result[track_id] = final_logits[position].detach().float().cpu().numpy().squeeze() > 0
                        self.predictor.reset_state(state)
                        return result, end_metadata
                except Exception as exc:
                    if self.torch is not None and isinstance(exc, self.torch.cuda.OutOfMemoryError):
                        self.torch.cuda.empty_cache()
                        raise RuntimeError("CUDA ran out of memory during fruit tracking. Close GPU applications and retry.") from exc
                    raise


def advance_session(
    session: dict[str, Any],
    root: Path,
    tracker: HarvestTracker,
    *,
    skip_track_ids: set[str] | None = None,
    end_track_ids: set[str] | None = None,
    request_id: str | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    skip_track_ids = skip_track_ids or set()
    end_track_ids = end_track_ids or set()
    if any(
        item.get("status") == "needs_mask" and item.get("video_id") == session.get("current_video_id")
        for item in session.get("tracks", [])
    ):
        raise ValueError("Apply a replacement mask or delete the paused track before advancing.")
    completed_advances = session.setdefault("completed_advances", [])
    if request_id:
        completed = next((item for item in completed_advances if item.get("request_id") == request_id), None)
        if completed:
            committed_keys = set(completed.get("committed_keys", []))
            return {
                "session": public_session(session),
                "committed": [item for item in session.get("samples", []) if item.get("key") in committed_keys],
                "idempotent_replay": True,
            }
    active = [item for item in session["tracks"] if item["status"] == "active" and item["video_id"] == session["current_video_id"]]
    video = _video(session, session["current_video_id"])
    current_ms = int(session["current_time_ms"])
    frame_path, frame_metadata = cached_frame(session, root, video["video_id"], current_ms)
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not reload the current harvest frame.")
    committed = []
    for track in active:
        _cancelled(cancel_check)
        committed.append(_save_sample(session, root, track, frame, frame_metadata, manual_skip=track["id"] in skip_track_ids))
        if track["id"] in end_track_ids:
            track["status"] = "ended"
            track["end_time_ms"] = current_ms
        elif int(track.get("invalid_count", 0)) >= 2:
            track["status"] = "ended"
            track["end_time_ms"] = current_ms
            track["end_reason"] = "two consecutive invalid samples"
    next_ms = snap_time_ms(current_ms + HARVEST_INTERVAL_MS, video.get("duration_ms"))
    continuing = [item for item in active if item["status"] == "active"]
    if next_ms > current_ms and continuing:
        masks = {item["id"]: _read_mask(root / item["current_mask_path"]) for item in continuing}
        propagated, end_metadata = tracker.propagate(video, current_ms, next_ms, masks, root / ".work", cancel_check)
        for item in continuing:
            next_mask = propagated.get(item["id"])
            if next_mask is None:
                next_mask = np.zeros_like(masks[item["id"]])
            destination = _mask_path(root, item["id"], next_ms)
            _write_mask(destination, next_mask)
            item["current_mask_path"] = destination.relative_to(root).as_posix()
            item["current_time_ms"] = next_ms
        end_frame = end_metadata.pop("frame")
        destination = _frame_path(root, video["video_id"], next_ms)
        ok, encoded = cv2.imencode(".jpg", end_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            atomic_write_bytes(destination, encoded.tobytes(), backup=False)
            atomic_write_json(destination.with_suffix(".json"), end_metadata, backup=False)
    session["current_time_ms"] = next_ms
    if next_ms == current_ms:
        for item in continuing:
            item["status"] = "ended"
            item["end_time_ms"] = current_ms
            item["end_reason"] = "end of video"
    if request_id:
        completed_advances.append({
            "request_id": request_id,
            "from_time_ms": current_ms,
            "to_time_ms": next_ms,
            "committed_keys": [item["key"] for item in committed],
        })
    _write_session(session, root)
    return {"session": public_session(session), "committed": committed, "idempotent_replay": False}


def finish_session(session: dict[str, Any], root: Path) -> dict[str, Any]:
    saved = [item for item in session["samples"] if item.get("status") == "saved"]
    if not saved:
        raise ValueError("No valid naartjie cut-outs have been saved yet.")
    for track in session["tracks"]:
        if track["status"] in {"active", "needs_mask"}:
            track["status"] = "ended"
            track["end_time_ms"] = int(session["current_time_ms"])
            track["end_reason"] = "harvest finished"
    foregrounds = root / "foregrounds"
    project_root = root / "segmentation-project"
    if project_root.exists() and any(project_root.iterdir()) and not (project_root / "jano-project.json").is_file():
        raise FileExistsError(f"Segmentation project destination is not empty: {project_root}")
    project = create_or_update_project(foregrounds, project_root, f"{session['name']} segmentation")
    project["annotation_mode"] = "segmentation"
    project["source_kind"] = "harvested_foregrounds"
    project["segmentation_classes"] = session["segmentation_classes"]
    samples_by_name = {item["file_name"]: item for item in saved}
    for image in project["images"]:
        sample = samples_by_name.get(image["file_name"])
        if not sample:
            continue
        image.update({
            "fruit_id": sample["track_id"],
            "view_id": f"t{int(sample['time_ms']):012d}",
            "harvest_source_video_id": sample["video_id"],
            "harvest_source_video_name": sample["video_name"],
            "harvest_time_ms": sample["time_ms"],
            "harvest_source_frame_index": sample["source_frame_index"],
        })
    project["harvest_manifest"] = str((root / MANIFEST_FILENAME).resolve())
    project["harvest_session_dir"] = str(root.resolve())
    project["_dirty_annotation_image_ids"] = [str(image["id"]) for image in project["images"]]
    save_project(project, project_root)
    create_split_manifest(project_root, {"train": 0.7, "val": 0.2, "test": 0.1}, seed=42)
    project = load_project(project_root)
    session["state"] = "finished"
    session["project_dir"] = str(project_root)
    _write_session(session, root)
    return {"project_dir": str(project_root), "project": project}
