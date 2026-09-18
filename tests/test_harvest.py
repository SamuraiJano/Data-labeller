from __future__ import annotations

import base64
import io
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from tools.dataset_labeler.app import create_app
from tools.dataset_labeler.core import SEGMENTATION_CLASS_DEFINITIONS, load_project, replace_image_annotations, save_project
from tools.dataset_labeler.harvest import (
    HarvestCancelled,
    HarvestTracker,
    _decode_window,
    _quality_mask,
    _write_session,
    advance_session,
    cached_frame,
    correct_track,
    create_session,
    create_track,
    delete_track,
    delete_track_mask,
    finish_session,
    load_session,
    rewind_session,
    snap_time_ms,
)


def _moving_video(path: Path, frames: int = 20, fps: float = 20.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (128, 96))
    if not writer.isOpened():
        raise RuntimeError("Test video writer is unavailable.")
    try:
        for index in range(frames):
            frame = np.full((96, 128, 3), 20, dtype=np.uint8)
            cv2.circle(frame, (52 + index // 5 * 2, 48), 22, (20, 125, 235), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _mask_data(mask: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(output, "PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _wait_job(client: TestClient, job_id: str, timeout: float = 10) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/harvest/jobs/{job_id}").json()
        if job["state"] in {"complete", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"Harvest job {job_id} did not finish in {timeout}s")


def test_harvest_snap_quality_and_skip_rules() -> None:
    assert snap_time_ms(124) == 0
    assert snap_time_ms(125) == 250
    assert snap_time_ms(126) == 250
    assert snap_time_ms(999, 820) == 750
    assert snap_time_ms(1000, 1000) == 750
    valid = np.zeros((96, 128), dtype=bool)
    valid[25:70, 30:80] = True
    cleaned, reason, metadata = _quality_mask(valid, None)
    assert reason is None and cleaned is not None and metadata["bbox"] == [30, 25, 50, 45]
    partial = valid.copy()
    partial[:, :40] = False
    partial[0:40, 5:45] = True
    boundary_mask, boundary_reason, boundary_metadata = _quality_mask(partial, None)
    assert boundary_mask is not None and boundary_reason is None
    assert boundary_metadata["touches_boundary"] is True
    tiny = np.zeros_like(valid)
    tiny[20:30, 20:30] = True
    assert "too small" in str(_quality_mask(tiny, None)[1])
    fragmented = np.zeros_like(valid)
    fragmented[10:50, 10:50] = True
    fragmented[55:85, 85:115] = True
    assert "fragmented" in str(_quality_mask(fragmented, None)[1])
    assert "implausibly" in str(_quality_mask(valid, 100)[1])


def test_boundary_masks_are_saved_and_legacy_boundary_skips_are_recovered(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "boundary.avi")
    session, root = create_session([str(video)], tmp_path / "outputs", "Boundary", [])
    boundary = np.zeros((96, 128), dtype=bool)
    boundary[20:65, 0:50] = True
    track = create_track(session, root, session["current_video_id"], 0, _mask_data(boundary))

    future = advance_session(session, root, HarvestTracker(mock=True))["committed"][0]

    assert future["status"] == "saved"
    assert future["quality"]["touches_boundary"] is True
    assert future["mask_bbox"] == [0, 20, 50, 45]
    assert (root / future["relative_path"]).is_file()

    legacy_session, legacy_root = create_session(
        [str(video)], tmp_path / "legacy-outputs", "Legacy boundary", []
    )
    legacy_track = create_track(
        legacy_session,
        legacy_root,
        legacy_session["current_video_id"],
        0,
        _mask_data(boundary),
    )
    _frame_path, frame_metadata = cached_frame(
        legacy_session, legacy_root, legacy_session["current_video_id"], 0
    )
    legacy_session["samples"].append({
        "key": f"{legacy_track['video_id']}:{legacy_track['id']}:0",
        "video_id": legacy_track["video_id"],
        "video_name": legacy_session["videos"][0]["file_name"],
        "track_id": legacy_track["id"],
        "time_ms": 0,
        "actual_time_ms": frame_metadata["actual_time_ms"],
        "source_frame_index": frame_metadata["source_frame_index"],
        "quality": {"area": 2250, "bbox": [0, 20, 50, 45], "largest_component_fraction": 1.0},
        "status": "skipped",
        "reason": "fruit touches the source-frame boundary",
    })
    legacy_track.update({"skipped_count": 1, "invalid_count": 1})
    _write_session(legacy_session, legacy_root)

    recovered_session = load_session(legacy_root)
    recovered = recovered_session["samples"][0]

    assert recovered["status"] == "saved"
    assert recovered["recovered_from_reason"] == "fruit touches the source-frame boundary"
    assert recovered["quality"]["touches_boundary"] is True
    assert (legacy_root / recovered["relative_path"]).is_file()
    assert recovered_session["tracks"][0]["saved_count"] == 1
    assert recovered_session["tracks"][0]["skipped_count"] == 0
    assert recovered_session["tracks"][0]["invalid_count"] == 0
    assert len(load_session(legacy_root)["samples"]) == 1
    assert len(list((legacy_root / "foregrounds").glob("*.png"))) == 1


def test_native_tracking_window_decodes_every_frame(tmp_path: Path) -> None:
    video_path = _moving_video(tmp_path / "native.avi")
    session, root = create_session([str(video_path)], tmp_path / "out", "Native", [])
    frame_dir = root / ".work" / "window-test"
    frame_dir.mkdir(parents=True)
    paths, metadata = _decode_window(session["videos"][0], 0, 250, frame_dir)
    assert len(paths) == 6  # 20 FPS: frames 0 through 5, not only the two samples.
    assert all(path.is_file() for path in paths)
    assert metadata["time_ms"] == 250
    assert metadata["source_frame_index"] == 5


def test_harvest_tracks_saves_rgba_and_creates_unlabelled_project(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "conveyor.avi")
    session, root = create_session(
        [str(video)],
        tmp_path / "outputs",
        "Defective fruit",
        [dict(item) for item in SEGMENTATION_CLASS_DEFINITIONS],
    )
    mask = np.zeros((96, 128), dtype=bool)
    cv2.circle(mask.view(np.uint8), (52, 48), 22, 1, -1)
    track = create_track(session, root, session["current_video_id"], 0, _mask_data(mask))
    tracker = HarvestTracker(mock=True)
    for _ in range(3):
        result = advance_session(session, root, tracker)
        assert result["committed"][0]["track_id"] == track["id"]
    saved = [item for item in session["samples"] if item["status"] == "saved"]
    assert [item["time_ms"] for item in saved] == [0, 250, 500]
    assert len(list((root / "foregrounds").glob("*.png"))) == 3
    for path in (root / "foregrounds").glob("*.png"):
        with Image.open(path) as opened:
            assert opened.mode == "RGBA"
            alpha = np.asarray(opened.getchannel("A"))
            assert alpha.max() == 255 and alpha.min() == 0

    finished = finish_session(session, root)
    project = load_project(finished["project_dir"])
    assert project["annotation_mode"] == "segmentation"
    assert project["annotations"] == []
    assert all(image["status"] == "unlabelled" for image in project["images"])
    assert {image["fruit_id"] for image in project["images"]} == {track["id"]}
    assert len({image["split"] for image in project["images"]}) == 1


def test_finished_harvest_can_reopen_without_erasing_existing_defect_labels(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "return-to-harvest.avi")
    session, root = create_session(
        [str(video)],
        tmp_path / "outputs",
        "Return to harvest",
        [dict(item) for item in SEGMENTATION_CLASS_DEFINITIONS],
    )
    mask = np.zeros((96, 128), dtype=bool)
    mask[20:65, 25:75] = True
    create_track(session, root, session["current_video_id"], 0, _mask_data(mask))
    advance_session(session, root, HarvestTracker(mock=True))
    finished = finish_session(session, root)
    project_dir = Path(finished["project_dir"])
    project = load_project(project_dir)
    image = project["images"][0]
    defect_class = project["segmentation_classes"][0]
    replace_image_annotations(
        project,
        image["id"],
        [{
            "type": "defect_mask",
            "class_uuid": defect_class["uuid"],
            "points": [[5, 5], [15, 5], [15, 15], [5, 15]],
            "source": "manual",
        }],
        {"status": "approved"},
        image["revision"],
    )
    save_project(project, project_dir)

    client = TestClient(create_app(default_project_dir=project_dir, provider_name="mock"))
    reopened_response = client.post("/api/harvest/sessions/reopen-current-project", json={})

    assert reopened_response.status_code == 200, reopened_response.text
    assert reopened_response.json()["session"]["state"] == "active"
    reopened = load_session(root)
    assert reopened["state"] == "active"
    finish_session(reopened, root)
    regenerated = load_project(project_dir)
    assert len(regenerated["annotations"]) == 1
    assert regenerated["annotations"][0]["annotation_kind"] == "defect_mask"
    assert regenerated["images"][0]["status"] == "approved"


def test_crop_filename_padding_and_retry_are_deterministic(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "stable name.avi")
    session, root = create_session([str(video)], tmp_path / "outputs", "Stable", [])
    mask = np.zeros((96, 128), dtype=bool)
    mask[20:60, 30:80] = True
    track = create_track(session, root, session["current_video_id"], 0, _mask_data(mask))
    first_result = advance_session(session, root, HarvestTracker(mock=True), request_id="retry-safe-step")
    first = first_result["committed"][0]
    assert first["file_name"] == "stable-name__fruit-00001__t000000000000.png"
    assert first["mask_bbox"] == [30, 20, 50, 40]
    assert first["crop_bbox"] == [28, 18, 54, 44]
    with Image.open(root / first["relative_path"]) as opened:
        assert opened.size == (54, 44)
    assert track["saved_count"] == 1
    assert len(list((root / "foregrounds").glob("*t000000000000.png"))) == 1
    replay = advance_session(session, root, HarvestTracker(mock=True), request_id="retry-safe-step")
    assert replay["idempotent_replay"] is True
    assert replay["session"]["current_time_ms"] == 250
    assert track["saved_count"] == 1


def test_previous_sample_restores_masks_and_replaces_the_saved_output(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "rewind.avi")
    session, root = create_session([str(video)], tmp_path / "outputs", "Rewind", [])
    mask = np.zeros((96, 128), dtype=bool)
    mask[20:65, 25:75] = True
    track = create_track(session, root, session["current_video_id"], 0, _mask_data(mask))
    tracker = HarvestTracker(mock=True)
    advance_session(session, root, tracker)
    advance_session(session, root, tracker)
    original_sample = next(item for item in session["samples"] if item["time_ms"] == 250)
    original_output = root / original_sample["relative_path"]
    assert original_output.is_file()

    result = rewind_session(session, root)

    assert result["to_time_ms"] == 250
    assert result["restored_track_ids"] == [track["id"]]
    assert result["removed_sample_count"] == 1
    assert session["current_time_ms"] == 250
    assert track["status"] == "active"
    assert track["current_time_ms"] == 250
    assert track["current_mask_path"].endswith("t000000000250.png")
    assert [item["time_ms"] for item in session["samples"]] == [0]
    assert not original_output.exists()
    assert Path(result["backup_dir"], original_output.name).is_file()

    corrected_mask = np.zeros((96, 128), dtype=bool)
    corrected_mask[22:67, 35:85] = True
    correct_track(session, root, track["id"], _mask_data(corrected_mask))
    replayed = advance_session(session, root, tracker)
    corrected_sample = replayed["committed"][0]
    assert corrected_sample["time_ms"] == 250
    assert corrected_sample["mask_bbox"] == [35, 22, 50, 45]
    assert track["saved_count"] == 2


def test_multiple_tracks_can_start_on_different_samples(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "multiple.avi")
    session, root = create_session([str(video)], tmp_path / "outputs", "Multiple", [])
    first_mask = np.zeros((96, 128), dtype=bool)
    first_mask[20:60, 20:60] = True
    first = create_track(session, root, session["current_video_id"], 0, _mask_data(first_mask))
    tracker = HarvestTracker(mock=True)
    advance_session(session, root, tracker)
    second_mask = np.zeros((96, 128), dtype=bool)
    second_mask[35:75, 75:115] = True
    second = create_track(session, root, session["current_video_id"], 250, _mask_data(second_mask))
    result = advance_session(session, root, tracker)
    assert {item["track_id"] for item in result["committed"]} == {first["id"], second["id"]}
    assert first["saved_count"] == 2
    assert second["saved_count"] == 1
    assert first["color"] != second["color"]


def test_delete_track_backs_up_masks_and_saved_pngs_without_reusing_id(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "delete.avi")
    session, root = create_session([str(video)], tmp_path / "outputs", "Delete", [])
    mask = np.zeros((96, 128), dtype=bool)
    mask[20:65, 25:75] = True
    track = create_track(session, root, session["current_video_id"], 0, _mask_data(mask))
    committed = advance_session(session, root, HarvestTracker(mock=True))["committed"][0]
    foreground = root / committed["relative_path"]
    live_masks = root / ".work" / "masks" / track["id"]
    assert foreground.is_file() and live_masks.is_dir()

    result = delete_track(session, root, track["id"])

    backup = Path(result["backup_dir"])
    assert result["deleted_saved_count"] == 1
    assert not session["tracks"] and not session["samples"]
    assert not foreground.exists() and not live_masks.exists()
    assert (backup / "deleted-track.json").is_file()
    assert (backup / "files" / committed["relative_path"]).is_file()
    assert (backup / "files" / ".work" / "masks" / track["id"]).is_dir()

    replacement = create_track(
        session,
        root,
        session["current_video_id"],
        session["current_time_ms"],
        _mask_data(mask),
    )
    assert replacement["id"] == "fruit-00002"


def test_delete_current_mask_pauses_track_until_replacement(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "delete-mask.avi")
    session, root = create_session([str(video)], tmp_path / "outputs", "Delete mask", [])
    mask = np.zeros((96, 128), dtype=bool)
    mask[20:65, 25:75] = True
    track = create_track(session, root, session["current_video_id"], 0, _mask_data(mask))
    original = root / track["current_mask_path"]

    result = delete_track_mask(session, root, track["id"])

    assert track["status"] == "needs_mask"
    assert track["current_mask_path"] is None
    assert not original.exists()
    assert list(Path(result["backup_dir"]).glob("*.png"))
    corrected = correct_track(session, root, track["id"], _mask_data(mask))
    assert corrected["status"] == "active"
    assert (root / corrected["current_mask_path"]).is_file()


def test_single_mask_api_saves_once_without_starting_tracker_or_advancing(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "single-mask.avi")
    output_parent = tmp_path / "harvests"
    client = TestClient(create_app(provider_name="mock"))
    created = client.post("/api/harvest/sessions", json={
        "video_paths": [str(video)],
        "output_parent": str(output_parent),
        "name": "Singles",
    })
    assert created.status_code == 200, created.text
    session = created.json()["session"]
    mask = np.zeros((96, 128), dtype=bool)
    mask[20:65, 25:75] = True

    saved = client.post(f"/api/harvest/sessions/{session['session_id']}/single-mask", json={
        "video_id": session["current_video_id"],
        "time_ms": 0,
        "mask_data": _mask_data(mask),
    })

    assert saved.status_code == 200, saved.text
    payload = saved.json()
    assert payload["sample"]["status"] == "saved"
    assert payload["session"]["current_time_ms"] == 0
    assert payload["track"]["status"] == "ended"
    assert payload["track"]["capture_mode"] == "single"
    assert not [item for item in payload["session"]["tracks"] if item["status"] == "active"]
    assert (output_parent / "Singles" / payload["sample"]["relative_path"]).is_file()


def test_two_invalid_samples_end_track_and_correction_resets_counter(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "reject.avi")
    session, root = create_session([str(video)], tmp_path / "outputs", "Reject", [])
    tiny = np.zeros((96, 128), dtype=bool)
    tiny[30:40, 30:40] = True
    track = create_track(session, root, session["current_video_id"], 0, _mask_data(tiny))
    tracker = HarvestTracker(mock=True)
    advance_session(session, root, tracker)
    assert track["status"] == "active" and track["invalid_count"] == 1
    valid = np.zeros_like(tiny)
    valid[20:65, 25:75] = True
    correct_track(session, root, track["id"], _mask_data(valid))
    assert track["invalid_count"] == 0
    advance_session(session, root, tracker)
    assert track["status"] == "active" and track["invalid_count"] == 0

    second = create_track(session, root, session["current_video_id"], 500, _mask_data(tiny))
    advance_session(session, root, tracker)
    assert second["status"] == "active" and second["invalid_count"] == 1
    advance_session(session, root, tracker)
    assert second["status"] == "ended"
    assert second["end_reason"] == "two consecutive invalid samples"


def test_cancel_checksum_and_protected_output_guards(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "guards.avi")
    protected = tmp_path / "outputs" / "Protected"
    protected.mkdir(parents=True)
    (protected / "unrelated.txt").write_text("keep", encoding="utf-8")
    try:
        create_session([str(video)], tmp_path / "outputs", "Protected", [])
    except FileExistsError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("An unrelated non-empty output folder was overwritten")

    session, root = create_session([str(video)], tmp_path / "outputs", "Resume", [])
    mask = np.zeros((96, 128), dtype=bool)
    mask[20:65, 25:75] = True
    create_track(session, root, session["current_video_id"], 0, _mask_data(mask))
    try:
        advance_session(session, root, HarvestTracker(mock=True), cancel_check=lambda: True)
    except HarvestCancelled:
        pass
    else:
        raise AssertionError("A cancelled tracking step continued")
    assert session["current_time_ms"] == 0 and not session["samples"]

    with video.open("ab") as stream:
        stream.write(b"changed")
    try:
        load_session(root, verify_sources=True)
    except ValueError as exc:
        assert "changed since" in str(exc)
    else:
        raise AssertionError("A changed source video was accepted on resume")


def test_harvest_api_job_is_resumable_and_idempotent(tmp_path: Path) -> None:
    video = _moving_video(tmp_path / "line.avi")
    client = TestClient(create_app(provider_name="mock"))
    created = client.post("/api/harvest/sessions", json={
        "video_paths": [str(video)],
        "output_parent": str(tmp_path / "harvests"),
        "name": "Selection",
    })
    assert created.status_code == 200, created.text
    assert Path(created.json()["session_dir"]) == (tmp_path / "harvests" / "Selection").resolve()
    session = created.json()["session"]
    session_id = session["session_id"]
    video_id = session["videos"][0]["video_id"]
    frame_response = client.get(f"/api/harvest/sessions/{session_id}/videos/{video_id}/frame?time_ms=0")
    assert frame_response.status_code == 200
    assert frame_response.headers["content-type"].startswith("image/jpeg")
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        media_response = client.get(session["videos"][0]["media_url"])
        assert media_response.status_code == 200
        assert media_response.headers["content-type"].startswith("video/mp4")
    proposal_started = client.post(f"/api/harvest/sessions/{session_id}/proposals", json={
        "video_id": video_id,
        "time_ms": 0,
        "request_id": "seed",
        "points": [{"x": 52, "y": 48, "label": 1}],
    })
    assert proposal_started.status_code == 200, proposal_started.text
    proposal_job = _wait_job(client, proposal_started.json()["job_id"])
    assert proposal_job["state"] == "complete", proposal_job
    proposal = proposal_job["result"]
    added = client.post(f"/api/harvest/sessions/{session_id}/tracks", json={
        "video_id": video_id,
        "time_ms": 0,
        "mask_data": proposal["proposals"][0]["mask_data"],
    })
    assert added.status_code == 200, added.text
    started = client.post(f"/api/harvest/sessions/{session_id}/advance", json={})
    assert started.status_code == 200
    job = _wait_job(client, started.json()["job_id"])
    assert job["state"] == "complete", job
    refreshed = client.get(f"/api/harvest/sessions/{session_id}").json()["session"]
    assert refreshed["current_time_ms"] == 250
    assert refreshed["samples"][0]["status"] == "saved"

    second_added = client.post(f"/api/harvest/sessions/{session_id}/tracks", json={
        "video_id": video_id,
        "time_ms": 250,
        "mask_data": proposal["proposals"][0]["mask_data"],
    })
    assert second_added.status_code == 200, second_added.text
    second_track_id = second_added.json()["track"]["id"]
    mask_deleted = client.delete(f"/api/harvest/sessions/{session_id}/tracks/{second_track_id}/mask")
    assert mask_deleted.status_code == 200, mask_deleted.text
    paused = next(item for item in mask_deleted.json()["session"]["tracks"] if item["id"] == second_track_id)
    assert paused["status"] == "needs_mask" and "mask_url" not in paused
    repaired = client.patch(f"/api/harvest/sessions/{session_id}/tracks/{second_track_id}", json={
        "mask_data": proposal["proposals"][0]["mask_data"],
    })
    assert repaired.status_code == 200, repaired.text
    assert repaired.json()["track"]["status"] == "active"
    deleted = client.delete(f"/api/harvest/sessions/{session_id}/tracks/{second_track_id}")
    assert deleted.status_code == 200, deleted.text
    assert Path(deleted.json()["backup_dir"]).is_dir()
    assert [item["id"] for item in deleted.json()["session"]["tracks"]] == ["fruit-00001"]

    reopened = client.post("/api/harvest/sessions/open", json={"session_dir": str(tmp_path / "harvests" / "Selection")})
    assert reopened.status_code == 200, reopened.text
    assert Path(reopened.json()["session_dir"]) == (tmp_path / "harvests" / "Selection").resolve()
    assert reopened.json()["session"]["session_id"] == session_id

    finished = client.post(f"/api/harvest/sessions/{session_id}/finish", json={})
    assert finished.status_code == 200, finished.text
    finish_job = _wait_job(client, finished.json()["job_id"])
    assert finish_job["state"] == "complete", finish_job
    generated = load_project(finish_job["result"]["project_dir"])
    assert generated["annotations"] == []
    assert all(image["status"] == "unlabelled" for image in generated["images"])
    assert client.get("/api/state").json()["project_dir"] == finish_job["result"]["project_dir"]
