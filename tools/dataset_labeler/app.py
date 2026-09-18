from __future__ import annotations

import os
import platform
import sys
import threading
import uuid
from collections import Counter
from importlib import metadata, util
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .core import (
    CLASS_DEFINITIONS,
    DEFAULT_SEGMENTATION_CLASS_NAMES,
    SEGMENTATION_CLASS_DEFINITIONS,
    atomic_write_bytes,
    create_or_update_project,
    find_image,
    image_abs_path,
    load_project,
    mask_path,
    project_path,
    project_summary,
    replace_image_annotations,
    safe_resolve,
    save_project,
    validate_class_definitions,
    validate_mask_bytes,
)
from .deblur_dataset import DeblurGenerationConfig, generate_deblur_dataset, preview_deblur, validate_deblur_dataset
from .detection_providers import DetectionProviderManager
from .exporters import export_coco, export_dataset
from .harvest import (
    HARVEST_INTERVAL_MS,
    HarvestCancelled,
    HarvestTracker,
    advance_session,
    browser_media_path,
    cached_frame,
    correct_track,
    create_session,
    create_track,
    delete_track,
    delete_track_mask,
    finish_session,
    load_session,
    public_session,
    reopen_session,
    rewind_session,
    save_single_mask,
    set_position,
    track_mask_path,
)
from .image_edits import ImageEditOperation, edit_project_image, erase_project_image_pixels
from .mask_ops import load_binary_mask
from .previews import generate_previews
from .providers import ProviderManager
from .splitting import create_split_manifest, ensure_split_manifest
from .synthetic import generate_synthetic_project
from .validation import write_validation_report
from .video_import import (
    DEFAULT_FRAME_INTERVAL_SECONDS,
    VIDEO_EXTENSIONS,
    VideoImportCancelled,
    create_class_definitions,
    create_video_project,
    inspect_videos,
)

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
REPO_ROOT = PACKAGE_DIR.parents[1]
REACT_DIST_DIR = REPO_ROOT / "frontend" / "dist"
REACT_INDEX = REACT_DIST_DIR / "index.html"


def frontend_index_path() -> Path:
    return REACT_INDEX if REACT_INDEX.exists() else STATIC_DIR / "index.html"


class ProjectCreateRequest(BaseModel):
    source_dir: str
    project_dir: str
    project_name: str | None = None
    annotation_mode: Literal["detection", "segmentation", "classification", "mixed"] = "mixed"
    class_names: list[str] = Field(default_factory=list)


class ProjectOpenRequest(BaseModel):
    project_dir: str


class InstancePayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str | int | None = None
    uuid: str | None = None
    type: Literal["fruit_box", "defect_mask", "foreground_mask"] | None = None
    annotation_kind: Literal["fruit_box", "defect_mask", "foreground_mask", "fruit_grade", "defect_region"] = "defect_mask"
    class_uuid: str | None = None
    export_index: int | None = None
    training_class_id: int | None = None
    category_id: int | None = None
    display_label: str | None = None
    name: str | None = None
    color: str | None = None
    parent_fruit_instance_id: str | None = None
    provenance: Literal[
        "manual",
        "sam2",
        "sam2_plus_manual",
        "mock",
        "mock_plus_manual",
        "sam3",
        "sam3_plus_manual",
        "grounding_dino",
        "grounding_dino_plus_manual",
        "synthetic",
    ] = "manual"
    prompt_metadata: dict[str, Any] = Field(default_factory=dict)
    review_status: str = "draft"
    visible: bool = True
    locked: bool = False
    notes: str = ""
    mask_data: str | None = None
    points: list[list[float]] | None = None
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)


class ImageUpdatePayload(BaseModel):
    status: Literal["unlabelled", "draft", "review", "approved", "excluded"] = "unlabelled"
    reviewed: bool = False
    excluded: bool = False
    exclusion_reason: str = ""
    fruit_id: str = ""
    view_id: str = ""
    split: Literal["train", "val", "test"] | None = None
    classification_class_uuid: str | None = None


class ImageSaveRequest(BaseModel):
    expected_revision: int | None = None
    image: ImageUpdatePayload
    annotations: list[InstancePayload] = Field(default_factory=list)


class BulkNoInstancesRequest(BaseModel):
    image_ids: list[str] = Field(min_length=1, max_length=10_000)
    expected_revisions: dict[str, int] = Field(default_factory=dict)


class ImageEditRequest(BaseModel):
    operation: ImageEditOperation
    expected_revision: int | None = None


class PixelEraseRequest(BaseModel):
    erase_mask_data: str = Field(min_length=1)
    expected_revision: int | None = None


class SplitRequest(BaseModel):
    ratios: dict[str, float] | None = None
    seed: int | None = None
    pilot_mode: bool = False
    freeze_test: bool = False
    confirm_frozen_change: bool = False
    dataset_version: str | None = None


class ExportRequest(BaseModel):
    output_dir: str | None = None
    minimum_iou: float = 0.98
    simplification_pixels: float = 0.5
    precision: int = 6
    force_split: bool = False
    allow_warnings: bool = True
    formats: list[Literal["yolo26_detection", "yolo26_segmentation", "dfine_detection", "synthetic_assets", "image_classification"]] | None = None


class VideoInspectRequest(BaseModel):
    video_paths: list[str] = Field(min_length=1)
    interval_seconds: Literal[0.5] = DEFAULT_FRAME_INTERVAL_SECONDS


class VideoImportRequest(BaseModel):
    video_paths: list[str] = Field(min_length=1)
    output_parent: str
    project_name: str = Field(min_length=1, max_length=120)
    annotation_mode: Literal["detection", "segmentation", "classification"]
    class_names: list[str] = Field(default_factory=list)
    interval_seconds: Literal[0.5] = DEFAULT_FRAME_INTERVAL_SECONDS
    ratios: dict[str, float] = Field(default_factory=lambda: {"train": 0.7, "val": 0.2, "test": 0.1})
    seed: int = 42


class SyntheticGenerateRequest(BaseModel):
    foreground_dir: str
    background_dir: str
    output_dir: str
    objects_per_image: int = Field(
        default=6,
        ge=1,
        le=300,
        description="Maximum objects per generated image; actual counts vary from zero to this value.",
    )
    total_images: int = Field(default=100, ge=2, le=100_000)
    seed: int = 42
    min_scale_percent: float = Field(default=10.0, gt=0)
    max_scale_percent: float = Field(default=17.0, gt=0)
    max_rotation_degrees: float = Field(default=360.0, ge=0, le=360)
    overlap_probability_percent: float = Field(
        default=30.0,
        ge=0,
        le=100,
        description="Chance that each foreground after the first is deliberately placed over another object.",
    )
    max_overlap_percent: float = Field(
        default=20.0,
        gt=0,
        le=80,
        description="Maximum fraction of any foreground's alpha pixels that may overlap another foreground.",
    )
    placement_attempts: int = Field(default=500, ge=1, le=10_000)


class DeblurDatasetRequest(BaseModel):
    source_dir: str
    output_dir: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    resume: bool = False
    source_index: int = 0
    variant: int = 0


def _run_synthetic_generation(
    payload: SyntheticGenerateRequest,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    return generate_synthetic_project(
        payload.foreground_dir,
        payload.background_dir,
        payload.output_dir,
        objects_per_image=payload.objects_per_image,
        total_images=payload.total_images,
        seed=payload.seed,
        min_scale=payload.min_scale_percent / 100.0,
        max_scale=payload.max_scale_percent / 100.0,
        max_rotation_degrees=payload.max_rotation_degrees,
        overlap_probability=payload.overlap_probability_percent / 100.0,
        max_overlap_fraction=payload.max_overlap_percent / 100.0,
        placement_attempts=payload.placement_attempts,
        progress_callback=progress_callback,
    )


class PromptPoint(BaseModel):
    x: float
    y: float
    label: Literal[0, 1]


class PredictionRequest(BaseModel):
    image_id: str
    request_id: str
    points: list[PromptPoint] = Field(default_factory=list)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    text: str = ""


class DetectionPredictionRequest(BaseModel):
    image_id: str
    request_id: str
    labels: list[str] = Field(default_factory=lambda: ["orange", "naartjie", "citrus fruit"], min_length=1, max_length=12)
    confidence: float = Field(default=0.30, ge=0.05, le=0.95)
    text_threshold: float = Field(default=0.25, ge=0.05, le=0.95)
    max_detections: int = Field(default=25, ge=1, le=100)


class HarvestSessionCreateRequest(BaseModel):
    video_paths: list[str] = Field(min_length=1)
    output_parent: str
    name: str = Field(min_length=1, max_length=120)
    segmentation_classes: list[dict[str, Any]] = Field(default_factory=list)
    interval_ms: Literal[250] = HARVEST_INTERVAL_MS


class HarvestSessionOpenRequest(BaseModel):
    session_dir: str


class HarvestPositionRequest(BaseModel):
    video_id: str
    time_ms: int = Field(ge=0)


class HarvestProposalRequest(BaseModel):
    video_id: str
    time_ms: int = Field(ge=0)
    request_id: str
    points: list[PromptPoint] = Field(default_factory=list)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)


class HarvestTrackRequest(BaseModel):
    video_id: str
    time_ms: int = Field(ge=0)
    mask_data: str = Field(min_length=1)
    prompt_metadata: dict[str, Any] = Field(default_factory=dict)


class HarvestCorrectionRequest(BaseModel):
    mask_data: str = Field(min_length=1)


class HarvestAdvanceRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=120)
    skip_track_ids: list[str] = Field(default_factory=list)
    end_track_ids: list[str] = Field(default_factory=list)


class DeriveFruitBoxRequest(BaseModel):
    foreground_uuid: str
    detection_class_uuid: str


class AppState:
    def __init__(self, project_dir: Path | None = None) -> None:
        self.project_dir = project_dir.expanduser().resolve() if project_dir else None
        self.current_provider_image_id: str | None = None

    def require_project_dir(self) -> Path:
        if not self.project_dir:
            raise HTTPException(status_code=404, detail="No project is open.")
        return self.project_dir

    def load_project(self, annotation_image_ids: set[str] | None = None) -> dict[str, Any]:
        return load_project(self.require_project_dir(), annotation_image_ids=annotation_image_ids)


def _client_project(project: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in project.items() if key not in {"annotations", "_dirty_annotation_image_ids", "_legacy_source"}}


def _client_instances(instances: list[dict[str, Any]], image_id: str) -> list[dict[str, Any]]:
    return [
        ({**instance, "mask_url": f"/api/masks/{image_id}/{instance['id']}"} if instance.get("annotation_kind") != "fruit_box" else dict(instance))
        for instance in instances
    ]


def _cuda_summary() -> dict[str, Any]:
    if "torch" not in sys.modules:
        available = util.find_spec("torch") is not None
        try:
            version = metadata.version("torch") if available else None
        except metadata.PackageNotFoundError:
            version = None
        return {
            "torch_available": available,
            "torch_version": version,
            "cuda_available": None,
            "probe_deferred": available,
        }
    try:
        torch = sys.modules["torch"]

        return {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        return {"torch_available": False, "cuda_available": False, "error": str(exc)}


def create_app(
    default_project_dir: Path | str | None = None,
    default_source_dir: Path | str | None = None,
    provider_name: str | None = None,
    detection_provider_name: str | None = None,
) -> FastAPI:
    project_dir = Path(default_project_dir).expanduser().resolve() if default_project_dir else None
    source_dir = Path(default_source_dir).expanduser().resolve() if default_source_dir else None
    state = AppState(project_dir if project_dir and project_path(project_dir).exists() else None)
    providers = ProviderManager(provider_name)
    detection_providers = DetectionProviderManager(
        detection_provider_name or ("mock" if provider_name == "mock" else None)
    )
    synthetic_jobs: dict[str, dict[str, Any]] = {}
    synthetic_jobs_lock = threading.Lock()
    synthetic_worker_lock = threading.Lock()
    deblur_jobs: dict[str, dict[str, Any]] = {}
    deblur_cancel_events: dict[str, threading.Event] = {}
    deblur_jobs_lock = threading.Lock()
    deblur_worker_lock = threading.Lock()
    video_jobs: dict[str, dict[str, Any]] = {}
    video_cancel_events: dict[str, threading.Event] = {}
    video_jobs_lock = threading.Lock()
    video_worker_lock = threading.Lock()
    harvest_roots: dict[str, Path] = {}
    harvest_sessions_lock = threading.Lock()
    harvest_jobs: dict[str, dict[str, Any]] = {}
    harvest_cancel_events: dict[str, threading.Event] = {}
    harvest_jobs_lock = threading.Lock()
    harvest_worker_lock = threading.Lock()
    harvest_tracker = HarvestTracker(mock=provider_name == "mock")
    if project_dir and source_dir and not project_path(project_dir).exists():
        create_or_update_project(source_dir, project_dir)
        state.project_dir = project_dir

    app = FastAPI(title="Data Labeller", version="3.2.0")
    if (REACT_DIST_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=REACT_DIST_DIR / "assets"), name="react-assets")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return frontend_index_path().read_text(encoding="utf-8")

    def require_harvest(session_id: str) -> tuple[dict[str, Any], Path]:
        with harvest_sessions_lock:
            root = harvest_roots.get(session_id)
        if root is None:
            raise HTTPException(status_code=404, detail="Unknown harvest session. Reopen its session folder first.")
        try:
            return load_session(root, verify_sources=False), root
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def start_harvest_job(session_id: str, action: str, operation: Any) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        cancel_event = threading.Event()
        with harvest_jobs_lock:
            harvest_jobs[job_id] = {
                "job_id": job_id,
                "session_id": session_id,
                "state": "queued",
                "stage": action,
                "percent": 0,
                "message": f"Waiting to {action}.",
                "result": None,
                "error": None,
            }
            harvest_cancel_events[job_id] = cancel_event

        def run() -> None:
            try:
                with harvest_worker_lock:
                    if cancel_event.is_set():
                        raise HarvestCancelled("Harvest operation was cancelled.")
                    with harvest_jobs_lock:
                        harvest_jobs[job_id].update({"state": "running", "percent": 15, "message": f"Working: {action}…"})
                    result = operation(cancel_event)
                with harvest_jobs_lock:
                    harvest_jobs[job_id].update({"state": "complete", "percent": 100, "message": f"{action.capitalize()} complete.", "result": result})
            except HarvestCancelled:
                with harvest_jobs_lock:
                    harvest_jobs[job_id].update({"state": "cancelled", "message": "Harvest operation cancelled."})
            except Exception as exc:
                with harvest_jobs_lock:
                    harvest_jobs[job_id].update({"state": "failed", "message": f"Could not {action}.", "error": str(exc)})

        threading.Thread(target=run, name=f"harvest-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "backend_version": "3.2.0",
            "python": platform.python_version(),
            "host_binding_default": "127.0.0.1",
            "project_open": state.project_dir is not None,
            "cuda": _cuda_summary(),
            "segmentation": providers.status(),
            "detection": detection_providers.status(),
            "harvest": harvest_tracker.status(),
        }

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        opened = state.project_dir is not None and project_path(state.project_dir).exists()
        project = state.load_project() if opened else None
        return {
            "opened": opened,
            "project_dir": str(state.project_dir) if state.project_dir else None,
            "detection_classes": project.get("detection_classes", CLASS_DEFINITIONS) if project else CLASS_DEFINITIONS,
            "segmentation_classes": project.get("segmentation_classes", SEGMENTATION_CLASS_DEFINITIONS) if project else SEGMENTATION_CLASS_DEFINITIONS,
            "project": _client_project(project) if project else None,
            "summary": project_summary(project) if project else None,
            "segmentation": providers.status(),
            "detection": detection_providers.status(),
        }

    @app.post("/api/project/create")
    def create_project(payload: ProjectCreateRequest) -> dict[str, Any]:
        try:
            project_root = Path(payload.project_dir).expanduser().resolve()
            is_new_project = not project_path(project_root).is_file()
            project = create_or_update_project(payload.source_dir, project_root, payload.project_name)
            if is_new_project:
                project["annotation_mode"] = payload.annotation_mode
            if is_new_project and payload.annotation_mode != "mixed":
                defaults = {
                    "detection": ["Object"],
                    "segmentation": DEFAULT_SEGMENTATION_CLASS_NAMES,
                    "classification": ["Good", "Defective"],
                }
                active_classes = create_class_definitions(
                    payload.class_names or defaults[payload.annotation_mode],
                    f"{project_root.name}:{payload.annotation_mode}",
                )
                class_key = {
                    "detection": "detection_classes",
                    "segmentation": "segmentation_classes",
                    "classification": "classification_classes",
                }[payload.annotation_mode]
                project[class_key] = active_classes
                if payload.annotation_mode == "detection":
                    project["classes"] = [dict(item) for item in active_classes]
            save_project(project, project_root)
            project = load_project(project_root)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        state.project_dir = project_root
        return {"project": _client_project(project), "summary": project_summary(project)}

    @app.post("/api/project/open")
    def open_project(payload: ProjectOpenRequest) -> dict[str, Any]:
        try:
            project_root = Path(payload.project_dir).expanduser().resolve()
            project = load_project(project_root)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        state.project_dir = project_root
        return {"project": _client_project(project), "summary": project_summary(project)}

    @app.post("/api/project/close")
    def close_project() -> dict[str, Any]:
        """Return to the project picker without changing or deleting project data."""
        providers.clear_image()
        state.current_provider_image_id = None
        state.project_dir = None
        return {"closed": True}

    @app.post("/api/videos/inspect")
    def inspect_selected_videos(payload: VideoInspectRequest) -> dict[str, Any]:
        try:
            return inspect_videos(payload.video_paths, interval_seconds=payload.interval_seconds)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/videos/jobs")
    def start_video_import_job(payload: VideoImportRequest) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        cancel_event = threading.Event()
        with video_jobs_lock:
            completed = [
                key for key, value in video_jobs.items()
                if value.get("state") in {"complete", "cancelled", "failed"}
            ]
            for stale_id in completed[:-40]:
                video_jobs.pop(stale_id, None)
                video_cancel_events.pop(stale_id, None)
            video_jobs[job_id] = {
                "job_id": job_id,
                "state": "queued",
                "completed": 0,
                "total": 0,
                "percent": 0,
                "stage": "queued",
                "message": "Waiting to import videos.",
                "result": None,
                "error": None,
            }
            video_cancel_events[job_id] = cancel_event

        def update(progress: dict[str, Any]) -> None:
            with video_jobs_lock:
                if job_id in video_jobs:
                    video_jobs[job_id].update(progress)

        def run() -> None:
            try:
                with video_worker_lock:
                    result = create_video_project(
                        payload.video_paths,
                        payload.output_parent,
                        payload.project_name,
                        payload.annotation_mode,
                        payload.class_names,
                        interval_seconds=payload.interval_seconds,
                        ratios=payload.ratios,
                        seed=payload.seed,
                        progress_callback=update,
                        cancel_check=cancel_event.is_set,
                    )
                # The creation wizard transitions directly into labeling. Keep
                # the server's active project in sync before clients observe a
                # completed job and request the first image.
                state.project_dir = Path(result["project_dir"]).expanduser().resolve()
                providers.clear_image()
                state.current_provider_image_id = None
                result["project"] = _client_project(result["project"])
                update({
                    "state": "complete",
                    "stage": "complete",
                    "completed": result["counts"]["frames"],
                    "total": result["counts"]["frames"],
                    "percent": 100,
                    "message": "Video dataset is ready.",
                    "result": result,
                })
            except VideoImportCancelled:
                update({"state": "cancelled", "stage": "cancelled", "message": "Video import was cancelled."})
            except Exception as exc:
                update({"state": "failed", "stage": "failed", "message": "Video import failed.", "error": str(exc)})

        threading.Thread(target=run, name=f"video-import-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/videos/jobs/{job_id}")
    def get_video_import_job(job_id: str) -> dict[str, Any]:
        with video_jobs_lock:
            job = video_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown video-import job.")
            return dict(job)

    @app.post("/api/videos/jobs/{job_id}/cancel")
    def cancel_video_import_job(job_id: str) -> dict[str, Any]:
        with video_jobs_lock:
            job = video_jobs.get(job_id)
            event = video_cancel_events.get(job_id)
            if job is None or event is None:
                raise HTTPException(status_code=404, detail="Unknown video-import job.")
            if job.get("state") in {"queued", "running"}:
                event.set()
                job.update({"stage": "cancelling", "message": "Cancellation requested; finishing the current frame."})
            return dict(job)

    @app.post("/api/harvest/sessions")
    def create_harvest_session(payload: HarvestSessionCreateRequest) -> dict[str, Any]:
        try:
            classes = payload.segmentation_classes
            if not classes and state.project_dir and project_path(state.project_dir).is_file():
                classes = state.load_project(annotation_image_ids=set()).get("segmentation_classes", [])
            classes = validate_class_definitions(classes or SEGMENTATION_CLASS_DEFINITIONS)
            session, root = create_session(payload.video_paths, payload.output_parent, payload.name, classes)
            with harvest_sessions_lock:
                harvest_roots[session["session_id"]] = root
            return {"session": public_session(session), "session_dir": str(root), "tracker": harvest_tracker.status()}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/harvest/sessions/open")
    def open_harvest_session(payload: HarvestSessionOpenRequest) -> dict[str, Any]:
        try:
            session = load_session(payload.session_dir, verify_sources=True)
            root = Path(session["session_root"])
            with harvest_sessions_lock:
                harvest_roots[session["session_id"]] = root
            return {"session": public_session(session), "session_dir": str(root), "tracker": harvest_tracker.status()}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/harvest/sessions/reopen-current-project")
    def reopen_current_project_harvest() -> dict[str, Any]:
        try:
            project = state.load_project(annotation_image_ids=set())
            if project.get("source_kind") != "harvested_foregrounds":
                raise ValueError("The open project was not created by a harvest session.")
            session_dir = project.get("harvest_session_dir")
            if session_dir:
                root = Path(session_dir).expanduser().resolve()
            else:
                manifest_value = project.get("harvest_manifest")
                if not manifest_value:
                    raise FileNotFoundError("This project does not record its original harvest session.")
                manifest_path = Path(manifest_value).expanduser().resolve()
                if manifest_path.name != "harvest-manifest.json" or not manifest_path.is_file():
                    raise FileNotFoundError("The original harvest manifest is missing.")
                root = manifest_path.parent
            session = load_session(root, verify_sources=True)
            expected_project = Path(session.get("project_dir", "")).expanduser().resolve()
            if expected_project != state.require_project_dir().resolve():
                raise ValueError("The harvest session belongs to a different segmentation project.")
            reopen_session(session, root)
            with harvest_sessions_lock:
                harvest_roots[session["session_id"]] = root
            return {"session": public_session(session), "session_dir": str(root), "tracker": harvest_tracker.status()}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/harvest/sessions/{session_id}")
    def get_harvest_session(session_id: str) -> dict[str, Any]:
        session, _root = require_harvest(session_id)
        return {"session": public_session(session), "tracker": harvest_tracker.status()}

    @app.post("/api/harvest/sessions/{session_id}/position")
    def position_harvest_session(session_id: str, payload: HarvestPositionRequest) -> dict[str, Any]:
        session, root = require_harvest(session_id)
        try:
            return {"session": public_session(set_position(session, root, payload.video_id, payload.time_ms))}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/harvest/sessions/{session_id}/previous")
    def previous_harvest_sample(session_id: str) -> dict[str, Any]:
        session, root = require_harvest(session_id)
        try:
            result = rewind_session(session, root)
            return {**result, "session": public_session(session)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/harvest/sessions/{session_id}/videos/{video_id}/media")
    def harvest_video_media(session_id: str, video_id: str) -> FileResponse:
        session, _root = require_harvest(session_id)
        try:
            video = next(item for item in session["videos"] if str(item["video_id"]) == video_id)
            path = browser_media_path(session, _root, video_id)
            if path.suffix.lower() not in VIDEO_EXTENSIONS or not path.is_file():
                raise FileNotFoundError("Harvest source video is unavailable.")
            return FileResponse(path)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/harvest/sessions/{session_id}/videos/{video_id}/frame")
    def harvest_video_frame(session_id: str, video_id: str, time_ms: int = 0) -> FileResponse:
        session, root = require_harvest(session_id)
        try:
            path, _metadata = cached_frame(session, root, video_id, time_ms)
            return FileResponse(path, media_type="image/jpeg")
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/harvest/sessions/{session_id}/proposals")
    def harvest_mask_proposals(session_id: str, payload: HarvestProposalRequest) -> dict[str, Any]:
        def operation(cancel_event: threading.Event) -> dict[str, Any]:
            session, root = require_harvest(session_id)
            if cancel_event.is_set():
                raise HarvestCancelled("Harvest operation was cancelled.")
            path, metadata_payload = cached_frame(session, root, payload.video_id, payload.time_ms)
            proposals = providers.predict(path, {
                "points": [point.model_dump() for point in payload.points],
                "box": payload.box,
                "text": "",
            })
            if cancel_event.is_set():
                raise HarvestCancelled("Harvest operation was cancelled.")
            return {
                "request_id": payload.request_id,
                "proposals": proposals,
                "frame": metadata_payload,
                "status": providers.status(),
            }
        return start_harvest_job(session_id, "create whole-fruit mask", operation)

    @app.post("/api/harvest/sessions/{session_id}/detections")
    def harvest_fruit_detections(session_id: str, payload: HarvestPositionRequest) -> dict[str, Any]:
        def operation(cancel_event: threading.Event) -> dict[str, Any]:
            session, root = require_harvest(session_id)
            if cancel_event.is_set():
                raise HarvestCancelled("Harvest operation was cancelled.")
            path, metadata_payload = cached_frame(session, root, payload.video_id, payload.time_ms)
            proposals = detection_providers.predict(path, {
                "labels": ["naartjie", "orange", "citrus fruit"],
                "confidence": 0.25,
                "text_threshold": 0.25,
                "max_detections": 50,
            })
            if cancel_event.is_set():
                raise HarvestCancelled("Harvest operation was cancelled.")
            return {"proposals": proposals, "frame": metadata_payload, "status": detection_providers.status()}
        return start_harvest_job(session_id, "find naartjies", operation)

    @app.post("/api/harvest/sessions/{session_id}/tracks")
    def add_harvest_track(session_id: str, payload: HarvestTrackRequest) -> dict[str, Any]:
        session, root = require_harvest(session_id)
        try:
            track = create_track(session, root, payload.video_id, payload.time_ms, payload.mask_data, payload.prompt_metadata)
            public = public_session(session)
            return {"track": next(item for item in public["tracks"] if item["id"] == track["id"]), "session": public}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/harvest/sessions/{session_id}/single-mask")
    def add_single_harvest_mask(session_id: str, payload: HarvestTrackRequest) -> dict[str, Any]:
        session, root = require_harvest(session_id)
        try:
            track, sample = save_single_mask(
                session,
                root,
                payload.video_id,
                payload.time_ms,
                payload.mask_data,
                payload.prompt_metadata,
            )
            public = public_session(session)
            return {
                "track": next(item for item in public["tracks"] if item["id"] == track["id"]),
                "sample": sample,
                "session": public,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/harvest/sessions/{session_id}/tracks/{track_id}")
    def patch_harvest_track(session_id: str, track_id: str, payload: HarvestCorrectionRequest) -> dict[str, Any]:
        session, root = require_harvest(session_id)
        try:
            track = correct_track(session, root, track_id, payload.mask_data)
            public = public_session(session)
            return {"track": next(item for item in public["tracks"] if item["id"] == track["id"]), "session": public}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/harvest/sessions/{session_id}/tracks/{track_id}")
    def delete_harvest_track(session_id: str, track_id: str) -> dict[str, Any]:
        session, root = require_harvest(session_id)
        try:
            result = delete_track(session, root, track_id)
            return {**result, "session": public_session(session)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/harvest/sessions/{session_id}/tracks/{track_id}/mask")
    def harvest_track_mask(session_id: str, track_id: str) -> FileResponse:
        session, root = require_harvest(session_id)
        try:
            return FileResponse(track_mask_path(session, root, track_id), media_type="image/png")
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/harvest/sessions/{session_id}/tracks/{track_id}/mask")
    def delete_harvest_track_mask(session_id: str, track_id: str) -> dict[str, Any]:
        session, root = require_harvest(session_id)
        try:
            result = delete_track_mask(session, root, track_id)
            return {**result, "session": public_session(session)}
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/harvest/sessions/{session_id}/advance")
    def advance_harvest(session_id: str, payload: HarvestAdvanceRequest) -> dict[str, Any]:
        def operation(cancel_event: threading.Event) -> dict[str, Any]:
            session, root = require_harvest(session_id)
            return advance_session(
                session,
                root,
                harvest_tracker,
                skip_track_ids=set(payload.skip_track_ids),
                end_track_ids=set(payload.end_track_ids),
                request_id=payload.request_id,
                cancel_check=cancel_event.is_set,
            )
        return start_harvest_job(session_id, "advance fruit tracks", operation)

    @app.post("/api/harvest/sessions/{session_id}/finish")
    def finish_harvest(session_id: str) -> dict[str, Any]:
        def operation(_cancel_event: threading.Event) -> dict[str, Any]:
            session, root = require_harvest(session_id)
            result = finish_session(session, root)
            state.project_dir = Path(result["project_dir"])
            providers.clear_image()
            return {
                "project_dir": result["project_dir"],
                "project": _client_project(result["project"]),
                "summary": project_summary(result["project"]),
            }
        return start_harvest_job(session_id, "create segmentation project", operation)

    @app.get("/api/harvest/jobs/{job_id}")
    def get_harvest_job(job_id: str) -> dict[str, Any]:
        with harvest_jobs_lock:
            job = harvest_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown harvest job.")
            return dict(job)

    @app.post("/api/harvest/jobs/{job_id}/cancel")
    def cancel_harvest_job(job_id: str) -> dict[str, Any]:
        with harvest_jobs_lock:
            job = harvest_jobs.get(job_id)
            event = harvest_cancel_events.get(job_id)
            if job is None or event is None:
                raise HTTPException(status_code=404, detail="Unknown harvest job.")
            if job["state"] in {"queued", "running"}:
                event.set()
                job["message"] = "Cancellation requested; completed PNGs remain safe."
            return dict(job)

    @app.post("/api/synthetic/generate")
    def generate_synthetic(payload: SyntheticGenerateRequest) -> dict[str, Any]:
        """Generate composites without requiring an annotation project to be open."""
        try:
            return _run_synthetic_generation(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/synthetic/jobs")
    def start_synthetic_job(payload: SyntheticGenerateRequest) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        with synthetic_jobs_lock:
            completed_jobs = [
                key for key, value in synthetic_jobs.items()
                if value.get("state") in {"complete", "failed"}
            ]
            for stale_id in completed_jobs[:-40]:
                synthetic_jobs.pop(stale_id, None)
            synthetic_jobs[job_id] = {
                "job_id": job_id,
                "state": "queued",
                "completed": 0,
                "total": payload.total_images,
                "percent": 0,
                "stage": "queued",
                "message": "Waiting to start…",
                "result": None,
                "error": None,
            }

        def update(progress: dict[str, Any]) -> None:
            with synthetic_jobs_lock:
                if job_id in synthetic_jobs:
                    synthetic_jobs[job_id].update(progress)

        def run() -> None:
            try:
                with synthetic_worker_lock:
                    update({"state": "running", "stage": "preparing", "message": "Starting generation…"})
                    result = _run_synthetic_generation(payload, update)
                update({
                    "state": "complete",
                    "completed": payload.total_images,
                    "total": payload.total_images,
                    "percent": 100,
                    "stage": "complete",
                    "message": "Synthetic dataset is ready.",
                    "result": result,
                })
            except Exception as exc:
                update({
                    "state": "failed",
                    "stage": "failed",
                    "message": "Generation failed.",
                    "error": str(exc),
                })

        threading.Thread(target=run, name=f"synthetic-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/synthetic/jobs/{job_id}")
    def get_synthetic_job(job_id: str) -> dict[str, Any]:
        with synthetic_jobs_lock:
            job = synthetic_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown synthetic-generation job.")
            return dict(job)

    @app.post("/api/deblur/preview")
    def deblur_preview(payload: DeblurDatasetRequest) -> dict[str, Any]:
        try:
            config = DeblurGenerationConfig.from_mapping(payload.config)
            return preview_deblur(payload.source_dir, config, payload.source_index, payload.variant)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/deblur/validate")
    def deblur_validate(payload: DeblurDatasetRequest) -> dict[str, Any]:
        try:
            return validate_deblur_dataset(payload.output_dir)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/deblur/jobs")
    def start_deblur_job(payload: DeblurDatasetRequest) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        cancel_event = threading.Event()
        config = DeblurGenerationConfig.from_mapping(payload.config)
        try:
            config.validate()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with deblur_jobs_lock:
            completed = [key for key, value in deblur_jobs.items() if value.get("state") in {"complete", "complete_with_errors", "cancelled", "failed"}]
            for stale_id in completed[:-40]:
                deblur_jobs.pop(stale_id, None)
                deblur_cancel_events.pop(stale_id, None)
            deblur_jobs[job_id] = {
                "job_id": job_id, "state": "queued", "completed": 0, "total": 0, "percent": 0,
                "stage": "queued", "message": "Waiting to generate paired blur data.", "result": None, "error": None,
            }
            deblur_cancel_events[job_id] = cancel_event

        def update(progress: dict[str, Any]) -> None:
            with deblur_jobs_lock:
                if job_id in deblur_jobs:
                    deblur_jobs[job_id].update(progress)

        def run() -> None:
            try:
                with deblur_worker_lock:
                    update({"state": "running", "stage": "preparing", "message": "Scanning and grouping source images."})
                    result = generate_deblur_dataset(
                        payload.source_dir, payload.output_dir, config, resume=payload.resume,
                        progress_callback=update, cancel_check=cancel_event.is_set,
                    )
                update({
                    "state": result["state"], "completed": result["generated_pairs"], "total": result["planned_pairs"],
                    "percent": round(result["generated_pairs"] * 100 / max(1, result["planned_pairs"])),
                    "stage": result["state"], "message": "Generation cancelled." if result["state"] == "cancelled" else "Paired blur dataset is ready.",
                    "result": result,
                })
            except Exception as exc:
                update({"state": "failed", "stage": "failed", "message": "Paired blur generation failed.", "error": str(exc)})

        threading.Thread(target=run, name=f"deblur-dataset-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/deblur/jobs/{job_id}")
    def get_deblur_job(job_id: str) -> dict[str, Any]:
        with deblur_jobs_lock:
            job = deblur_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Unknown paired-blur generation job.")
            return dict(job)

    @app.post("/api/deblur/jobs/{job_id}/cancel")
    def cancel_deblur_job(job_id: str) -> dict[str, Any]:
        with deblur_jobs_lock:
            event = deblur_cancel_events.get(job_id)
            job = deblur_jobs.get(job_id)
            if event is None or job is None:
                raise HTTPException(status_code=404, detail="Unknown paired-blur generation job.")
            if job.get("state") in {"queued", "running"}:
                event.set()
                job.update({"message": "Cancellation requested; finishing the current file.", "stage": "cancelling"})
            return dict(job)

    @app.patch("/api/project/settings")
    def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
        project = state.load_project()
        try:
            if "export_settings" in payload:
                project["export_settings"].update(payload["export_settings"])
            if "detection_classes" in payload or "classes" in payload:
                project["detection_classes"] = validate_class_definitions(list(payload.get("detection_classes") or payload["classes"]))
                project["classes"] = [dict(item) for item in project["detection_classes"]]
            if "segmentation_classes" in payload:
                project["segmentation_classes"] = validate_class_definitions(list(payload["segmentation_classes"]))
            if "classification_classes" in payload:
                project["classification_classes"] = validate_class_definitions(list(payload["classification_classes"]))
            save_project(project, state.require_project_dir())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"project": _client_project(load_project(state.require_project_dir()))}

    @app.get("/api/images")
    def list_images() -> dict[str, Any]:
        # Per-image annotation records are authoritative for status and
        # revision. Load them before returning the sidebar collection so an
        # older manifest entry cannot display an approved image as unlabelled.
        project = state.load_project()
        return {"images": project["images"], "last_image_id": project.get("last_image_id")}

    @app.post("/api/images/bulk-no-instances")
    def bulk_no_instances(payload: BulkNoInstancesRequest) -> dict[str, Any]:
        image_ids = list(dict.fromkeys(str(image_id) for image_id in payload.image_ids))
        project = state.load_project(annotation_image_ids=set(image_ids))
        by_id = {str(image["id"]): image for image in project["images"]}
        missing = [image_id for image_id in image_ids if image_id not in by_id]
        if missing:
            raise HTTPException(status_code=404, detail=f"Unknown image id(s): {', '.join(missing[:5])}")
        conflicts = [
            f"{image_id} (expected {payload.expected_revisions[image_id]}, current {by_id[image_id].get('revision', 0)})"
            for image_id in image_ids
            if image_id in payload.expected_revisions
            and int(payload.expected_revisions[image_id]) != int(by_id[image_id].get("revision", 0))
        ]
        if conflicts:
            raise HTTPException(
                status_code=409,
                detail="Bulk update stopped before making changes because annotations changed: " + "; ".join(conflicts[:5]),
            )

        annotation_counts = Counter(str(item["image_id"]) for item in project.get("annotations", []))
        changed = 0
        try:
            for image_id in image_ids:
                image = by_id[image_id]
                already_negative = (
                    image.get("status") == "approved"
                    and not image.get("excluded")
                    and annotation_counts[image_id] == 0
                )
                if already_negative:
                    continue
                replace_image_annotations(
                    project,
                    image_id,
                    [],
                    {"status": "approved", "reviewed": True, "excluded": False, "exclusion_reason": ""},
                    int(image.get("revision", 0)),
                )
                changed += 1
            if changed:
                save_project(project, state.require_project_dir())
            refreshed = load_project(state.require_project_dir())
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        refreshed_by_id = {str(image["id"]): image for image in refreshed["images"]}
        return {
            "images": [refreshed_by_id[image_id] for image_id in image_ids],
            "selected_count": len(image_ids),
            "changed_count": changed,
            "summary": project_summary(refreshed),
        }

    @app.get("/api/images/{image_id}")
    def get_image(image_id: str) -> dict[str, Any]:
        project = state.load_project(annotation_image_ids={image_id})
        try:
            image = find_image(project, image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if state.current_provider_image_id and state.current_provider_image_id != image_id:
            providers.clear_image()
        state.current_provider_image_id = image_id
        instances = [item for item in project.get("annotations", []) if str(item["image_id"]) == image_id]
        return {"image": image, "annotations": _client_instances(instances, image_id)}

    @app.post("/api/images/{image_id}/activate")
    def activate_image(image_id: str) -> dict[str, Any]:
        """Persist the displayed image as the next project resume position."""
        project = state.load_project(annotation_image_ids={image_id})
        try:
            find_image(project, image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if str(project.get("last_image_id") or "") != image_id:
            project["last_image_id"] = image_id
            save_project(project, state.require_project_dir())
        return {"last_image_id": image_id}

    @app.put("/api/images/{image_id}")
    def save_image(image_id: str, payload: ImageSaveRequest) -> dict[str, Any]:
        project = state.load_project(annotation_image_ids={image_id})
        try:
            try:
                image = replace_image_annotations(
                    project,
                    image_id,
                    [item.model_dump(exclude_none=True) for item in payload.annotations],
                    payload.image.model_dump(),
                    payload.expected_revision,
                )
                save_project(project, state.require_project_dir())
            except RuntimeError:
                current = find_image(project, image_id)
                current_instances = [
                    item for item in project.get("annotations", [])
                    if str(item["image_id"]) == image_id
                ]
                desired = payload.image.model_dump()
                metadata_fields = (
                    "status",
                    "reviewed",
                    "excluded",
                    "exclusion_reason",
                    "fruit_id",
                    "view_id",
                    "split",
                    "classification_class_uuid",
                )
                duplicate_approved_negative = (
                    not payload.annotations
                    and not current_instances
                    and desired["status"] == "approved"
                    and all(current.get(field) == desired.get(field) for field in metadata_fields)
                )
                if not duplicate_approved_negative:
                    raise
            refreshed = load_project(state.require_project_dir())
            image = find_image(refreshed, image_id)
            instances = [item for item in refreshed.get("annotations", []) if str(item["image_id"]) == image_id]
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "image": image,
            "annotations": _client_instances(instances, image_id),
            "summary": project_summary(refreshed),
        }

    @app.post("/api/images/{image_id}/edit")
    def edit_image(image_id: str, payload: ImageEditRequest) -> dict[str, Any]:
        try:
            result = edit_project_image(
                state.require_project_dir(),
                image_id,
                payload.operation,
                payload.expected_revision,
            )
            providers.clear_image()
            state.current_provider_image_id = None
            refreshed = load_project(state.require_project_dir())
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "image": result["image"],
            "annotations": _client_instances(result["annotations"], image_id),
            "backup_path": result["backup_path"],
            "summary": project_summary(refreshed),
        }

    @app.post("/api/images/{image_id}/erase-pixels")
    def erase_image_pixels(image_id: str, payload: PixelEraseRequest) -> dict[str, Any]:
        try:
            result = erase_project_image_pixels(
                state.require_project_dir(),
                image_id,
                payload.erase_mask_data,
                payload.expected_revision,
            )
            providers.clear_image()
            state.current_provider_image_id = None
            refreshed = load_project(state.require_project_dir())
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "image": result["image"],
            "annotations": _client_instances(result["annotations"], image_id),
            "backup_path": result["backup_path"],
            "erased_pixel_count": result["erased_pixel_count"],
            "converted_to_png": result["converted_to_png"],
            "summary": project_summary(refreshed),
        }

    @app.post("/api/images/{image_id}/derive-fruit-box")
    def derive_fruit_box(image_id: str, payload: DeriveFruitBoxRequest) -> dict[str, Any]:
        """Explicitly derive a fruit_box from a confirmed foreground mask."""
        project = state.load_project(annotation_image_ids={image_id})
        try:
            image = find_image(project, image_id)
            instances = [item for item in project.get("annotations", []) if str(item["image_id"]) == image_id]
            foreground = next(
                item for item in instances
                if str(item.get("uuid")) == payload.foreground_uuid and item.get("annotation_kind") == "foreground_mask"
            )
            mask = load_binary_mask(project, foreground)
            ys, xs = mask.nonzero()
            if not len(xs):
                raise ValueError("Confirmed foreground mask is empty.")
            bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
            class_item = next(item for item in project["detection_classes"] if str(item["uuid"]) == payload.detection_class_uuid)
            instances.append({
                "id": str(uuid.uuid4()),
                "type": "fruit_box",
                "annotation_kind": "fruit_box",
                "class_uuid": class_item["uuid"],
                "export_index": class_item["export_index"],
                "bbox": bbox,
                "source": "import",
                "prompt_metadata": {"derived_from_foreground_uuid": payload.foreground_uuid, "confirmed": True},
            })
            replace_image_annotations(project, image_id, instances, {"status": image.get("status", "draft")}, int(image.get("revision", 0)))
            save_project(project, state.require_project_dir())
            refreshed = load_project(state.require_project_dir(), annotation_image_ids={image_id})
            return {
                "image": find_image(refreshed, image_id),
                "annotations": _client_instances(
                    [item for item in refreshed.get("annotations", []) if str(item["image_id"]) == image_id], image_id
                ),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/image-file/{image_id}")
    def image_file(image_id: str) -> FileResponse:
        project = state.load_project(annotation_image_ids=set())
        try:
            image = find_image(project, image_id)
            path = image_abs_path(project, image)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Image file is missing: {path}")
        return FileResponse(path)

    @app.get("/api/thumbnails/{image_id}")
    def thumbnail(image_id: str) -> FileResponse:
        project = state.load_project(annotation_image_ids=set())
        try:
            image = find_image(project, image_id)
            source = image_abs_path(project, image)
            destination = safe_resolve(state.require_project_dir(), f"thumbnails/{image_id}.jpg")
            if not destination.exists() or destination.stat().st_mtime_ns < source.stat().st_mtime_ns:
                from PIL import Image

                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as opened:
                    preview = opened.convert("RGB")
                    preview.thumbnail((240, 160))
                    preview.save(destination, "JPEG", quality=82, optimize=True)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(destination, media_type="image/jpeg")

    @app.get("/api/masks/{image_id}/{instance_id}")
    def get_mask(image_id: str, instance_id: str) -> FileResponse:
        project = state.load_project(annotation_image_ids={image_id})
        try:
            find_image(project, image_id)
            instance = next(
                item for item in project.get("annotations", [])
                if str(item["image_id"]) == image_id and str(item["id"]) == instance_id
            )
            destination = safe_resolve(state.require_project_dir(), instance["mask_path"])
        except (KeyError, StopIteration, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Mask instance was not found.") from exc
        return FileResponse(destination, media_type="image/png")

    @app.put("/api/masks/{image_id}/{instance_id}")
    async def put_mask(image_id: str, instance_id: str, request: Request) -> dict[str, Any]:
        project = state.load_project(annotation_image_ids={image_id})
        try:
            image = find_image(project, image_id)
            if not any(str(item["image_id"]) == image_id and str(item["id"]) == instance_id for item in project.get("annotations", [])):
                raise ValueError("Instance does not exist; save its metadata first.")
            payload = await request.body()
            cleaned = validate_mask_bytes(payload, int(image["width"]), int(image["height"]))
            destination = mask_path(state.require_project_dir(), image_id, instance_id)
            atomic_write_bytes(destination, cleaned)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "mask_url": f"/api/masks/{image_id}/{instance_id}"}

    @app.get("/api/segmentation/status")
    def segmentation_status() -> dict[str, Any]:
        return providers.status()

    @app.post("/api/segmentation/load")
    def segmentation_load() -> dict[str, Any]:
        status = providers.load()
        if status["state"] == "failed":
            raise HTTPException(status_code=503, detail=status.get("error") or "Provider load failed.")
        return status

    @app.post("/api/segmentation/predict")
    def segmentation_predict(payload: PredictionRequest) -> dict[str, Any]:
        project = state.load_project(annotation_image_ids=set())
        if project.get("annotation_mode") == "classification":
            raise HTTPException(status_code=400, detail="Mask proposals are not used in Classification projects.")
        try:
            image = find_image(project, payload.image_id)
            path = image_abs_path(project, image)
            proposals = providers.predict(
                path,
                {
                    "points": [point.model_dump() for point in payload.points],
                    "box": payload.box,
                    "text": payload.text,
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"request_id": payload.request_id, "image_id": payload.image_id, "proposals": proposals, "status": providers.status()}

    @app.get("/api/detection/status")
    def detection_status() -> dict[str, Any]:
        return detection_providers.status()

    @app.post("/api/detection/load")
    def detection_load() -> dict[str, Any]:
        status = detection_providers.load()
        if status["state"] in {"failed", "unavailable"}:
            raise HTTPException(status_code=503, detail=status.get("error") or "Detection provider load failed.")
        return status

    @app.post("/api/detection/predict")
    def detection_predict(payload: DetectionPredictionRequest) -> dict[str, Any]:
        project = state.load_project(annotation_image_ids=set())
        if project.get("annotation_mode") in {"segmentation", "classification"}:
            raise HTTPException(status_code=400, detail="AI box proposals are available in Detection and mixed projects.")
        try:
            image = find_image(project, payload.image_id)
            path = image_abs_path(project, image)
            proposals = detection_providers.predict(
                path,
                {
                    "labels": payload.labels,
                    "confidence": payload.confidence,
                    "text_threshold": payload.text_threshold,
                    "max_detections": payload.max_detections,
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "request_id": payload.request_id,
            "image_id": payload.image_id,
            "proposals": proposals,
            "status": detection_providers.status(),
        }

    @app.post("/api/split")
    def split_project(payload: SplitRequest) -> dict[str, Any]:
        try:
            return create_split_manifest(
                state.require_project_dir(),
                payload.ratios,
                payload.seed,
                pilot_mode=payload.pilot_mode,
                freeze_test=payload.freeze_test,
                confirm_frozen_change=payload.confirm_frozen_change,
                dataset_version=payload.dataset_version,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/split")
    def get_split() -> dict[str, Any]:
        try:
            return ensure_split_manifest(state.require_project_dir())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/validate")
    def validate(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return write_validation_report(state.require_project_dir(), (payload or {}).get("output_dir"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/export/coco")
    def export_coco_endpoint(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return export_coco(state.require_project_dir(), (payload or {}).get("output_dir"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/export/dataset")
    @app.post("/api/export/yolo")
    def export_dataset_endpoint(payload: ExportRequest) -> dict[str, Any]:
        try:
            output_dir = payload.output_dir or str(state.require_project_dir() / "exports")
            return export_dataset(
                state.require_project_dir(),
                output_dir,
                minimum_iou=payload.minimum_iou,
                simplification_pixels=payload.simplification_pixels,
                precision=payload.precision,
                force_split=payload.force_split,
                allow_warnings=payload.allow_warnings,
                formats=payload.formats,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/previews")
    def previews(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            output_dir = payload.get("output_dir") or state.require_project_dir() / "exports" / "previews"
            return generate_previews(state.require_project_dir(), output_dir, int(payload.get("count", 25)), int(payload.get("seed", 42)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dialog/directory")
    def directory_dialog(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(title=(payload or {}).get("title", "Select folder"))
            root.destroy()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not open folder picker: {exc}") from exc
        return {"path": selected}

    @app.post("/api/dialog/videos")
    def video_file_dialog(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            patterns = " ".join(f"*{extension}" for extension in sorted(VIDEO_EXTENSIONS))
            selected = filedialog.askopenfilenames(
                title=(payload or {}).get("title", "Select videos"),
                filetypes=[("Supported videos", patterns), ("All files", "*.*")],
            )
            root.destroy()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not open video picker: {exc}") from exc
        return {"paths": list(selected)}

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    def spa_fallback(full_path: str) -> str:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        return frontend_index_path().read_text(encoding="utf-8")

    return app
