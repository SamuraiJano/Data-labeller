from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .core import (
    CLASS_DEFINITIONS,
    annotations_for_image,
    create_or_update_project,
    find_image,
    image_abs_path,
    load_project,
    project_path,
    project_summary,
    replace_image_annotations,
    save_project,
)
from .exporters import export_coco, export_yolo
from .previews import generate_previews
from .splitting import create_split_manifest, ensure_split_manifest
from .validation import write_validation_report

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
REPO_ROOT = PACKAGE_DIR.parents[1]
REACT_DIST_DIR = REPO_ROOT / "frontend" / "dist"
REACT_INDEX = REACT_DIST_DIR / "index.html"


def frontend_index_path() -> Path:
    return REACT_INDEX if REACT_INDEX.exists() else STATIC_DIR / "index.html"


class AppState:
    def __init__(self, project_dir: Path | None = None) -> None:
        self.project_dir = project_dir.expanduser().resolve() if project_dir else None

    def require_project_dir(self) -> Path:
        if not self.project_dir:
            raise HTTPException(status_code=404, detail="No project is open.")
        return self.project_dir

    def load_project(self) -> dict[str, Any]:
        return load_project(self.require_project_dir())


def create_app(default_project_dir: Path | str | None = None, default_source_dir: Path | str | None = None) -> FastAPI:
    project_dir = Path(default_project_dir).expanduser().resolve() if default_project_dir else None
    source_dir = Path(default_source_dir).expanduser().resolve() if default_source_dir else None
    state = AppState(project_dir if project_dir and project_path(project_dir).exists() else None)

    if project_dir and source_dir and not project_path(project_dir).exists():
        create_or_update_project(source_dir, project_dir)
        state.project_dir = project_dir

    app = FastAPI(title="Skripsie Segmentation Labeler", version="0.1.0")
    if (REACT_DIST_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=REACT_DIST_DIR / "assets"), name="react-assets")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return frontend_index_path().read_text(encoding="utf-8")

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        opened = state.project_dir is not None and project_path(state.project_dir).exists()
        project = state.load_project() if opened else None
        return {
            "opened": opened,
            "project_dir": str(state.project_dir) if state.project_dir else None,
            "classes": CLASS_DEFINITIONS,
            "project": project,
            "summary": project_summary(project) if project else None,
        }

    @app.post("/api/project/create")
    def create_project(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            source_dir = Path(payload["source_dir"]).expanduser().resolve()
            project_dir = Path(payload["project_dir"]).expanduser().resolve()
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Missing required field: {exc}") from exc
        project = create_or_update_project(source_dir, project_dir, payload.get("project_name"))
        state.project_dir = project_dir
        return {"project": project, "summary": project_summary(project)}

    @app.post("/api/project/open")
    def open_project(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            project_dir = Path(payload["project_dir"]).expanduser().resolve()
            project = load_project(project_dir)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        state.project_dir = project_dir
        return {"project": project, "summary": project_summary(project)}

    @app.get("/api/images")
    def list_images() -> dict[str, Any]:
        project = state.load_project()
        return {"images": project["images"], "last_image_id": project.get("last_image_id")}

    @app.get("/api/images/{image_id}")
    def get_image(image_id: int) -> dict[str, Any]:
        project = state.load_project()
        try:
            image = find_image(project, image_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        project["last_image_id"] = image_id
        save_project(project, state.require_project_dir())
        return {"image": image, "annotations": annotations_for_image(project, image_id)}

    @app.put("/api/images/{image_id}")
    def save_image(image_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        project = state.load_project()
        try:
            image = replace_image_annotations(
                project,
                image_id,
                payload.get("annotations", []),
                payload.get("image", {}),
            )
            save_project(project, state.require_project_dir())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "image": image,
            "annotations": annotations_for_image(project, image_id),
            "summary": project_summary(project),
        }

    @app.get("/api/image-file/{image_id}")
    def image_file(image_id: int) -> FileResponse:
        project = state.load_project()
        try:
            image = find_image(project, image_id)
            path = image_abs_path(project, image)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Image file is missing: {path}")
        return FileResponse(path)

    @app.post("/api/split")
    def split_project(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        try:
            manifest = create_split_manifest(
                state.require_project_dir(),
                payload.get("ratios"),
                payload.get("seed"),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return manifest

    @app.get("/api/split")
    def get_split() -> dict[str, Any]:
        try:
            return ensure_split_manifest(state.require_project_dir())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/validate")
    def validate(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        try:
            return write_validation_report(state.require_project_dir(), payload.get("output_dir"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/export/coco")
    def export_coco_endpoint(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        try:
            return export_coco(state.require_project_dir(), payload.get("output_dir"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/export/yolo")
    def export_yolo_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            output_dir = payload.get("output_dir")
            if not output_dir:
                output_dir = state.require_project_dir().parent / "exports" / "yolo_seg"
            return export_yolo(
                state.require_project_dir(),
                output_dir,
                include_hsv=bool(payload.get("include_hsv", True)),
                force_split=bool(payload.get("force_split", False)),
                allow_invalid=bool(payload.get("allow_invalid", False)),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/previews")
    def previews(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            output_dir = payload.get("output_dir") or state.require_project_dir().parent / "exports" / "previews"
            return generate_previews(
                state.require_project_dir(),
                output_dir,
                count=int(payload.get("count", 25)),
                seed=int(payload.get("seed", 42)),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/dialog/directory")
    def directory_dialog(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(title=payload.get("title", "Select folder"))
            root.destroy()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not open folder picker: {exc}") from exc
        return {"path": selected}

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    def spa_fallback(full_path: str) -> str:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        return frontend_index_path().read_text(encoding="utf-8")

    return app
