import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Brush,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Download,
  Eye,
  EyeOff,
  FileJson,
  FolderOpen,
  Hand,
  ListChecks,
  Maximize,
  MousePointer2,
  PenTool,
  Redo2,
  ScissorsLineDashed,
  Trash2,
  Undo2
} from "lucide-react";
import "./styles.css";

const api = async (path, options = {}) => {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
};

const clone = (value) => JSON.parse(JSON.stringify(value));
const toolNames = { select: "Select", polygon: "Polygon", freehand: "Freehand", pan: "Pan" };

function pointsFromAnnotation(annotation) {
  if (annotation.points?.length) return annotation.points.map((point) => [Number(point[0]), Number(point[1])]);
  const flat = annotation.segmentation?.[0] || [];
  const points = [];
  for (let i = 0; i < flat.length; i += 2) points.push([Number(flat[i]), Number(flat[i + 1])]);
  return points;
}

function normalizeAnnotation(annotation) {
  return {
    id: annotation.id,
    category_id: Number(annotation.category_id),
    points: pointsFromAnnotation(annotation),
    source: annotation.source || "manual",
    notes: annotation.notes || ""
  };
}

function pointInPolygon(point, polygon) {
  let inside = false;
  let j = polygon.length - 1;
  for (let i = 0; i < polygon.length; i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const crosses = yi > point[1] !== yj > point[1];
    if (crosses && point[0] < ((xj - xi) * (point[1] - yi)) / ((yj - yi) || 1e-9) + xi) inside = !inside;
    j = i;
  }
  return inside;
}

function distanceToSegment(point, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  if (dx === 0 && dy === 0) return Math.hypot(point[0] - a[0], point[1] - a[1]);
  const t = Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy));
}

function hexToRgba(hex, alpha) {
  const raw = (hex || "#5aa7d8").replace("#", "");
  const r = parseInt(raw.slice(0, 2), 16);
  const g = parseInt(raw.slice(2, 4), 16);
  const b = parseInt(raw.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function simplifyTrace(points, minDistance) {
  if (points.length <= 3) return points;
  const out = [points[0]];
  for (let i = 1; i < points.length; i += 1) {
    const last = out[out.length - 1];
    if (Math.hypot(points[i][0] - last[0], points[i][1] - last[1]) >= minDistance) out.push(points[i]);
  }
  return out.length >= 3 ? out : points;
}

function IconButton({ title, active = false, onClick, children, disabled = false }) {
  return (
    <button className={active ? "icon-button active" : "icon-button"} title={title} aria-label={title} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

function ProjectLauncher({ onOpen, setOutput }) {
  const [sourceDir, setSourceDir] = useState("");
  const [projectDir, setProjectDir] = useState("dataset/annotations");

  const browse = async (setter, title) => {
    const data = await api("/api/dialog/directory", { method: "POST", body: JSON.stringify({ title }) });
    if (data.path) setter(data.path);
  };

  const createProject = async () => {
    if (!sourceDir.trim() || !projectDir.trim()) {
      setOutput("Source and project folders are required.");
      return;
    }
    const data = await api("/api/project/create", {
      method: "POST",
      body: JSON.stringify({ source_dir: sourceDir.trim(), project_dir: projectDir.trim() })
    });
    onOpen(data.project, data.summary);
  };

  const openProject = async () => {
    if (!projectDir.trim()) {
      setOutput("Project folder is required.");
      return;
    }
    const data = await api("/api/project/open", {
      method: "POST",
      body: JSON.stringify({ project_dir: projectDir.trim() })
    });
    onOpen(data.project, data.summary);
  };

  return (
    <section className="project-launcher">
      <div className="launcher-grid">
        <label>
          <span>Source image folder</span>
          <div className="path-row">
            <input value={sourceDir} onChange={(event) => setSourceDir(event.target.value)} placeholder="Folder containing citrus canker, healthy, melanose" />
            <button onClick={() => browse(setSourceDir, "Select source image folder")}>
              <FolderOpen size={17} /> Browse
            </button>
          </div>
        </label>
        <label>
          <span>Annotation project folder</span>
          <div className="path-row">
            <input value={projectDir} onChange={(event) => setProjectDir(event.target.value)} />
            <button onClick={() => browse(setProjectDir, "Select annotation project folder")}>
              <FolderOpen size={17} /> Browse
            </button>
          </div>
        </label>
        <div className="launcher-actions">
          <button onClick={createProject}><Check size={17} /> Create / Import</button>
          <button onClick={openProject}><FolderOpen size={17} /> Open</button>
        </div>
      </div>
    </section>
  );
}

function TopBar({ opened, dirty, onValidate, onSplit, onCoco, onYolo }) {
  return (
    <header className="topbar">
      <div className="brand">
        <strong>Skripsie Segmentation Labeler</strong>
        <span className={dirty ? "save-state dirty" : "save-state"}>{opened ? (dirty ? "Unsaved" : "Saved") : "No project"}</span>
      </div>
      <div className="top-actions">
        <button onClick={onValidate} disabled={!opened}><ListChecks size={17} /> Validate</button>
        <button onClick={onSplit} disabled={!opened}><ScissorsLineDashed size={17} /> Split</button>
        <button onClick={onCoco} disabled={!opened}><FileJson size={17} /> COCO</button>
        <button onClick={onYolo} disabled={!opened}><Download size={17} /> YOLO + HSV</button>
      </div>
    </header>
  );
}

function ImageSidebar({ images, currentIndex, currentImage, annotations, onSelect, onPrev, onNext }) {
  const start = Math.max(0, currentIndex - 80);
  const end = Math.min(images.length, currentIndex + 120);
  return (
    <aside className="sidebar left">
      <div className="panel-head">
        <span>{images.length ? `${currentIndex + 1} / ${images.length}` : "0 / 0"}</span>
        <span>{currentImage?.status || "unlabelled"}</span>
      </div>
      {currentImage && (
        <div className="image-info">
          <strong>{currentImage.file_name}</strong>
          <span>{currentImage.relative_path}</span>
          <span>{currentImage.width} x {currentImage.height}</span>
          <span>Folder: {currentImage.source_folder || "unknown"}</span>
          <span>Class: {currentImage.source_class || "unknown"}</span>
          <span>SHA-256: {(currentImage.sha256 || "").slice(0, 16)}</span>
          <span>Instances: {annotations.length}</span>
        </div>
      )}
      <div className="nav-row">
        <button onClick={onPrev} disabled={!images.length || currentIndex === 0}><ChevronLeft size={17} /> Prev</button>
        <button onClick={onNext} disabled={!images.length || currentIndex >= images.length - 1}>Next <ChevronRight size={17} /></button>
      </div>
      <div className="image-list">
        {images.slice(start, end).map((image, offset) => {
          const index = start + offset;
          return (
            <button key={image.id} className={index === currentIndex ? "image-row active" : "image-row"} onClick={() => onSelect(index)}>
              <span>{index + 1}. {image.file_name}</span>
              <small>{image.status || "unlabelled"} | {image.source_class || "unknown"}</small>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

function AnnotationCanvas({
  image,
  annotations,
  setAnnotations,
  classes,
  selectedClassId,
  selectedAnnotationId,
  setSelectedAnnotationId,
  onMeaningfulEdit,
  onImageStatus,
  tool,
  setTool,
  showMasks,
  setShowMasks,
  opacity,
  setOpacity,
  undo,
  redo,
  deleteSelected,
  pushHistory
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const viewRef = useRef({ scale: 1, offsetX: 0, offsetY: 0 });
  const dragRef = useRef(null);
  const drawingRef = useRef([]);
  const [, redrawTick] = useState(0);

  const classById = useCallback((id) => classes.find((cls) => Number(cls.id) === Number(id)) || classes[0], [classes]);
  const requestDraw = useCallback(() => redrawTick((value) => value + 1), []);

  const canvasSize = useCallback(() => {
    const rect = canvasRef.current?.getBoundingClientRect();
    return { width: rect?.width || 1, height: rect?.height || 1 };
  }, []);

  const toScreen = useCallback((point) => {
    const view = viewRef.current;
    return [point[0] * view.scale + view.offsetX, point[1] * view.scale + view.offsetY];
  }, []);

  const toImage = useCallback((point) => {
    const view = viewRef.current;
    return [(point[0] - view.offsetX) / view.scale, (point[1] - view.offsetY) / view.scale];
  }, []);

  const clampImagePoint = useCallback((point) => {
    if (!image) return point;
    return [
      Math.max(0, Math.min(Number(image.width), point[0])),
      Math.max(0, Math.min(Number(image.height), point[1]))
    ];
  }, [image]);

  const mousePoint = useCallback((event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }, []);

  const fitView = useCallback(() => {
    if (!image) return;
    const size = canvasSize();
    const scale = Math.min(size.width / Number(image.width), size.height / Number(image.height)) * 0.96;
    viewRef.current = {
      scale,
      offsetX: (size.width - Number(image.width) * scale) / 2,
      offsetY: (size.height - Number(image.height) * scale) / 2
    };
    requestDraw();
  }, [canvasSize, image, requestDraw]);

  const hitVertex = useCallback((screenPoint) => {
    const threshold = 9;
    for (const ann of annotations.slice().reverse()) {
      for (let i = 0; i < ann.points.length; i += 1) {
        const [x, y] = toScreen(ann.points[i]);
        if (Math.hypot(screenPoint[0] - x, screenPoint[1] - y) <= threshold) return { annotation: ann, vertexIndex: i };
      }
    }
    return null;
  }, [annotations, toScreen]);

  const hitEdge = useCallback((screenPoint) => {
    const imagePoint = toImage(screenPoint);
    const threshold = 9 / viewRef.current.scale;
    for (const ann of annotations.slice().reverse()) {
      for (let i = 0; i < ann.points.length; i += 1) {
        if (distanceToSegment(imagePoint, ann.points[i], ann.points[(i + 1) % ann.points.length]) <= threshold) {
          return { annotation: ann, insertAfter: i };
        }
      }
    }
    return null;
  }, [annotations, toImage]);

  const hitPolygon = useCallback((screenPoint) => {
    const imagePoint = toImage(screenPoint);
    return annotations.slice().reverse().find((ann) => pointInPolygon(imagePoint, ann.points)) || null;
  }, [annotations, toImage]);

  const finishPolygon = useCallback((points, source) => {
    const clean = points.map(clampImagePoint);
    const distinct = new Set(clean.map((point) => `${point[0].toFixed(3)},${point[1].toFixed(3)}`));
    drawingRef.current = [];
    if (distinct.size < 3) {
      requestDraw();
      return;
    }
    pushHistory();
    const annotation = {
      id: `tmp_${Date.now()}`,
      category_id: selectedClassId,
      points: clean,
      source,
      notes: ""
    };
    setAnnotations((current) => [...current, annotation]);
    setSelectedAnnotationId(annotation.id);
    onImageStatus({ status: "labelled" }, false);
    onMeaningfulEdit();
    requestDraw();
  }, [clampImagePoint, onImageStatus, onMeaningfulEdit, pushHistory, requestDraw, selectedClassId, setAnnotations, setSelectedAnnotationId]);

  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    requestDraw();
  }, [requestDraw]);

  useEffect(() => {
    const img = new Image();
    imageRef.current = null;
    if (!image) {
      requestDraw();
      return undefined;
    }
    img.onload = () => {
      imageRef.current = img;
      fitView();
    };
    img.src = `/api/image-file/${image.id}?t=${Date.now()}`;
    return () => {
      imageRef.current = null;
    };
  }, [fitView, image, requestDraw]);

  useEffect(() => {
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    return () => window.removeEventListener("resize", resizeCanvas);
  }, [resizeCanvas]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const size = canvasSize();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.fillStyle = "#0b0e12";
    ctx.fillRect(0, 0, size.width, size.height);

    if (!image || !imageRef.current) return;
    const view = viewRef.current;
    ctx.drawImage(imageRef.current, view.offsetX, view.offsetY, Number(image.width) * view.scale, Number(image.height) * view.scale);

    if (showMasks) {
      for (const ann of annotations) {
        if (ann.points.length < 2) continue;
        const cls = classById(ann.category_id);
        ctx.beginPath();
        ann.points.forEach((point, index) => {
          const [x, y] = toScreen(point);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        const selected = String(ann.id) === String(selectedAnnotationId);
        ctx.fillStyle = hexToRgba(cls?.color, opacity);
        ctx.strokeStyle = selected ? "#ffffff" : cls?.color || "#5aa7d8";
        ctx.lineWidth = selected ? 3 : 2;
        ctx.fill();
        ctx.stroke();
        if (selected) {
          for (const point of ann.points) {
            const [x, y] = toScreen(point);
            ctx.beginPath();
            ctx.fillStyle = "#10151b";
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2;
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          }
        }
      }
    }

    if (drawingRef.current.length) {
      ctx.beginPath();
      drawingRef.current.forEach((point, index) => {
        const [x, y] = toScreen(point);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = "#d8e7ff";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
      for (const point of drawingRef.current) {
        const [x, y] = toScreen(point);
        ctx.beginPath();
        ctx.fillStyle = "#d8e7ff";
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  });

  const onMouseDown = (event) => {
    if (!image) return;
    const screen = mousePoint(event);
    if (tool === "pan" || event.button === 1) {
      dragRef.current = { type: "pan", start: screen, ox: viewRef.current.offsetX, oy: viewRef.current.offsetY };
      return;
    }
    if (tool === "polygon") {
      if (event.button === 2) {
        drawingRef.current = [];
        requestDraw();
        return;
      }
      drawingRef.current = [...drawingRef.current, clampImagePoint(toImage(screen))];
      requestDraw();
      return;
    }
    if (tool === "freehand") {
      pushHistory();
      dragRef.current = { type: "freehand" };
      drawingRef.current = [clampImagePoint(toImage(screen))];
      requestDraw();
      return;
    }
    if (tool === "select") {
      const vertex = hitVertex(screen);
      if (vertex) {
        pushHistory();
        setSelectedAnnotationId(vertex.annotation.id);
        dragRef.current = { type: "vertex", annotationId: vertex.annotation.id, vertexIndex: vertex.vertexIndex };
        requestDraw();
        return;
      }
      const annotation = hitPolygon(screen);
      if (annotation) {
        pushHistory();
        setSelectedAnnotationId(annotation.id);
        dragRef.current = { type: "polygon", annotationId: annotation.id, last: toImage(screen) };
        requestDraw();
        return;
      }
      setSelectedAnnotationId(null);
      requestDraw();
    }
  };

  const onMouseMove = (event) => {
    const drag = dragRef.current;
    if (!drag) return;
    const screen = mousePoint(event);
    if (drag.type === "pan") {
      viewRef.current = { ...viewRef.current, offsetX: drag.ox + screen[0] - drag.start[0], offsetY: drag.oy + screen[1] - drag.start[1] };
      requestDraw();
      return;
    }
    if (drag.type === "freehand") {
      const point = clampImagePoint(toImage(screen));
      const last = drawingRef.current[drawingRef.current.length - 1];
      if (!last || Math.hypot(point[0] - last[0], point[1] - last[1]) > 2 / viewRef.current.scale) {
        drawingRef.current = [...drawingRef.current, point];
        requestDraw();
      }
      return;
    }
    if (drag.type === "vertex") {
      setAnnotations((current) => current.map((ann) => {
        if (String(ann.id) !== String(drag.annotationId)) return ann;
        const next = clone(ann);
        next.points[drag.vertexIndex] = clampImagePoint(toImage(screen));
        return next;
      }));
      requestDraw();
      return;
    }
    if (drag.type === "polygon") {
      const now = toImage(screen);
      const dx = now[0] - drag.last[0];
      const dy = now[1] - drag.last[1];
      drag.last = now;
      setAnnotations((current) => current.map((ann) => {
        if (String(ann.id) !== String(drag.annotationId)) return ann;
        return { ...ann, points: ann.points.map((point) => clampImagePoint([point[0] + dx, point[1] + dy])) };
      }));
      requestDraw();
    }
  };

  const onMouseUp = () => {
    const drag = dragRef.current;
    if (!drag) return;
    if (drag.type === "freehand") {
      finishPolygon(simplifyTrace(drawingRef.current, 1.5 / viewRef.current.scale), "freehand");
      dragRef.current = null;
      return;
    }
    if (drag.type !== "pan") onMeaningfulEdit();
    dragRef.current = null;
  };

  const onDoubleClick = (event) => {
    const screen = mousePoint(event);
    if (tool === "polygon") {
      if (drawingRef.current.length >= 3) finishPolygon(drawingRef.current, "manual");
      return;
    }
    if (tool !== "select") return;
    const edge = hitEdge(screen);
    if (!edge) return;
    pushHistory();
    const point = clampImagePoint(toImage(screen));
    setAnnotations((current) => current.map((ann) => {
      if (String(ann.id) !== String(edge.annotation.id)) return ann;
      const next = clone(ann);
      next.points.splice(edge.insertAfter + 1, 0, point);
      return next;
    }));
    setSelectedAnnotationId(edge.annotation.id);
    onMeaningfulEdit();
  };

  const onContextMenu = (event) => {
    event.preventDefault();
    const screen = mousePoint(event);
    if (tool === "polygon") {
      drawingRef.current = [];
      requestDraw();
      return;
    }
    if (tool !== "select") return;
    const vertex = hitVertex(screen);
    if (!vertex || vertex.annotation.points.length <= 3) return;
    pushHistory();
    setAnnotations((current) => current.map((ann) => {
      if (String(ann.id) !== String(vertex.annotation.id)) return ann;
      const next = clone(ann);
      next.points.splice(vertex.vertexIndex, 1);
      return next;
    }));
    setSelectedAnnotationId(vertex.annotation.id);
    onMeaningfulEdit();
  };

  const onWheel = (event) => {
    if (!image) return;
    event.preventDefault();
    const screen = mousePoint(event);
    const before = toImage(screen);
    const factor = event.deltaY < 0 ? 1.12 : 0.89;
    const scale = Math.max(0.02, Math.min(80, viewRef.current.scale * factor));
    viewRef.current = {
      scale,
      offsetX: screen[0] - before[0] * scale,
      offsetY: screen[1] - before[1] * scale
    };
    requestDraw();
  };

  useEffect(() => {
    const onKey = (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) return;
      if (event.ctrlKey && event.key.toLowerCase() === "z") {
        event.preventDefault();
        undo();
      } else if (event.ctrlKey && event.key.toLowerCase() === "y") {
        event.preventDefault();
        redo();
      } else if (event.key === "Delete") {
        deleteSelected();
      } else if (event.key.toLowerCase() === "v") {
        setTool("select");
      } else if (event.key.toLowerCase() === "p") {
        setTool("polygon");
      } else if (event.key.toLowerCase() === "h") {
        setTool("pan");
      } else if (event.key === "Enter" && tool === "polygon" && drawingRef.current.length >= 3) {
        finishPolygon(drawingRef.current, "manual");
      } else if (event.key === "Escape") {
        drawingRef.current = [];
        setSelectedAnnotationId(null);
        requestDraw();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [deleteSelected, finishPolygon, redo, requestDraw, setSelectedAnnotationId, setTool, tool, undo]);

  return (
    <section className="canvas-shell">
      <div className="toolbar">
        <IconButton title="Select" active={tool === "select"} onClick={() => setTool("select")}><MousePointer2 size={18} /></IconButton>
        <IconButton title="Polygon" active={tool === "polygon"} onClick={() => setTool("polygon")}><PenTool size={18} /></IconButton>
        <IconButton title="Freehand" active={tool === "freehand"} onClick={() => setTool("freehand")}><Brush size={18} /></IconButton>
        <IconButton title="Pan" active={tool === "pan"} onClick={() => setTool("pan")}><Hand size={18} /></IconButton>
        <IconButton title="Fit" onClick={fitView}><Maximize size={18} /></IconButton>
        <IconButton title="Undo" onClick={undo}><Undo2 size={18} /></IconButton>
        <IconButton title="Redo" onClick={redo}><Redo2 size={18} /></IconButton>
        <IconButton title="Delete" onClick={deleteSelected}><Trash2 size={18} /></IconButton>
        <IconButton title={showMasks ? "Hide masks" : "Show masks"} active={showMasks} onClick={() => setShowMasks((value) => !value)}>
          {showMasks ? <Eye size={18} /> : <EyeOff size={18} />}
        </IconButton>
        <label className="range-control">
          <span>Opacity</span>
          <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} />
        </label>
        <span className="tool-state">{toolNames[tool]}</span>
      </div>
      <canvas
        ref={canvasRef}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onDoubleClick={onDoubleClick}
        onContextMenu={onContextMenu}
        onWheel={onWheel}
      />
    </section>
  );
}

function Inspector({
  classes,
  selectedClassId,
  setSelectedClassId,
  currentImage,
  updateImage,
  annotations,
  selectedAnnotationId,
  setSelectedAnnotationId,
  updateSelectedClass,
  summary,
  output
}) {
  const selected = annotations.find((ann) => String(ann.id) === String(selectedAnnotationId));
  return (
    <aside className="sidebar right">
      <label className="field">
        <span>Class</span>
        <select value={selected?.category_id ?? selectedClassId} onChange={(event) => {
          const value = Number(event.target.value);
          setSelectedClassId(value);
          if (selected) updateSelectedClass(value);
        }}>
          {classes.map((cls) => <option key={cls.id} value={cls.id}>{cls.id}: {cls.name}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Status</span>
        <select value={currentImage?.status || "unlabelled"} onChange={(event) => updateImage({ status: event.target.value })} disabled={!currentImage}>
          <option value="unlabelled">unlabelled</option>
          <option value="labelled">labelled</option>
          <option value="reviewed">reviewed</option>
          <option value="excluded">excluded</option>
        </select>
      </label>
      <label className="check-row">
        <input type="checkbox" checked={Boolean(currentImage?.reviewed)} onChange={(event) => updateImage({ reviewed: event.target.checked })} disabled={!currentImage} />
        Reviewed
      </label>
      <label className="check-row">
        <input type="checkbox" checked={Boolean(currentImage?.excluded)} onChange={(event) => updateImage({ excluded: event.target.checked })} disabled={!currentImage} />
        Excluded
      </label>
      <textarea
        value={currentImage?.exclusion_reason || ""}
        onChange={(event) => updateImage({ exclusion_reason: event.target.value }, true)}
        placeholder="Exclusion reason"
        disabled={!currentImage}
      />
      <div className="panel-head instance-head">
        <span>Instances</span>
        <span>{annotations.length}</span>
      </div>
      <div className="instance-list">
        {annotations.map((annotation) => {
          const cls = classes.find((item) => Number(item.id) === Number(annotation.category_id));
          return (
            <button
              key={annotation.id}
              className={String(annotation.id) === String(selectedAnnotationId) ? "instance-row active" : "instance-row"}
              onClick={() => setSelectedAnnotationId(annotation.id)}
            >
              <span>{cls?.name || `class_${annotation.category_id}`}</span>
              <small>{annotation.points.length} vertices</small>
            </button>
          );
        })}
      </div>
      <div className="report-box">
        {summary && (
          <div className="summary-text">
            <span>Images: {summary.image_count}</span>
            <span>Instances: {summary.annotation_count}</span>
            <span>Reviewed: {summary.reviewed_count}</span>
            <span>Excluded: {summary.excluded_count}</span>
            <span>Unlabelled: {summary.unlabelled_count}</span>
          </div>
        )}
        <pre>{output}</pre>
      </div>
    </aside>
  );
}

function App() {
  const [project, setProject] = useState(null);
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentImage, setCurrentImage] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [classes, setClasses] = useState([]);
  const [selectedClassId, setSelectedClassId] = useState(0);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState(null);
  const [tool, setTool] = useState("select");
  const [summary, setSummary] = useState(null);
  const [output, setOutput] = useState("");
  const [dirty, setDirty] = useState(false);
  const [showMasks, setShowMasks] = useState(true);
  const [opacity, setOpacity] = useState(0.35);
  const [history, setHistory] = useState([]);
  const [redoStack, setRedoStack] = useState([]);
  const saveTimerRef = useRef(null);
  const projectRef = useRef(null);
  const currentImageRef = useRef(null);
  const annotationsRef = useRef([]);
  const currentIndexRef = useRef(0);

  useEffect(() => {
    projectRef.current = project;
    currentImageRef.current = currentImage;
    annotationsRef.current = annotations;
    currentIndexRef.current = currentIndex;
  }, [annotations, currentImage, currentIndex, project]);

  const openLoadedProject = useCallback((nextProject, nextSummary) => {
    setProject(nextProject);
    setClasses(nextProject.classes || []);
    setImages(nextProject.images || []);
    const lastId = nextProject.last_image_id;
    const index = lastId ? Math.max(0, (nextProject.images || []).findIndex((image) => Number(image.id) === Number(lastId))) : 0;
    setCurrentIndex(index >= 0 ? index : 0);
    setSummary(nextSummary);
  }, []);

  const loadCurrentImage = useCallback(async () => {
    if (!images.length) {
      setCurrentImage(null);
      setAnnotations([]);
      return;
    }
    const image = images[currentIndex];
    const data = await api(`/api/images/${image.id}`);
    setCurrentImage(data.image);
    setImages((current) => current.map((item, index) => (index === currentIndex ? data.image : item)));
    setAnnotations((data.annotations || []).map(normalizeAnnotation));
    setSelectedAnnotationId(null);
    setHistory([]);
    setRedoStack([]);
    setDirty(false);
    if (data.image.source_class_id !== null && data.image.source_class_id !== undefined) setSelectedClassId(Number(data.image.source_class_id));
  }, [currentIndex, images]);

  useEffect(() => {
    api("/api/state")
      .then((data) => {
        setClasses(data.classes || []);
        if (data.opened) openLoadedProject(data.project, data.summary);
      })
      .catch((error) => setOutput(error.message));
  }, [openLoadedProject]);

  useEffect(() => {
    if (project && images.length) loadCurrentImage().catch((error) => setOutput(error.message));
  }, [currentIndex, project]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const beforeUnload = (event) => {
      if (!dirty) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", beforeUnload);
    return () => window.removeEventListener("beforeunload", beforeUnload);
  }, [dirty]);

  const snapshot = useCallback(() => JSON.stringify({
    annotations,
    image: currentImage ? {
      status: currentImage.status,
      reviewed: currentImage.reviewed,
      excluded: currentImage.excluded,
      exclusion_reason: currentImage.exclusion_reason
    } : null
  }), [annotations, currentImage]);

  const restoreSnapshot = useCallback((text) => {
    const data = JSON.parse(text);
    setAnnotations(data.annotations || []);
    if (data.image) setCurrentImage((current) => ({ ...current, ...data.image }));
    setDirty(true);
  }, []);

  const pushHistory = useCallback(() => {
    setHistory((current) => [...current.slice(-79), snapshot()]);
    setRedoStack([]);
  }, [snapshot]);

  const saveCurrent = useCallback(async () => {
    const activeProject = projectRef.current;
    const activeImage = currentImageRef.current;
    const activeAnnotations = annotationsRef.current;
    const activeIndex = currentIndexRef.current;
    if (!activeProject || !activeImage) return;
    clearTimeout(saveTimerRef.current);
    const payload = {
      image: {
        status: activeImage.status,
        reviewed: Boolean(activeImage.reviewed),
        excluded: Boolean(activeImage.excluded),
        exclusion_reason: activeImage.exclusion_reason || "",
        capture_group: activeImage.capture_group || "",
        tree_id: activeImage.tree_id || "",
        fruit_id: activeImage.fruit_id || "",
        session_id: activeImage.session_id || ""
      },
      annotations: activeAnnotations.map((annotation) => ({
        ...(Number.isInteger(Number(annotation.id)) && Number(annotation.id) > 0 ? { id: annotation.id } : {}),
        category_id: annotation.category_id,
        points: annotation.points,
        source: annotation.source || "manual",
        notes: annotation.notes || ""
      }))
    };
    const data = await api(`/api/images/${activeImage.id}`, { method: "PUT", body: JSON.stringify(payload) });
    setCurrentImage(data.image);
    setImages((current) => current.map((image, index) => (index === activeIndex ? data.image : image)));
    setAnnotations((data.annotations || []).map(normalizeAnnotation));
    setSummary(data.summary);
    setDirty(false);
  }, []);

  const scheduleSave = useCallback(() => {
    setDirty(true);
    clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      saveCurrent().catch((error) => setOutput(`Save failed: ${error.message}`));
    }, 300);
  }, [saveCurrent]);

  const onMeaningfulEdit = useCallback(() => scheduleSave(), [scheduleSave]);

  const updateImage = useCallback((patch, quiet = false) => {
    if (!currentImage) return;
    if (!quiet) pushHistory();
    setCurrentImage((current) => {
      const next = { ...current, ...patch };
      if (patch.excluded === true || next.status === "excluded") {
        next.excluded = true;
        next.status = "excluded";
      } else if (patch.excluded === false && current.status === "excluded") {
        next.excluded = false;
        next.status = annotations.length ? "labelled" : "unlabelled";
      }
      if (patch.reviewed === true || next.status === "reviewed") {
        next.reviewed = true;
        next.status = "reviewed";
      } else if (patch.reviewed === false && current.status === "reviewed") {
        next.reviewed = false;
        next.status = annotations.length ? "labelled" : "unlabelled";
      }
      return next;
    });
    scheduleSave();
  }, [annotations.length, currentImage, pushHistory, scheduleSave]);

  const onImageStatus = useCallback((patch, quiet = false) => {
    setCurrentImage((current) => current ? { ...current, ...patch } : current);
    if (!quiet) scheduleSave();
  }, [scheduleSave]);

  const updateSelectedClass = useCallback((classId) => {
    if (!selectedAnnotationId) return;
    pushHistory();
    setAnnotations((current) => current.map((annotation) => (
      String(annotation.id) === String(selectedAnnotationId) ? { ...annotation, category_id: classId } : annotation
    )));
    scheduleSave();
  }, [pushHistory, scheduleSave, selectedAnnotationId]);

  const deleteSelected = useCallback(() => {
    if (!selectedAnnotationId) return;
    pushHistory();
    setAnnotations((current) => current.filter((annotation) => String(annotation.id) !== String(selectedAnnotationId)));
    setSelectedAnnotationId(null);
    scheduleSave();
  }, [pushHistory, scheduleSave, selectedAnnotationId]);

  const undo = useCallback(() => {
    if (!history.length) return;
    setRedoStack((current) => [...current, snapshot()]);
    const previous = history[history.length - 1];
    setHistory((current) => current.slice(0, -1));
    restoreSnapshot(previous);
    scheduleSave();
  }, [history, restoreSnapshot, scheduleSave, snapshot]);

  const redo = useCallback(() => {
    if (!redoStack.length) return;
    setHistory((current) => [...current, snapshot()]);
    const next = redoStack[redoStack.length - 1];
    setRedoStack((current) => current.slice(0, -1));
    restoreSnapshot(next);
    scheduleSave();
  }, [redoStack, restoreSnapshot, scheduleSave, snapshot]);

  const changeImage = useCallback(async (index) => {
    await saveCurrent();
    setCurrentIndex(Math.max(0, Math.min(images.length - 1, index)));
  }, [images.length, saveCurrent]);

  const runOperation = useCallback(async (name, fn) => {
    setOutput(`${name}...`);
    try {
      setOutput(JSON.stringify(await fn(), null, 2));
    } catch (error) {
      setOutput(`${name} failed: ${error.message}`);
    }
  }, []);

  const opened = Boolean(project);
  const workspace = useMemo(() => opened && currentImage, [currentImage, opened]);

  return (
    <>
      <TopBar
        opened={opened}
        dirty={dirty}
        onValidate={() => runOperation("Validate", () => api("/api/validate", { method: "POST", body: JSON.stringify({}) }))}
        onSplit={() => runOperation("Split", () => api("/api/split", { method: "POST", body: JSON.stringify({ seed: 42 }) }))}
        onCoco={() => runOperation("COCO export", () => api("/api/export/coco", { method: "POST", body: JSON.stringify({}) }))}
        onYolo={() => runOperation("YOLO export", () => api("/api/export/yolo", { method: "POST", body: JSON.stringify({ include_hsv: true }) }))}
      />
      {!opened && <ProjectLauncher onOpen={openLoadedProject} setOutput={setOutput} />}
      {workspace && (
        <main className="workspace">
          <ImageSidebar
            images={images}
            currentIndex={currentIndex}
            currentImage={currentImage}
            annotations={annotations}
            onSelect={changeImage}
            onPrev={() => changeImage(currentIndex - 1)}
            onNext={() => changeImage(currentIndex + 1)}
          />
          <AnnotationCanvas
            image={currentImage}
            annotations={annotations}
            setAnnotations={setAnnotations}
            classes={classes}
            selectedClassId={selectedClassId}
            selectedAnnotationId={selectedAnnotationId}
            setSelectedAnnotationId={setSelectedAnnotationId}
            onMeaningfulEdit={onMeaningfulEdit}
            onImageStatus={onImageStatus}
            tool={tool}
            setTool={setTool}
            showMasks={showMasks}
            setShowMasks={setShowMasks}
            opacity={opacity}
            setOpacity={setOpacity}
            undo={undo}
            redo={redo}
            deleteSelected={deleteSelected}
            pushHistory={pushHistory}
          />
          <Inspector
            classes={classes}
            selectedClassId={selectedClassId}
            setSelectedClassId={setSelectedClassId}
            currentImage={currentImage}
            updateImage={updateImage}
            annotations={annotations}
            selectedAnnotationId={selectedAnnotationId}
            setSelectedAnnotationId={setSelectedAnnotationId}
            updateSelectedClass={updateSelectedClass}
            summary={summary}
            output={output}
          />
        </main>
      )}
      {opened && !currentImage && (
        <section className="empty-state">
          <CircleAlert size={24} />
          <span>No images are loaded in this project.</span>
        </section>
      )}
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
