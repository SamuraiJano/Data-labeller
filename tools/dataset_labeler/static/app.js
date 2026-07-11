"use strict";

const els = {};
const state = {
  project: null,
  images: [],
  currentIndex: 0,
  currentImage: null,
  annotations: [],
  classes: [],
  selectedClassId: 0,
  selectedAnnId: null,
  tool: "select",
  imageObj: null,
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  drawing: [],
  freehand: false,
  drag: null,
  history: [],
  redo: [],
  dirty: false,
  saveTimer: null,
  overlayOpacity: 0.35,
  showMasks: true,
};

function $(id) {
  return document.getElementById(id);
}

function initElements() {
  [
    "saveState", "projectPanel", "workspace", "sourceDir", "projectDir", "browseSourceBtn",
    "browseProjectBtn", "createProjectBtn", "openProjectBtn", "validateBtn", "splitBtn",
    "cocoBtn", "yoloBtn", "imageCounter", "imageStatus", "imageInfo", "prevBtn",
    "nextBtn", "imageList", "canvas", "fitBtn", "undoBtn", "redoBtn", "deleteBtn",
    "maskVisible", "opacity", "classSelect", "statusSelect", "reviewedCheck",
    "excludedCheck", "excludeReason", "instanceCount", "instanceList", "summaryText",
    "operationOutput"
  ].forEach((id) => { els[id] = $(id); });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.detail || response.statusText);
  }
  return data;
}

function setOutput(value) {
  els.operationOutput.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function setSaveState(text) {
  els.saveState.textContent = text;
}

function classById(id) {
  return state.classes.find((cls) => Number(cls.id) === Number(id)) || state.classes[0];
}

function pointsFromAnn(ann) {
  if (ann.points && ann.points.length) {
    return ann.points.map((p) => [Number(p[0]), Number(p[1])]);
  }
  const flat = ann.segmentation && ann.segmentation[0] ? ann.segmentation[0] : [];
  const points = [];
  for (let i = 0; i < flat.length; i += 2) {
    points.push([Number(flat[i]), Number(flat[i + 1])]);
  }
  return points;
}

function normalizeAnn(ann) {
  return {
    id: ann.id,
    category_id: Number(ann.category_id),
    points: pointsFromAnn(ann),
    source: ann.source || "manual",
    notes: ann.notes || "",
  };
}

function visibleAnnotations() {
  return state.annotations.map(normalizeAnn);
}

function imageUrl(imageId) {
  return `/api/image-file/${imageId}?t=${Date.now()}`;
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  els.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  els.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  draw();
}

function canvasSize() {
  const rect = els.canvas.getBoundingClientRect();
  return { width: rect.width, height: rect.height };
}

function toScreen(point) {
  return [point[0] * state.scale + state.offsetX, point[1] * state.scale + state.offsetY];
}

function toImage(point) {
  return [(point[0] - state.offsetX) / state.scale, (point[1] - state.offsetY) / state.scale];
}

function clampImagePoint(point) {
  if (!state.currentImage) return point;
  return [
    Math.max(0, Math.min(Number(state.currentImage.width), point[0])),
    Math.max(0, Math.min(Number(state.currentImage.height), point[1])),
  ];
}

function mousePoint(event) {
  const rect = els.canvas.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

function fitView() {
  if (!state.currentImage) return;
  const size = canvasSize();
  const iw = Number(state.currentImage.width);
  const ih = Number(state.currentImage.height);
  state.scale = Math.min(size.width / iw, size.height / ih) * 0.96;
  state.offsetX = (size.width - iw * state.scale) / 2;
  state.offsetY = (size.height - ih * state.scale) / 2;
  draw();
}

function draw() {
  const canvas = els.canvas;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const size = canvasSize();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size.width, size.height);
  ctx.fillStyle = "#0b0e12";
  ctx.fillRect(0, 0, size.width, size.height);

  if (!state.currentImage || !state.imageObj) return;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(
    state.imageObj,
    state.offsetX,
    state.offsetY,
    Number(state.currentImage.width) * state.scale,
    Number(state.currentImage.height) * state.scale
  );

  if (state.showMasks) {
    for (const ann of visibleAnnotations()) {
      drawPolygon(ctx, ann, Number(ann.id) === Number(state.selectedAnnId));
    }
  }
  if (state.drawing.length) {
    drawDraft(ctx, state.drawing);
  }
}

function drawPolygon(ctx, ann, selected) {
  const points = ann.points;
  if (points.length < 2) return;
  const cls = classById(ann.category_id);
  ctx.beginPath();
  points.forEach((point, index) => {
    const [x, y] = toScreen(point);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = hexToRgba(cls.color, state.overlayOpacity);
  ctx.strokeStyle = selected ? "#ffffff" : cls.color;
  ctx.lineWidth = selected ? 3 : 2;
  ctx.fill();
  ctx.stroke();

  if (selected) {
    for (const point of points) {
      const [x, y] = toScreen(point);
      ctx.fillStyle = "#10151b";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}

function drawDraft(ctx, points) {
  ctx.beginPath();
  points.forEach((point, index) => {
    const [x, y] = toScreen(point);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#d8e7ff";
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);
  ctx.stroke();
  ctx.setLineDash([]);
  for (const point of points) {
    const [x, y] = toScreen(point);
    ctx.fillStyle = "#d8e7ff";
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function hexToRgba(hex, alpha) {
  const raw = (hex || "#5aa7d8").replace("#", "");
  const r = parseInt(raw.slice(0, 2), 16);
  const g = parseInt(raw.slice(2, 4), 16);
  const b = parseInt(raw.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function setDirty(value) {
  state.dirty = value;
  setSaveState(value ? "Unsaved" : "Saved");
}

function snapshot() {
  return JSON.stringify({
    annotations: visibleAnnotations(),
    image: {
      status: state.currentImage?.status,
      reviewed: state.currentImage?.reviewed,
      excluded: state.currentImage?.excluded,
      exclusion_reason: state.currentImage?.exclusion_reason,
    },
  });
}

function restoreSnapshot(text) {
  const data = JSON.parse(text);
  state.annotations = data.annotations;
  if (state.currentImage && data.image) {
    Object.assign(state.currentImage, data.image);
  }
  refreshPanels();
  draw();
}

function pushHistory() {
  state.history.push(snapshot());
  if (state.history.length > 80) state.history.shift();
  state.redo = [];
}

function undo() {
  if (!state.history.length) return;
  state.redo.push(snapshot());
  restoreSnapshot(state.history.pop());
  saveCurrent();
}

function redo() {
  if (!state.redo.length) return;
  state.history.push(snapshot());
  restoreSnapshot(state.redo.pop());
  saveCurrent();
}

function scheduleSave() {
  setDirty(true);
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(saveCurrent, 250);
}

async function saveCurrent() {
  if (!state.project || !state.currentImage) return;
  clearTimeout(state.saveTimer);
  const payload = {
    image: {
      status: state.currentImage.status,
      reviewed: Boolean(state.currentImage.reviewed),
      excluded: Boolean(state.currentImage.excluded),
      exclusion_reason: state.currentImage.exclusion_reason || "",
      capture_group: state.currentImage.capture_group || "",
      tree_id: state.currentImage.tree_id || "",
      fruit_id: state.currentImage.fruit_id || "",
      session_id: state.currentImage.session_id || "",
    },
    annotations: visibleAnnotations().map((ann) => ({
      ...(ann.id && Number(ann.id) > 0 ? { id: ann.id } : {}),
      category_id: ann.category_id,
      points: ann.points,
      source: ann.source || "manual",
      notes: ann.notes || "",
    })),
  };
  try {
    const data = await api(`/api/images/${state.currentImage.id}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    state.currentImage = data.image;
    state.images[state.currentIndex] = data.image;
    state.annotations = data.annotations.map(normalizeAnn);
    updateSummary(data.summary);
    setDirty(false);
    refreshPanels();
    draw();
  } catch (error) {
    setSaveState(`Save failed: ${error.message}`);
  }
}

async function loadState() {
  const data = await api("/api/state");
  if (data.opened) {
    openLoadedProject(data.project, data.summary);
  } else {
    state.classes = data.classes;
    renderClasses();
  }
}

function openLoadedProject(project, summary) {
  state.project = project;
  state.classes = project.classes;
  state.images = project.images;
  const lastId = project.last_image_id;
  const index = lastId ? state.images.findIndex((img) => Number(img.id) === Number(lastId)) : 0;
  state.currentIndex = Math.max(0, index);
  renderClasses();
  els.projectPanel.classList.add("hidden");
  els.workspace.classList.remove("hidden");
  updateSummary(summary);
  loadCurrentImage();
}

async function createProject() {
  const sourceDir = els.sourceDir.value.trim();
  const projectDir = els.projectDir.value.trim();
  if (!sourceDir || !projectDir) {
    setOutput("Source and project folders are required.");
    return;
  }
  const data = await api("/api/project/create", {
    method: "POST",
    body: JSON.stringify({ source_dir: sourceDir, project_dir: projectDir }),
  });
  openLoadedProject(data.project, data.summary);
}

async function openProject() {
  const projectDir = els.projectDir.value.trim();
  if (!projectDir) {
    setOutput("Project folder is required.");
    return;
  }
  const data = await api("/api/project/open", {
    method: "POST",
    body: JSON.stringify({ project_dir: projectDir }),
  });
  openLoadedProject(data.project, data.summary);
}

async function browse(target, title) {
  const data = await api("/api/dialog/directory", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
  if (data.path) target.value = data.path;
}

async function loadCurrentImage() {
  if (!state.images.length) return;
  state.currentImage = state.images[state.currentIndex];
  const data = await api(`/api/images/${state.currentImage.id}`);
  state.currentImage = data.image;
  state.images[state.currentIndex] = data.image;
  state.annotations = data.annotations.map(normalizeAnn);
  state.selectedAnnId = null;
  state.drawing = [];
  state.history = [];
  state.redo = [];
  const img = new Image();
  img.onload = () => {
    state.imageObj = img;
    fitView();
    refreshPanels();
    setSaveState("Saved");
  };
  img.src = imageUrl(state.currentImage.id);
}

function renderClasses() {
  els.classSelect.innerHTML = "";
  for (const cls of state.classes) {
    const option = document.createElement("option");
    option.value = cls.id;
    option.textContent = `${cls.id}: ${cls.name}`;
    els.classSelect.appendChild(option);
  }
  els.classSelect.value = String(state.selectedClassId);
}

function updateSummary(summary) {
  if (!summary) return;
  els.summaryText.innerHTML = [
    `Images: ${summary.image_count}`,
    `Instances: ${summary.annotation_count}`,
    `Reviewed: ${summary.reviewed_count}`,
    `Excluded: ${summary.excluded_count}`,
    `Unlabelled: ${summary.unlabelled_count}`,
  ].join("<br>");
}

function refreshPanels() {
  if (!state.currentImage) return;
  const image = state.currentImage;
  els.imageCounter.textContent = `${state.currentIndex + 1} / ${state.images.length}`;
  els.imageStatus.textContent = image.status || "unlabelled";
  els.imageInfo.innerHTML = [
    `<strong>${escapeHtml(image.file_name)}</strong>`,
    `${escapeHtml(image.relative_path)}`,
    `${image.width} x ${image.height}`,
    `Folder: ${escapeHtml(image.source_folder || "unknown")}`,
    `Class: ${escapeHtml(image.source_class || "unknown")}`,
    `SHA-256: ${escapeHtml((image.sha256 || "").slice(0, 16))}`,
  ].join("<br>");
  els.statusSelect.value = image.status || "unlabelled";
  els.reviewedCheck.checked = Boolean(image.reviewed);
  els.excludedCheck.checked = Boolean(image.excluded);
  els.excludeReason.value = image.exclusion_reason || "";
  if (image.source_class_id !== null && image.source_class_id !== undefined && state.annotations.length === 0) {
    state.selectedClassId = Number(image.source_class_id);
    els.classSelect.value = String(state.selectedClassId);
  }
  renderImageList();
  renderInstanceList();
}

function renderImageList() {
  els.imageList.innerHTML = "";
  const start = Math.max(0, state.currentIndex - 80);
  const end = Math.min(state.images.length, state.currentIndex + 120);
  for (let i = start; i < end; i += 1) {
    const image = state.images[i];
    const row = document.createElement("div");
    row.className = `image-row ${i === state.currentIndex ? "active" : ""}`;
    row.innerHTML = `<span>${i + 1}. ${escapeHtml(image.file_name)}</span><small>${escapeHtml(image.status || "unlabelled")} | ${escapeHtml(image.source_class || "unknown")}</small>`;
    row.addEventListener("click", async () => {
      await saveCurrent();
      state.currentIndex = i;
      loadCurrentImage();
    });
    els.imageList.appendChild(row);
  }
}

function renderInstanceList() {
  els.instanceCount.textContent = String(state.annotations.length);
  els.instanceList.innerHTML = "";
  for (const ann of visibleAnnotations()) {
    const cls = classById(ann.category_id);
    const row = document.createElement("div");
    row.className = `instance-row ${Number(ann.id) === Number(state.selectedAnnId) ? "active" : ""}`;
    row.innerHTML = `<span>${escapeHtml(cls.name)}</span><small>${ann.points.length} vertices</small>`;
    row.addEventListener("click", () => {
      state.selectedAnnId = ann.id;
      draw();
      renderInstanceList();
    });
    els.instanceList.appendChild(row);
  }
}

function setTool(tool) {
  state.tool = tool;
  document.querySelectorAll(".tool").forEach((btn) => btn.classList.toggle("active", btn.dataset.tool === tool));
  els.canvas.style.cursor = tool === "pan" ? "grab" : "crosshair";
}

function finishPolygon(points, source) {
  const clean = points.map(clampImagePoint);
  if (new Set(clean.map((p) => `${p[0].toFixed(3)},${p[1].toFixed(3)}`)).size < 3) {
    state.drawing = [];
    draw();
    return;
  }
  pushHistory();
  state.annotations.push({
    id: `tmp_${Date.now()}`,
    category_id: state.selectedClassId,
    points: clean,
    source,
    notes: "",
  });
  state.selectedAnnId = state.annotations[state.annotations.length - 1].id;
  state.drawing = [];
  if (state.currentImage.status === "unlabelled") state.currentImage.status = "labelled";
  refreshPanels();
  draw();
  scheduleSave();
}

function hitVertex(screenPoint) {
  const threshold = 9;
  for (const ann of visibleAnnotations().slice().reverse()) {
    for (let i = 0; i < ann.points.length; i += 1) {
      const [x, y] = toScreen(ann.points[i]);
      if (Math.hypot(screenPoint[0] - x, screenPoint[1] - y) <= threshold) {
        return { ann, vertexIndex: i };
      }
    }
  }
  return null;
}

function distanceToSegment(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  if (dx === 0 && dy === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

function hitEdge(screenPoint) {
  const imagePoint = toImage(screenPoint);
  const threshold = 9 / state.scale;
  for (const ann of visibleAnnotations().slice().reverse()) {
    for (let i = 0; i < ann.points.length; i += 1) {
      const a = ann.points[i];
      const b = ann.points[(i + 1) % ann.points.length];
      if (distanceToSegment(imagePoint, a, b) <= threshold) {
        return { ann, insertAfter: i };
      }
    }
  }
  return null;
}

function hitPolygon(screenPoint) {
  const imagePoint = toImage(screenPoint);
  for (const ann of visibleAnnotations().slice().reverse()) {
    if (pointInPolygon(imagePoint, ann.points)) return ann;
  }
  return null;
}

function pointInPolygon(point, polygon) {
  let inside = false;
  let j = polygon.length - 1;
  for (let i = 0; i < polygon.length; i += 1) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];
    const cross = (yi > point[1]) !== (yj > point[1]);
    if (cross && point[0] < ((xj - xi) * (point[1] - yi)) / ((yj - yi) || 1e-9) + xi) inside = !inside;
    j = i;
  }
  return inside;
}

function replaceAnnotation(ann) {
  const idx = state.annotations.findIndex((item) => String(item.id) === String(ann.id));
  if (idx >= 0) state.annotations[idx] = ann;
}

function deleteSelected() {
  if (!state.selectedAnnId) return;
  pushHistory();
  state.annotations = state.annotations.filter((ann) => String(ann.id) !== String(state.selectedAnnId));
  state.selectedAnnId = null;
  refreshPanels();
  draw();
  scheduleSave();
}

function canvasMouseDown(event) {
  if (!state.currentImage) return;
  const screen = mousePoint(event);
  if (state.tool === "pan" || event.button === 1) {
    state.drag = { type: "pan", start: screen, ox: state.offsetX, oy: state.offsetY };
    return;
  }
  if (state.tool === "polygon") {
    if (event.button === 2) {
      state.drawing = [];
      draw();
      return;
    }
    state.drawing.push(clampImagePoint(toImage(screen)));
    draw();
    return;
  }
  if (state.tool === "freehand") {
    pushHistory();
    state.freehand = true;
    state.drawing = [clampImagePoint(toImage(screen))];
    return;
  }
  if (state.tool === "select") {
    const vertex = hitVertex(screen);
    if (vertex) {
      pushHistory();
      state.selectedAnnId = vertex.ann.id;
      state.drag = { type: "vertex", annId: vertex.ann.id, vertexIndex: vertex.vertexIndex };
      refreshPanels();
      draw();
      return;
    }
    const ann = hitPolygon(screen);
    if (ann) {
      pushHistory();
      state.selectedAnnId = ann.id;
      state.drag = { type: "polygon", annId: ann.id, last: toImage(screen) };
      refreshPanels();
      draw();
      return;
    }
    state.selectedAnnId = null;
    refreshPanels();
    draw();
  }
}

function canvasMouseMove(event) {
  const screen = mousePoint(event);
  if (state.drag?.type === "pan") {
    state.offsetX = state.drag.ox + screen[0] - state.drag.start[0];
    state.offsetY = state.drag.oy + screen[1] - state.drag.start[1];
    draw();
    return;
  }
  if (state.freehand && state.tool === "freehand") {
    const point = clampImagePoint(toImage(screen));
    const last = state.drawing[state.drawing.length - 1];
    if (!last || Math.hypot(point[0] - last[0], point[1] - last[1]) > 2 / state.scale) {
      state.drawing.push(point);
      draw();
    }
    return;
  }
  if (state.drag?.type === "vertex") {
    const ann = visibleAnnotations().find((item) => String(item.id) === String(state.drag.annId));
    if (!ann) return;
    ann.points[state.drag.vertexIndex] = clampImagePoint(toImage(screen));
    replaceAnnotation(ann);
    draw();
    return;
  }
  if (state.drag?.type === "polygon") {
    const ann = visibleAnnotations().find((item) => String(item.id) === String(state.drag.annId));
    if (!ann) return;
    const now = toImage(screen);
    const dx = now[0] - state.drag.last[0];
    const dy = now[1] - state.drag.last[1];
    ann.points = ann.points.map((point) => clampImagePoint([point[0] + dx, point[1] + dy]));
    state.drag.last = now;
    replaceAnnotation(ann);
    draw();
  }
}

function canvasMouseUp() {
  if (state.freehand) {
    const points = simplifyTrace(state.drawing, 1.5 / state.scale);
    state.freehand = false;
    finishPolygon(points, "freehand");
    return;
  }
  if (state.drag && state.drag.type !== "pan") {
    refreshPanels();
    scheduleSave();
  }
  state.drag = null;
}

function canvasDoubleClick(event) {
  const screen = mousePoint(event);
  if (state.tool === "polygon") {
    if (state.drawing.length >= 3) finishPolygon(state.drawing, "manual");
    return;
  }
  if (state.tool === "select") {
    const edge = hitEdge(screen);
    if (edge) {
      pushHistory();
      edge.ann.points.splice(edge.insertAfter + 1, 0, clampImagePoint(toImage(screen)));
      state.selectedAnnId = edge.ann.id;
      replaceAnnotation(edge.ann);
      refreshPanels();
      draw();
      scheduleSave();
    }
  }
}

function canvasContextMenu(event) {
  event.preventDefault();
  const screen = mousePoint(event);
  if (state.tool === "polygon") {
    state.drawing = [];
    draw();
    return;
  }
  if (state.tool === "select") {
    const vertex = hitVertex(screen);
    if (vertex && vertex.ann.points.length > 3) {
      pushHistory();
      vertex.ann.points.splice(vertex.vertexIndex, 1);
      state.selectedAnnId = vertex.ann.id;
      replaceAnnotation(vertex.ann);
      refreshPanels();
      draw();
      scheduleSave();
    }
  }
}

function simplifyTrace(points, minDistance) {
  if (points.length <= 3) return points;
  const out = [points[0]];
  for (let i = 1; i < points.length; i += 1) {
    const last = out[out.length - 1];
    if (Math.hypot(points[i][0] - last[0], points[i][1] - last[1]) >= minDistance) {
      out.push(points[i]);
    }
  }
  return out.length >= 3 ? out : points;
}

function zoom(event) {
  if (!state.currentImage) return;
  event.preventDefault();
  const screen = mousePoint(event);
  const before = toImage(screen);
  const factor = event.deltaY < 0 ? 1.12 : 0.89;
  state.scale = Math.max(0.02, Math.min(80, state.scale * factor));
  state.offsetX = screen[0] - before[0] * state.scale;
  state.offsetY = screen[1] - before[1] * state.scale;
  draw();
}

function updateImageStateFromControls() {
  if (!state.currentImage) return;
  pushHistory();
  state.currentImage.status = els.statusSelect.value;
  state.currentImage.reviewed = els.reviewedCheck.checked || els.statusSelect.value === "reviewed";
  state.currentImage.excluded = els.excludedCheck.checked || els.statusSelect.value === "excluded";
  state.currentImage.exclusion_reason = els.excludeReason.value;
  if (state.currentImage.excluded) state.currentImage.status = "excluded";
  if (state.currentImage.reviewed && !state.currentImage.excluded) state.currentImage.status = "reviewed";
  refreshPanels();
  scheduleSave();
}

function nextImage(delta) {
  if (!state.images.length) return;
  saveCurrent().then(() => {
    state.currentIndex = Math.max(0, Math.min(state.images.length - 1, state.currentIndex + delta));
    loadCurrentImage();
  });
}

async function runOperation(name, fn) {
  setOutput(`${name}...`);
  try {
    setOutput(await fn());
  } catch (error) {
    setOutput(`${name} failed: ${error.message}`);
  }
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[ch]));
}

function wireEvents() {
  els.browseSourceBtn.addEventListener("click", () => browse(els.sourceDir, "Select source image folder"));
  els.browseProjectBtn.addEventListener("click", () => browse(els.projectDir, "Select annotation project folder"));
  els.createProjectBtn.addEventListener("click", () => runOperation("Create project", createProject));
  els.openProjectBtn.addEventListener("click", () => runOperation("Open project", openProject));
  els.prevBtn.addEventListener("click", () => nextImage(-1));
  els.nextBtn.addEventListener("click", () => nextImage(1));
  els.fitBtn.addEventListener("click", fitView);
  els.undoBtn.addEventListener("click", undo);
  els.redoBtn.addEventListener("click", redo);
  els.deleteBtn.addEventListener("click", deleteSelected);
  els.classSelect.addEventListener("change", () => {
    state.selectedClassId = Number(els.classSelect.value);
    if (state.selectedAnnId) {
      const ann = visibleAnnotations().find((item) => String(item.id) === String(state.selectedAnnId));
      if (ann) {
        pushHistory();
        ann.category_id = state.selectedClassId;
        replaceAnnotation(ann);
        refreshPanels();
        draw();
        scheduleSave();
      }
    }
  });
  els.statusSelect.addEventListener("change", updateImageStateFromControls);
  els.reviewedCheck.addEventListener("change", updateImageStateFromControls);
  els.excludedCheck.addEventListener("change", updateImageStateFromControls);
  els.excludeReason.addEventListener("change", updateImageStateFromControls);
  els.maskVisible.addEventListener("change", () => {
    state.showMasks = els.maskVisible.checked;
    draw();
  });
  els.opacity.addEventListener("input", () => {
    state.overlayOpacity = Number(els.opacity.value);
    draw();
  });
  document.querySelectorAll(".tool").forEach((btn) => btn.addEventListener("click", () => setTool(btn.dataset.tool)));
  els.canvas.addEventListener("mousedown", canvasMouseDown);
  els.canvas.addEventListener("mousemove", canvasMouseMove);
  els.canvas.addEventListener("mouseup", canvasMouseUp);
  els.canvas.addEventListener("mouseleave", canvasMouseUp);
  els.canvas.addEventListener("dblclick", canvasDoubleClick);
  els.canvas.addEventListener("contextmenu", canvasContextMenu);
  els.canvas.addEventListener("wheel", zoom, { passive: false });
  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("beforeunload", (event) => {
    if (state.dirty) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) return;
    if (event.ctrlKey && event.key.toLowerCase() === "z") { event.preventDefault(); undo(); return; }
    if (event.ctrlKey && event.key.toLowerCase() === "y") { event.preventDefault(); redo(); return; }
    if (event.key === "Delete") { deleteSelected(); return; }
    if (event.key === "ArrowRight" || event.key.toLowerCase() === "d") { nextImage(1); return; }
    if (event.key === "ArrowLeft" || event.key.toLowerCase() === "a") { nextImage(-1); return; }
    if (event.key.toLowerCase() === "v") setTool("select");
    if (event.key.toLowerCase() === "p") setTool("polygon");
    if (event.key.toLowerCase() === "f") fitView();
    if (event.key.toLowerCase() === "h") setTool("pan");
    if (event.key === "Enter" && state.tool === "polygon" && state.drawing.length >= 3) finishPolygon(state.drawing, "manual");
    if (event.key === "Escape") { state.drawing = []; state.selectedAnnId = null; draw(); refreshPanels(); }
  });
  els.validateBtn.addEventListener("click", () => runOperation("Validate", () => api("/api/validate", { method: "POST", body: JSON.stringify({}) })));
  els.splitBtn.addEventListener("click", () => runOperation("Split", () => api("/api/split", { method: "POST", body: JSON.stringify({ seed: 42 }) })));
  els.cocoBtn.addEventListener("click", () => runOperation("COCO export", () => api("/api/export/coco", { method: "POST", body: JSON.stringify({}) })));
  els.yoloBtn.addEventListener("click", () => runOperation("YOLO export", () => api("/api/export/yolo", { method: "POST", body: JSON.stringify({ include_hsv: true }) })));
}

initElements();
wireEvents();
resizeCanvas();
setTool("select");
loadState().catch((error) => setOutput(error.message));
