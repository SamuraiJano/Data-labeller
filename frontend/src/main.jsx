import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BoxSelect,
  Brush,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  CircleHelp,
  Download,
  Eraser,
  Eye,
  EyeOff,
  FlipHorizontal2,
  FlipVertical2,
  FolderOpen,
  Hand,
  ListChecks,
  LoaderCircle,
  Maximize,
  MousePointer2,
  Plus,
  Pencil,
  Redo2,
  RotateCcw,
  RotateCw,
  Save,
  ScanSearch,
  ScissorsLineDashed,
  Sparkles,
  Tag,
  Trash2,
  Undo2,
  ZoomIn,
  ZoomOut,
  X
} from "lucide-react";
import { boxContainsPoint, boxHandlePoints, clampImagePoint, fitImageTransform, hitBoxHandle, imageToScreen, interpolateStroke, moveBox, resizeBox, screenToImage, shouldIgnoreShortcut, zoomTransformAtPoint } from "./canvasMath.js";
import { AUTOSAVE_IDLE_MS, initialImageIndex, patchImageCollection, patchProjectImage, reconcileSaveResponse } from "./autosave.js";
import { formatApiError } from "./apiErrors.js";
import { DEFAULT_DETECTION_PROMPT, detectionPromptMetadata, parseDetectionLabels } from "./detectionAssist.js";
import { SYNTHETIC_LIMITS, validateSyntheticSettings } from "./syntheticSettings.js";
import { retainExistingSelections, updateImageSelection } from "./bulkSelection.js";
import { HARVEST_STEP_MS, activeHarvestTracks as selectActiveHarvestTracks, containedFrameRect, harvestModeAfterLoad, openHarvestTracks as selectOpenHarvestTracks, snapHarvestTime, visibleHarvestMasks as selectVisibleHarvestMasks } from "./harvestState.js";
import { forgetRememberedHarvestSession, readRememberedHarvestSession, rememberHarvestSession, shouldResumeHarvestOnLaunch } from "./harvestSessionMemory.js";
import "./styles.css";

const api = async (path, options = {}) => {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const text = await response.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; }
  catch { data = text ? { detail: text } : {}; }
  if (!response.ok) throw new Error(formatApiError(data, response.statusText));
  return data;
};

const clone = (value) => JSON.parse(JSON.stringify(value));
const blankPrompts = () => ({ points: [], box: null, text: "" });
const DEFAULT_SEGMENTATION_CLASSES = "Skin wrinkling, Dark marking, Dark scarring, Wind scarring, Light-brown discolouration, Rough skin, Open skin";
const isFruit = (instance) => instance?.annotation_kind === "fruit_box";
const classLabel = (classes, trainingId) => classes?.find((item) => Number(item.export_index) === Number(trainingId))?.display_name || `Class ${Number(trainingId) + 1}`;
const BOX_RESIZE_CURSORS = { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize" };

async function maskCanvasFromSource(source, width, height) {
  const blob = await fetch(source).then((response) => {
    if (!response.ok) throw new Error(`Could not load mask (${response.status})`);
    return response.blob();
  });
  const bitmap = await createImageBitmap(blob);
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  sourceContext.drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();
  const pixels = sourceContext.getImageData(0, 0, width, height);
  let hasTransparency = false;
  for (let index = 3; index < pixels.data.length; index += 4) {
    if (pixels.data[index] < 255) { hasTransparency = true; break; }
  }
  for (let index = 0; index < pixels.data.length; index += 4) {
    const alpha = hasTransparency
      ? pixels.data[index + 3]
      : Math.max(pixels.data[index], pixels.data[index + 1], pixels.data[index + 2]);
    pixels.data[index] = 255;
    pixels.data[index + 1] = 255;
    pixels.data[index + 2] = 255;
    pixels.data[index + 3] = alpha > 0 ? 255 : 0;
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").putImageData(pixels, 0, 0);
  return canvas;
}

function blankMaskCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function cloneMaskCanvas(canvas) {
  const copy = blankMaskCanvas(canvas.width, canvas.height);
  copy.getContext("2d").drawImage(canvas, 0, 0);
  return copy;
}

function maskCanvasHasPixels(canvas) {
  if (!canvas) return false;
  const pixels = canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, canvas.width, canvas.height).data;
  for (let index = 3; index < pixels.length; index += 4) {
    if (pixels[index] > 0) return true;
  }
  return false;
}

function IconButton({ title, active = false, onClick, children, disabled = false }) {
  return (
    <button className={active ? "icon-button active" : "icon-button"} title={title} aria-label={title} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

function Modal({ title, onClose, children, wide = false }) {
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className={wide ? "modal wide" : "modal"} role="dialog" aria-modal="true" aria-label={title}>
        <header><h2>{title}</h2><IconButton title="Close" onClick={onClose}><X size={18} /></IconButton></header>
        {children}
      </section>
    </div>
  );
}

function HomeLauncher({ onOpen, setMessage, onAdvanced, onHarvest }) {
  const [view, setView] = useState("home");
  const [videoPaths, setVideoPaths] = useState([]);
  const [inspection, setInspection] = useState(null);
  const [outputParent, setOutputParent] = useState("dataset");
  const [projectName, setProjectName] = useState("Video dataset");
  const [annotationMode, setAnnotationMode] = useState("detection");
  const [classNames, setClassNames] = useState("Object");
  const [projectDir, setProjectDir] = useState("dataset/annotations");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(null);
  const jobIdRef = useRef(null);

  const browseDirectory = async (setter, title) => {
    try {
      const data = await api("/api/dialog/directory", { method: "POST", body: JSON.stringify({ title }) });
      if (data.path) setter(data.path);
    } catch (error) { setMessage(error.message, "error"); }
  };
  const chooseVideos = async () => {
    setBusy(true);
    try {
      const selected = await api("/api/dialog/videos", { method: "POST", body: JSON.stringify({ title: "Select videos to sample" }) });
      if (!selected.paths?.length) return;
      setVideoPaths(selected.paths);
      const data = await api("/api/videos/inspect", { method: "POST", body: JSON.stringify({ video_paths: selected.paths, interval_seconds: 0.5 }) });
      setInspection(data);
      if (data.warnings?.length) setMessage(data.warnings.join("\n"), "warning");
    } catch (error) { setMessage(`Could not inspect videos: ${error.message}`, "error"); }
    finally { setBusy(false); }
  };
  const createVideoDataset = async () => {
    if (!videoPaths.length || !outputParent.trim() || !projectName.trim()) return setMessage("Choose videos, a save location, and a project name first.", "error");
    const classes = classNames.split(",").map((value) => value.trim()).filter(Boolean);
    if (!classes.length) return setMessage("Enter at least one class name.", "error");
    setBusy(true);
    setProgress({ state: "queued", percent: 0, message: "Starting video import…", completed: 0, total: inspection?.expected_frames || 0 });
    try {
      const started = await api("/api/videos/jobs", { method: "POST", body: JSON.stringify({
        video_paths: videoPaths,
        output_parent: outputParent.trim(),
        project_name: projectName.trim(),
        annotation_mode: annotationMode,
        class_names: classes,
        interval_seconds: 0.5,
        ratios: { train: 0.7, val: 0.2, test: 0.1 },
        seed: 42
      }) });
      jobIdRef.current = started.job_id;
      while (jobIdRef.current === started.job_id) {
        const status = await api(`/api/videos/jobs/${started.job_id}`);
        setProgress(status);
        if (status.state === "complete") {
          jobIdRef.current = null;
          // Confirm the active backend project instead of relying only on the
          // project metadata carried by the background-job response.
          const opened = await api("/api/project/open", { method: "POST", body: JSON.stringify({ project_dir: status.result.project_dir }) });
          onOpen(opened.project, opened.summary);
          const warningText = status.result.warnings?.length ? `\n${status.result.warnings.join("\n")}` : "";
          setMessage(`Imported ${status.result.counts.frames} frames from ${status.result.counts.videos} videos.${warningText}`, status.result.warnings?.length ? "warning" : "success");
          break;
        }
        if (status.state === "failed") throw new Error(status.error || "Video import failed.");
        if (status.state === "cancelled") { setMessage("Video import cancelled.", "warning"); break; }
        await new Promise((resolve) => window.setTimeout(resolve, 250));
      }
    } catch (error) { setMessage(error.message, "error"); }
    finally { jobIdRef.current = null; setBusy(false); }
  };
  const cancel = async () => {
    if (!jobIdRef.current) return;
    try { await api(`/api/videos/jobs/${jobIdRef.current}/cancel`, { method: "POST", body: "{}" }); }
    catch (error) { setMessage(error.message, "error"); }
  };
  const openProject = async () => {
    if (!projectDir.trim()) return;
    setBusy(true);
    try {
      const data = await api("/api/project/open", { method: "POST", body: JSON.stringify({ project_dir: projectDir.trim() }) });
      onOpen(data.project, data.summary);
    } catch (error) { setMessage(error.message, "error"); }
    finally { setBusy(false); }
  };
  const chooseAnnotationMode = (mode) => {
    setAnnotationMode(mode);
    setClassNames((current) => {
      if (!["Object", "Defect", DEFAULT_SEGMENTATION_CLASSES, "Good, Defective"].includes(current)) return current;
      if (mode === "segmentation") return DEFAULT_SEGMENTATION_CLASSES;
      if (mode === "classification") return "Good, Defective";
      return "Object";
    });
  };

  return <section className="home-page">
    <div className="home-hero">
      <p className="eyebrow">Local computer-vision datasets</p>
      <h1>Turn images and videos into training data.</h1>
      <p>Build a focused detection, classification, or instance-segmentation dataset and export leakage-safe training, validation, and test splits.</p>
    </div>
    {view === "home" && <div className="home-actions">
      <button className="home-action primary harvest-action" onClick={onHarvest}><ScanSearch size={28} /><span><strong>Harvest defective fruit from video</strong><small>Click a naartjie once, track it, and save transparent PNGs every 0.25 seconds.</small></span><ChevronRight size={22} /></button>
      <button className="home-action primary" data-testid="new-video-dataset" onClick={() => setView("video")}><Sparkles size={28} /><span><strong>New video dataset</strong><small>Select videos and extract one frame every 0.5 seconds.</small></span><ChevronRight size={22} /></button>
      <button className="home-action" onClick={() => setView("open")}><FolderOpen size={28} /><span><strong>Open project</strong><small>Continue an existing annotation project.</small></span><ChevronRight size={22} /></button>
      <button className="home-action image-folder" onClick={onAdvanced}><Brush size={28} /><span><strong>New image-folder dataset</strong><small>Create an instance-segmentation project with SAM masks, or use the detection, classification, synthetic-data, and deblur tools.</small></span><ChevronRight size={22} /></button>
    </div>}
    {view === "open" && <div className="setup-card">
      <div className="setup-heading"><button className="text-button" onClick={() => setView("home")}><ChevronLeft size={16} /> Back</button><div><p className="eyebrow">Open</p><h2>Continue a project</h2></div></div>
      <label><span>Project folder</span><div className="path-row"><input value={projectDir} onChange={(event) => setProjectDir(event.target.value)} /><button onClick={() => browseDirectory(setProjectDir, "Select Data Labeller project")}><FolderOpen size={17} /> Browse</button></div></label>
      <button className="primary-action" onClick={openProject} disabled={busy}>{busy ? <LoaderCircle className="spin" size={17} /> : <FolderOpen size={17} />} Open project</button>
    </div>}
    {view === "video" && <div className="setup-card video-wizard" data-testid="video-wizard">
      <div className="setup-heading"><button className="text-button" onClick={() => setView("home")} disabled={busy}><ChevronLeft size={16} /> Back</button><div><p className="eyebrow">Import · Label · Review · Export</p><h2>Create a video dataset</h2></div></div>
      <div className="wizard-grid">
        <section className="wizard-section"><div className="step-number">1</div><div><h3>Select videos</h3><p>Frames remain full resolution. Original videos are never changed.</p><button onClick={chooseVideos} disabled={busy}><FolderOpen size={17} /> Choose videos</button>{inspection && <div className="import-summary"><strong>{inspection.video_count} video{inspection.video_count === 1 ? "" : "s"} · about {inspection.expected_frames} frames</strong><span>One image every 0.5 seconds</span>{inspection.videos.map((video) => <small key={video.video_id}>{video.file_name} · {(video.duration_ms / 1000).toFixed(1)}s · {video.width}×{video.height} · {video.expected_frames} frames</small>)}</div>}</div></section>
        <section className="wizard-section"><div className="step-number">2</div><div><h3>Choose the labeling task</h3><div className="mode-picker"><button className={annotationMode === "detection" ? "active" : ""} onClick={() => chooseAnnotationMode("detection")}><BoxSelect size={20} /> Object detection<small>Draw bounding boxes</small></button><button className={annotationMode === "segmentation" ? "active" : ""} onClick={() => chooseAnnotationMode("segmentation")}><Brush size={20} /> Segmentation<small>Paint or propose precise masks</small></button><button className={annotationMode === "classification" ? "active" : ""} onClick={() => chooseAnnotationMode("classification")}><Tag size={20} /> Classification<small>Assign one class per frame</small></button></div><label><span>{annotationMode === "classification" ? "Image classes" : annotationMode === "segmentation" ? "Mask classes" : "Object classes"}, separated by commas</span><input value={classNames} onChange={(event) => setClassNames(event.target.value)} placeholder={annotationMode === "classification" ? "Good, Defective" : annotationMode === "segmentation" ? DEFAULT_SEGMENTATION_CLASSES : "Object"} /></label></div></section>
        <section className="wizard-section"><div className="step-number">3</div><div><h3>Save and split</h3><label><span>Project name</span><input value={projectName} onChange={(event) => setProjectName(event.target.value)} /></label><label><span>Save inside</span><div className="path-row"><input value={outputParent} onChange={(event) => setOutputParent(event.target.value)} /><button onClick={() => browseDirectory(setOutputParent, "Choose where to save the project")}><FolderOpen size={17} /> Browse</button></div></label><p className="split-note"><Check size={16} /> Automatic 70% training / 20% validation / 10% test split. Every video stays entirely in one split.</p></div></section>
      </div>
      {progress && <div className="synthetic-progress" role="status" aria-live="polite"><div className="progress-meta"><strong>{progress.message}</strong><span>{progress.percent || 0}%</span></div><div className="progress-track" role="progressbar" aria-label="Video frame extraction" aria-valuemin="0" aria-valuemax="100" aria-valuenow={progress.percent || 0}><span className="progress-fill" style={{ width: `${progress.percent || 0}%` }} /></div><small>{progress.completed || 0} of {progress.total || inspection?.expected_frames || 0} frames · {progress.stage || progress.state}</small></div>}
      <div className="wizard-actions"><button className="primary-action" onClick={createVideoDataset} disabled={busy || !inspection?.video_count}>{busy ? <LoaderCircle className="spin" size={17} /> : <Sparkles size={17} />} Extract frames and create project</button>{jobIdRef.current && <button onClick={cancel}><X size={17} /> Cancel</button>}</div>
    </div>}
  </section>;
}

const loadBrowserImage = (source) => new Promise((resolve, reject) => {
  const image = new Image();
  image.onload = () => resolve(image);
  image.onerror = () => reject(new Error("Could not load the harvest preview."));
  image.src = source;
});

function HarvestWorkspace({ currentClasses, onOpenProject, onExit, setMessage }) {
  const [videoPaths, setVideoPaths] = useState([]);
  const [outputParent, setOutputParent] = useState("dataset");
  const [name, setName] = useState("Defective naartjies");
  const [session, setSession] = useState(null);
  const [sessionDir, setSessionDir] = useState("");
  const [restoreChecked, setRestoreChecked] = useState(false);
  const [resumeError, setResumeError] = useState("");
  const [mode, setMode] = useState("scan");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [points, setPoints] = useState([]);
  const [promptTool, setPromptTool] = useState("positive");
  const [proposals, setProposals] = useState([]);
  const [proposalIndex, setProposalIndex] = useState(0);
  const [detections, setDetections] = useState([]);
  const [selectedTrackId, setSelectedTrackId] = useState(null);
  const [autoStep, setAutoStep] = useState(false);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [harvestUndoCount, setHarvestUndoCount] = useState(0);
  const [scanTimeMs, setScanTimeMs] = useState(0);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const scanOverlayRef = useRef(null);
  const renderTokenRef = useRef(0);
  const scanRenderTokenRef = useRef(0);
  const boxStartRef = useRef(null);
  const positioningRef = useRef(false);
  const suppressNextScanPauseRef = useRef(false);
  const harvestEditHistoryRef = useRef([]);

  const activeVideo = session?.videos?.find((item) => item.video_id === session.current_video_id) || session?.videos?.[0];
  const activeTracks = useMemo(() => selectActiveHarvestTracks(session), [session]);
  const openTracks = useMemo(() => selectOpenHarvestTracks(session), [session]);
  const missingMaskCount = openTracks.filter((item) => item.status === "needs_mask").length;
  const selectedTrackRecord = session?.tracks?.find((item) => item.id === selectedTrackId) || null;
  const selectedTrack = ["active", "needs_mask"].includes(selectedTrackRecord?.status) ? selectedTrackRecord : null;
  const currentProposal = proposals[proposalIndex] || null;
  const visibleCurrentMasks = useMemo(
    () => selectVisibleHarvestMasks(session, activeVideo?.video_id, session?.current_time_ms),
    [activeVideo?.video_id, session]
  );
  const visibleScanMasks = useMemo(
    () => selectVisibleHarvestMasks(session, activeVideo?.video_id, scanTimeMs),
    [activeVideo?.video_id, scanTimeMs, session]
  );
  const savedCount = session?.samples?.filter((item) => item.status === "saved").length || 0;
  const skippedCount = session?.samples?.filter((item) => item.status === "skipped").length || 0;

  const harvestEditSnapshot = () => ({
    points: clone(points),
    proposals: clone(proposals),
    proposalIndex,
    detections: clone(detections),
    selectedTrackId,
  });
  const pushHarvestEditHistory = (snapshot = harvestEditSnapshot()) => {
    harvestEditHistoryRef.current = [...harvestEditHistoryRef.current.slice(-19), snapshot];
    setHarvestUndoCount(harvestEditHistoryRef.current.length);
  };
  const resetHarvestEditHistory = useCallback(() => {
    harvestEditHistoryRef.current = [];
    setHarvestUndoCount(0);
  }, []);
  const undoHarvestEdit = useCallback(() => {
    const previous = harvestEditHistoryRef.current.pop();
    if (!previous) return;
    setPoints(previous.points);
    setProposals(previous.proposals);
    setProposalIndex(previous.proposalIndex);
    setDetections(previous.detections);
    setSelectedTrackId(previous.selectedTrackId);
    setHarvestUndoCount(harvestEditHistoryRef.current.length);
  }, []);
  const clearHarvestPrompt = () => {
    if (!points.length && !proposals.length && !detections.length) return;
    pushHarvestEditHistory();
    setPoints([]);
    setProposals([]);
    setProposalIndex(0);
    setDetections([]);
  };
  const startNewFruit = () => {
    if (selectedTrackId || points.length || proposals.length || detections.length) pushHarvestEditHistory();
    setSelectedTrackId(null);
    setPoints([]);
    setProposals([]);
    setProposalIndex(0);
    setDetections([]);
  };

  useEffect(() => {
    let active = true;
    const remembered = readRememberedHarvestSession(window.localStorage);
    if (!remembered) {
      setRestoreChecked(true);
      return () => { active = false; };
    }
    setBusy(true);
    setProgress(`Continuing ${remembered.name || "your last harvest session"}…`);
    api("/api/harvest/sessions/open", { method: "POST", body: JSON.stringify({ session_dir: remembered.sessionDir }) })
      .then((data) => {
        if (!active) return;
        if (data.session.state === "finished") {
          forgetRememberedHarvestSession(window.localStorage);
          setMessage("The previous harvest session was already finished. Start a new harvest or resume another folder.", "info");
          return;
        }
        setSessionDir(data.session_dir || remembered.sessionDir);
        setSession(data.session);
        setSelectedTrackId(selectOpenHarvestTracks(data.session)[0]?.id || null);
        setMode(harvestModeAfterLoad(data.session));
        resetHarvestEditHistory();
        rememberHarvestSession(window.localStorage, {
          sessionDir: data.session_dir || remembered.sessionDir,
          sessionId: data.session.session_id,
          name: data.session.name,
          updatedAt: data.session.updated_at
        });
        setMessage(`Continued ${data.session.name} at ${(Number(data.session.current_time_ms) / 1000).toFixed(2)} seconds.`, "success");
      })
      .catch((error) => {
        if (!active) return;
        setResumeError(error.message);
        setMessage(`Could not automatically continue the last harvest: ${error.message}`, "error");
      })
      .finally(() => {
        if (!active) return;
        setBusy(false);
        setProgress("");
        setRestoreChecked(true);
      });
    return () => { active = false; };
  }, [setMessage]);

  useEffect(() => {
    if (!session || !sessionDir || session.state === "finished") return;
    rememberHarvestSession(window.localStorage, {
      sessionDir,
      sessionId: session.session_id,
      name: session.name,
      updatedAt: session.updated_at
    });
  }, [session, sessionDir]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !activeVideo || mode !== "scan") return undefined;
    const savedTime = Math.max(0, Math.min(Number(session?.current_time_ms || 0), Number(activeVideo.duration_ms || 0))) / 1000;
    const restorePosition = () => {
      if (Math.abs(video.currentTime - savedTime) > 0.02) video.currentTime = savedTime;
      setScanTimeMs(savedTime * 1000);
    };
    if (video.readyState >= 1) restorePosition();
    else video.addEventListener("loadedmetadata", restorePosition, { once: true });
    return () => video.removeEventListener("loadedmetadata", restorePosition);
  }, [activeVideo, mode, session?.current_time_ms]);

  const chooseVideos = async () => {
    try {
      const selected = await api("/api/dialog/videos", { method: "POST", body: JSON.stringify({ title: "Select videos containing defective naartjies" }) });
      if (selected.paths?.length) setVideoPaths(selected.paths);
    } catch (error) { setMessage(error.message, "error"); }
  };
  const chooseOutput = async () => {
    try {
      const selected = await api("/api/dialog/directory", { method: "POST", body: JSON.stringify({ title: "Choose where to save harvested naartjies" }) });
      if (selected.path) setOutputParent(selected.path);
    } catch (error) { setMessage(error.message, "error"); }
  };
  const createHarvest = async () => {
    if (!videoPaths.length || !outputParent.trim() || !name.trim()) return setMessage("Choose at least one video, an output folder, and a harvest name.", "error");
    setBusy(true);
    try {
      const data = await api("/api/harvest/sessions", { method: "POST", body: JSON.stringify({ video_paths: videoPaths, output_parent: outputParent, name, interval_ms: HARVEST_STEP_MS, segmentation_classes: currentClasses || [] }) });
      setSessionDir(data.session_dir || ""); setSession(data.session); setSelectedTrackId(selectOpenHarvestTracks(data.session)[0]?.id || null); setMode(harvestModeAfterLoad(data.session)); setResumeError(""); resetHarvestEditHistory();
      rememberHarvestSession(window.localStorage, { sessionDir: data.session_dir, sessionId: data.session.session_id, name: data.session.name, updatedAt: data.session.updated_at });
      if (data.session.warnings?.length) setMessage(data.session.warnings.join("\n"), "warning");
    } catch (error) { setMessage(`Could not create harvest: ${error.message}`, "error"); }
    finally { setBusy(false); }
  };
  const resumeHarvest = async () => {
    setBusy(true);
    try {
      const selected = await api("/api/dialog/directory", { method: "POST", body: JSON.stringify({ title: "Select a folder containing harvest-session.json" }) });
      if (!selected.path) return;
      const data = await api("/api/harvest/sessions/open", { method: "POST", body: JSON.stringify({ session_dir: selected.path }) });
      if (data.session.state === "finished") {
        forgetRememberedHarvestSession(window.localStorage);
        return setMessage("That harvest has already been finished. Its generated segmentation project remains available.", "info");
      }
      const openedDir = data.session_dir || selected.path;
      setSessionDir(openedDir); setSession(data.session); setSelectedTrackId(selectOpenHarvestTracks(data.session)[0]?.id || null); setMode(harvestModeAfterLoad(data.session)); setResumeError(""); resetHarvestEditHistory();
      rememberHarvestSession(window.localStorage, { sessionDir: openedDir, sessionId: data.session.session_id, name: data.session.name, updatedAt: data.session.updated_at });
    } catch (error) { setMessage(`Could not resume harvest: ${error.message}`, "error"); }
    finally { setBusy(false); }
  };

  const drawMask = async (context, source, color, opacity, token, tokenRef = renderTokenRef) => {
    const maskSource = source.startsWith("data:") ? source : `${source}${source.includes("?") ? "&" : "?"}v=${encodeURIComponent(session?.updated_at || "")}`;
    const mask = await loadBrowserImage(maskSource);
    if (token !== tokenRef.current) return;
    const temporary = document.createElement("canvas");
    temporary.width = context.canvas.width; temporary.height = context.canvas.height;
    const temporaryContext = temporary.getContext("2d", { willReadFrequently: true });
    temporaryContext.drawImage(mask, 0, 0, temporary.width, temporary.height);
    const pixels = temporaryContext.getImageData(0, 0, temporary.width, temporary.height);
    const rgb = [Number.parseInt(color.slice(1, 3), 16), Number.parseInt(color.slice(3, 5), 16), Number.parseInt(color.slice(5, 7), 16)];
    for (let index = 0; index < pixels.data.length; index += 4) {
      const strength = Math.max(pixels.data[index], pixels.data[index + 1], pixels.data[index + 2]);
      pixels.data[index] = rgb[0]; pixels.data[index + 1] = rgb[1]; pixels.data[index + 2] = rgb[2]; pixels.data[index + 3] = strength > 20 ? Math.round(255 * opacity) : 0;
    }
    temporaryContext.putImageData(pixels, 0, 0);
    context.drawImage(temporary, 0, 0);
  };

  const renderHarvestFrame = useCallback(async () => {
    if (!session || !activeVideo || mode !== "track" || !canvasRef.current) return;
    const token = ++renderTokenRef.current;
    try {
      const source = await loadBrowserImage(`${session.frame_url}&v=${encodeURIComponent(session.updated_at || "")}`);
      if (token !== renderTokenRef.current || !canvasRef.current) return;
      const canvas = canvasRef.current;
      canvas.width = Number(activeVideo.width); canvas.height = Number(activeVideo.height);
      const context = canvas.getContext("2d");
      context.drawImage(source, 0, 0, canvas.width, canvas.height);
      for (const track of visibleCurrentMasks) await drawMask(context, track.mask_url, track.color, track.id === selectedTrackId ? 0.52 : 0.34, token);
      if (currentProposal?.mask_data) await drawMask(context, currentProposal.mask_data, "#fff06a", 0.48, token);
      context.lineWidth = Math.max(2, canvas.width / 600);
      context.font = `${Math.max(13, canvas.width / 70)}px sans-serif`;
      for (const detection of detections) {
        const [x, y, width, height] = detection.bbox;
        context.strokeStyle = "#68d7ff"; context.strokeRect(x, y, width, height);
        context.fillStyle = "#68d7ff"; context.fillText(`${detection.label} ${Math.round(detection.score * 100)}%`, x + 3, Math.max(15, y - 5));
      }
      for (const point of points) {
        context.beginPath(); context.arc(point.x, point.y, Math.max(5, canvas.width / 160), 0, Math.PI * 2);
        context.fillStyle = point.label ? "#57e58e" : "#ff6874"; context.fill();
        context.strokeStyle = "#081018"; context.stroke();
      }
    } catch (error) { if (token === renderTokenRef.current) setMessage(error.message, "error"); }
  }, [activeVideo, currentProposal, detections, mode, points, selectedTrackId, session, setMessage, visibleCurrentMasks]);
  useEffect(() => { renderHarvestFrame(); }, [renderHarvestFrame]);

  useEffect(() => {
    const canvas = scanOverlayRef.current;
    if (!canvas || !activeVideo || mode !== "scan") return undefined;
    const token = ++scanRenderTokenRef.current;
    canvas.width = Number(activeVideo.width);
    canvas.height = Number(activeVideo.height);
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    const render = async () => {
      try {
        for (const track of visibleScanMasks) {
          await drawMask(context, track.mask_url, track.color, track.id === selectedTrackId ? 0.58 : 0.36, token, scanRenderTokenRef);
        }
      } catch (error) {
        if (token === scanRenderTokenRef.current) setMessage(`Could not display a saved mask: ${error.message}`, "error");
      }
    };
    render();
    return () => { scanRenderTokenRef.current += 1; };
  }, [activeVideo, mode, selectedTrackId, setMessage, session?.updated_at, visibleScanMasks]);

  const requestProposal = async (nextPoints, box = null) => {
    if (!session || !activeVideo) return;
    const previousEdit = harvestEditSnapshot();
    setBusy(true); setProgress("Creating whole-fruit mask with SAM 2…");
    try {
      const started = await api(`/api/harvest/sessions/${session.session_id}/proposals`, { method: "POST", body: JSON.stringify({ video_id: activeVideo.video_id, time_ms: session.current_time_ms, request_id: crypto.randomUUID(), points: nextPoints, box }) });
      const data = await waitForJob(started.job_id);
      pushHarvestEditHistory(previousEdit);
      setPoints(nextPoints); setProposals(data.proposals || []); setProposalIndex(0);
    } catch (error) { setMessage(`SAM mask failed: ${error.message}`, "error"); }
    finally { setBusy(false); setProgress(""); }
  };
  const canvasPoint = (event) => {
    const canvas = canvasRef.current; const rect = canvas.getBoundingClientRect();
    const fitted = containedFrameRect(rect.width, rect.height, canvas.width, canvas.height);
    return {
      x: Math.max(0, Math.min(canvas.width - 1, (event.clientX - rect.left - fitted.x) * canvas.width / Math.max(1, fitted.width))),
      y: Math.max(0, Math.min(canvas.height - 1, (event.clientY - rect.top - fitted.y) * canvas.height / Math.max(1, fitted.height)))
    };
  };
  const onHarvestPointerDown = (event) => {
    if (busy) return;
    const point = canvasPoint(event);
    if (promptTool === "box") { boxStartRef.current = point; canvasRef.current.setPointerCapture(event.pointerId); return; }
    const detection = detections.find((item) => { const [x, y, width, height] = item.bbox; return point.x >= x && point.x <= x + width && point.y >= y && point.y <= y + height; });
    if (detection) requestProposal([], [detection.bbox[0], detection.bbox[1], detection.bbox[0] + detection.bbox[2], detection.bbox[1] + detection.bbox[3]]);
    else requestProposal([...points, { ...point, label: promptTool === "negative" ? 0 : 1 }]);
  };
  const onHarvestPointerUp = (event) => {
    const start = boxStartRef.current;
    if (!start) return;
    boxStartRef.current = null;
    const end = canvasPoint(event);
    if (Math.abs(end.x - start.x) > 5 && Math.abs(end.y - start.y) > 5) requestProposal([], [Math.min(start.x, end.x), Math.min(start.y, end.y), Math.max(start.x, end.x), Math.max(start.y, end.y)]);
  };

  const pauseAtSample = async () => {
    if (suppressNextScanPauseRef.current) {
      suppressNextScanPauseRef.current = false;
      return;
    }
    if (!videoRef.current || !session || openTracks.length || positioningRef.current) return;
    positioningRef.current = true;
    videoRef.current.pause();
    const timeMs = snapHarvestTime(videoRef.current.currentTime * 1000, activeVideo.duration_ms);
    setBusy(true);
    try {
      const data = await api(`/api/harvest/sessions/${session.session_id}/position`, { method: "POST", body: JSON.stringify({ video_id: activeVideo.video_id, time_ms: timeMs }) });
      setSession(data.session); setMode("track"); setPoints([]); setProposals([]); setDetections([]);
      resetHarvestEditHistory();
    } catch (error) { setMessage(error.message, "error"); }
    finally { positioningRef.current = false; setBusy(false); }
  };
  const findFruit = async () => {
    const previousEdit = harvestEditSnapshot();
    setBusy(true); setProgress("Finding naartjies with Grounding DINO…");
    try {
      const started = await api(`/api/harvest/sessions/${session.session_id}/detections`, { method: "POST", body: JSON.stringify({ video_id: activeVideo.video_id, time_ms: session.current_time_ms }) });
      const data = await waitForJob(started.job_id);
      pushHarvestEditHistory(previousEdit);
      setDetections(data.proposals || []);
    } catch (error) { setMessage(`Fruit detection failed: ${error.message}`, "error"); }
    finally { setBusy(false); setProgress(""); }
  };
  const acceptProposal = async () => {
    if (!currentProposal) return;
    setBusy(true);
    try {
      if (selectedTrack) {
        const data = await api(`/api/harvest/sessions/${session.session_id}/tracks/${selectedTrack.id}`, { method: "PATCH", body: JSON.stringify({ mask_data: currentProposal.mask_data }) });
        setSession(data.session);
      } else {
        const data = await api(`/api/harvest/sessions/${session.session_id}/tracks`, { method: "POST", body: JSON.stringify({ video_id: activeVideo.video_id, time_ms: session.current_time_ms, mask_data: currentProposal.mask_data, prompt_metadata: { points, source: detections.length ? "dino_or_manual" : "manual_click", score: currentProposal.score } }) });
        setSession(data.session); setSelectedTrackId(data.track.id);
      }
      setPoints([]); setProposals([]); setDetections([]);
      resetHarvestEditHistory();
    } catch (error) { setMessage(error.message, "error"); }
    finally { setBusy(false); }
  };
  const saveProposalOnly = async () => {
    if (!currentProposal || selectedTrack || busy) return;
    setBusy(true);
    try {
      const data = await api(`/api/harvest/sessions/${session.session_id}/single-mask`, {
        method: "POST",
        body: JSON.stringify({
          video_id: activeVideo.video_id,
          time_ms: session.current_time_ms,
          mask_data: currentProposal.mask_data,
          prompt_metadata: { points, source: detections.length ? "dino_or_manual" : "manual_click", score: currentProposal.score }
        })
      });
      setSession(data.session);
      setSelectedTrackId(null);
      setPoints([]);
      setProposals([]);
      setProposalIndex(0);
      setDetections([]);
      resetHarvestEditHistory();
      if (data.sample.status !== "saved") setMessage(`${data.track.id} was not saved: ${data.sample.reason}.`, "warning");
    } catch (error) { setMessage(`Could not save this mask: ${error.message}`, "error"); }
    finally { setBusy(false); }
  };
  const deleteProposalMask = () => {
    if (!currentProposal || busy) return;
    pushHarvestEditHistory();
    const remaining = proposals.filter((_proposal, index) => index !== proposalIndex);
    setProposals(remaining);
    setProposalIndex(remaining.length ? Math.min(proposalIndex, remaining.length - 1) : 0);
  };
  const deleteCurrentTrackMask = async () => {
    if (!selectedTrackRecord?.mask_url || selectedTrackRecord.status !== "active" || busy) return;
    if (!window.confirm(`Delete the current mask for ${selectedTrackRecord.id}?\n\nThe track will pause until you draw and apply a replacement mask. A recovery backup will be kept.`)) return;
    setBusy(true);
    setAutoStep(false);
    try {
      const data = await api(`/api/harvest/sessions/${session.session_id}/tracks/${selectedTrackRecord.id}/mask`, { method: "DELETE" });
      setSession(data.session);
      setPoints([]);
      setProposals([]);
      setProposalIndex(0);
      setDetections([]);
      resetHarvestEditHistory();
      setMessage(`${selectedTrackRecord.id}'s current mask was deleted. Draw a replacement to continue this track. Recovery backup: ${data.backup_dir}.`, "success");
    } catch (error) { setMessage(`Could not delete the current mask: ${error.message}`, "error"); }
    finally { setBusy(false); }
  };
  const deleteSelectedTrack = async () => {
    if (!selectedTrackRecord || busy) return;
    const saved = Number(selectedTrackRecord.saved_count || 0);
    const skipped = Number(selectedTrackRecord.skipped_count || 0);
    const sampleDescription = saved || skipped
      ? ` This also removes ${saved} saved PNG${saved === 1 ? "" : "s"} and ${skipped} skipped sample${skipped === 1 ? "" : "s"} from this harvest.`
      : "";
    if (!window.confirm(`Delete ${selectedTrackRecord.id}?${sampleDescription}\n\nA recoverable backup will be kept inside the harvest folder.`)) return;
    setBusy(true);
    setAutoStep(false);
    try {
      const data = await api(`/api/harvest/sessions/${session.session_id}/tracks/${selectedTrackRecord.id}`, { method: "DELETE" });
      setSession(data.session);
      setSelectedTrackId(null);
      setPoints([]);
      setProposals([]);
      setProposalIndex(0);
      setDetections([]);
      resetHarvestEditHistory();
      const cleanupNote = data.cleanup_warnings?.length ? ` ${data.cleanup_warnings.join(" ")}` : "";
      setMessage(`${selectedTrackRecord.id} deleted. Recovery backup: ${data.backup_dir}.${cleanupNote}`, data.cleanup_warnings?.length ? "warning" : "success");
    } catch (error) { setMessage(`Could not delete ${selectedTrackRecord.id}: ${error.message}`, "error"); }
    finally { setBusy(false); }
  };

  const waitForJob = async (jobId) => {
    setCurrentJobId(jobId);
    try {
      while (true) {
        const job = await api(`/api/harvest/jobs/${jobId}`);
        setProgress(job.message || job.stage || "Processing…");
        if (job.state === "complete") return job.result;
        if (job.state === "failed") throw new Error(job.error || job.message);
        if (job.state === "cancelled") throw new Error("Harvest operation was cancelled.");
        await new Promise((resolve) => window.setTimeout(resolve, 300));
      }
    } finally {
      setCurrentJobId((value) => value === jobId ? null : value);
    }
  };
  const cancelJob = async () => {
    if (!currentJobId) return;
    try {
      await api(`/api/harvest/jobs/${currentJobId}/cancel`, { method: "POST", body: "{}" });
      setProgress("Cancellation requested; completed PNGs remain safe.");
    } catch (error) { setMessage(error.message, "error"); }
  };
  const previousSample = useCallback(async () => {
    if (!session || busy || Number(session.current_time_ms) <= 0) return;
    setBusy(true); setAutoStep(false); setProgress("Restoring the previous 0.25-second sample…");
    try {
      const data = await api(`/api/harvest/sessions/${session.session_id}/previous`, { method: "POST", body: "{}" });
      setSession(data.session);
      setSelectedTrackId((current) => data.restored_track_ids.includes(current) ? current : (data.restored_track_ids[0] || null));
      setPoints([]); setProposals([]); setProposalIndex(0); setDetections([]); resetHarvestEditHistory();
      const replacementNote = data.removed_sample_count
        ? ` The previous saved output was moved to a recovery backup; saving again will replace it with your edit.`
        : "";
      setMessage(`Returned to ${(Number(data.to_time_ms) / 1000).toFixed(2)} seconds.${replacementNote}`, "success");
    } catch (error) { setMessage(`Could not go to the previous sample: ${error.message}`, "error"); }
    finally { setBusy(false); setProgress(""); }
  }, [busy, resetHarvestEditHistory, session, setMessage]);
  const advance = useCallback(async ({ skip = false, end = false } = {}) => {
    if (!session || busy || missingMaskCount) return;
    setBusy(true); setProgress("Saving current cut-outs and tracking to the next sample…");
    try {
      const started = await api(`/api/harvest/sessions/${session.session_id}/advance`, { method: "POST", body: JSON.stringify({ request_id: crypto.randomUUID(), skip_track_ids: skip && selectedTrackId ? [selectedTrackId] : [], end_track_ids: end && selectedTrackId ? [selectedTrackId] : [] }) });
      const result = await waitForJob(started.job_id);
      setSession(result.session); setPoints([]); setProposals([]); setDetections([]);
      resetHarvestEditHistory();
      if (end) setSelectedTrackId(null);
      const ended = result.session.tracks.filter((item) => item.status === "ended" && item.end_time_ms === result.session.current_time_ms - HARVEST_STEP_MS && item.end_reason && item.end_reason !== "single-mask capture");
      if (ended.length) setMessage(ended.map((item) => `${item.id} ended: ${item.end_reason}`).join("\n"), "warning");
    } catch (error) { setAutoStep(false); setMessage(error.message, "error"); }
    finally { setBusy(false); setProgress(""); }
  }, [busy, missingMaskCount, resetHarvestEditHistory, selectedTrackId, session, setMessage]);
  useEffect(() => {
    if (!autoStep || busy) return undefined;
    if (missingMaskCount) { setAutoStep(false); return undefined; }
    if (!activeTracks.length) { setAutoStep(false); return undefined; }
    const timer = window.setTimeout(() => advance(), 180);
    return () => window.clearTimeout(timer);
  }, [activeTracks.length, advance, autoStep, busy, missingMaskCount, session?.current_time_ms]);
  useEffect(() => {
    const handler = (event) => {
      if (mode !== "track" || busy || shouldIgnoreShortcut(event.target)) return;
      const commandKey = event.ctrlKey || event.metaKey;
      if (commandKey && event.key.toLowerCase() === "z" && !event.shiftKey && !event.repeat) { event.preventDefault(); undoHarvestEdit(); }
      else if (event.key === "ArrowLeft" && !event.repeat) { event.preventDefault(); previousSample(); }
      else if (event.key === "ArrowRight" && !event.repeat) { event.preventDefault(); advance(); }
      else if (event.code === "Space" && activeTracks.length && !missingMaskCount && !event.repeat) { event.preventDefault(); setAutoStep((value) => !value); }
      else if (event.key === "Escape") clearHarvestPrompt();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [activeTracks.length, advance, busy, clearHarvestPrompt, missingMaskCount, mode, previousSample, undoHarvestEdit]);

  const changeVideo = async (videoId) => {
    if (openTracks.length) return;
    setBusy(true);
    try {
      const data = await api(`/api/harvest/sessions/${session.session_id}/position`, { method: "POST", body: JSON.stringify({ video_id: videoId, time_ms: 0 }) });
      setSession(data.session); setMode("scan"); setScanTimeMs(0); setPoints([]); setProposals([]); setProposalIndex(0); setDetections([]); resetHarvestEditHistory();
    } catch (error) { setMessage(error.message, "error"); }
    finally { setBusy(false); }
  };
  const returnToScan = () => {
    setMode("scan"); setAutoStep(false); setScanTimeMs(Number(session.current_time_ms));
    setPoints([]); setProposals([]); setProposalIndex(0); setDetections([]); resetHarvestEditHistory();
    requestAnimationFrame(() => { if (videoRef.current) videoRef.current.currentTime = Number(session.current_time_ms) / 1000; });
  };
  const selectHarvestTrack = (track) => {
    setSelectedTrackId(track.id);
    if (mode !== "scan" || track.video_id !== activeVideo?.video_id || !videoRef.current) return;
    const maskTimeMs = Number(track.current_time_ms || track.seed_time_ms || 0);
    if (!videoRef.current.paused) {
      suppressNextScanPauseRef.current = true;
      videoRef.current.pause();
    }
    videoRef.current.currentTime = maskTimeMs / 1000;
    setScanTimeMs(maskTimeMs);
  };
  const finish = async () => {
    if ((!savedCount && !activeTracks.length) || busy) return;
    setBusy(true); setAutoStep(false);
    try {
      let finalSession = session;
      if (activeTracks.length) {
        setProgress("Saving the current sample and ending active fruit tracks…");
        const commit = await api(`/api/harvest/sessions/${session.session_id}/advance`, { method: "POST", body: JSON.stringify({ request_id: crypto.randomUUID(), skip_track_ids: [], end_track_ids: activeTracks.map((track) => track.id) }) });
        const committed = await waitForJob(commit.job_id);
        finalSession = committed.session;
        setSession(finalSession); setSelectedTrackId(null);
      }
      setProgress("Creating the unlabelled defect-segmentation project…");
      const started = await api(`/api/harvest/sessions/${session.session_id}/finish`, { method: "POST", body: "{}" });
      const result = await waitForJob(started.job_id);
      forgetRememberedHarvestSession(window.localStorage);
      onOpenProject(result.project, result.summary);
      const finalSavedCount = finalSession.samples.filter((item) => item.status === "saved").length;
      setMessage(`Harvest complete: ${finalSavedCount} transparent naartjie PNGs are ready for manual defect labelling.`, "success");
    } catch (error) { setMessage(error.message, "error"); }
    finally { setBusy(false); setProgress(""); }
  };

  const startDifferentHarvest = () => {
    forgetRememberedHarvestSession(window.localStorage);
    setSession(null); setSessionDir(""); setMode("scan"); setSelectedTrackId(null); setAutoStep(false); setResumeError("");
    setScanTimeMs(0); resetHarvestEditHistory();
  };

  if (!session && !restoreChecked) return <section className="home-page harvest-setup-page"><div className="setup-card harvest-resume-card"><LoaderCircle className="spin" size={28} /><div><p className="eyebrow">Restoring workspace</p><h2>Continuing your last harvest session…</h2><p>The saved video, timestamp, masks, and active fruit tracks are being restored.</p></div></div></section>;

  if (!session) return <section className="home-page harvest-setup-page">
    <div className="home-hero"><p className="eyebrow">Video harvester</p><h1>Pick defective naartjies, not empty frames.</h1><p>The model extracts the whole fruit only. You will label its defects manually in the generated segmentation project.</p></div>
    <div className="setup-card harvest-setup">
      <div className="setup-heading"><button className="text-button" onClick={onExit}><ChevronLeft size={16} /> Back</button><div><p className="eyebrow">Scan · Select · Track · Label</p><h2>Create a harvest session</h2></div></div>
      {resumeError && <div className="harvest-resume-warning"><CircleAlert size={18} /><span><strong>The last session could not be opened automatically.</strong><small>{resumeError}</small></span><button className="compact" onClick={() => { forgetRememberedHarvestSession(window.localStorage); setResumeError(""); }}>Forget it</button></div>}
      <label><span>Source videos</span><div className="path-row"><input readOnly value={videoPaths.length ? `${videoPaths.length} video${videoPaths.length === 1 ? "" : "s"} selected` : "No videos selected"} /><button onClick={chooseVideos} disabled={busy}><FolderOpen size={17} /> Choose videos</button></div></label>
      {videoPaths.length > 0 && <div className="selected-video-list">{videoPaths.map((path) => <small key={path}>{path}</small>)}</div>}
      <label><span>Harvest name</span><input value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label><span>Save inside</span><div className="path-row"><input value={outputParent} onChange={(event) => setOutputParent(event.target.value)} /><button onClick={chooseOutput} disabled={busy}><FolderOpen size={17} /> Browse</button></div></label>
      <div className="harvest-setup-actions"><button className="primary-action" onClick={createHarvest} disabled={busy || !videoPaths.length}>{busy ? <LoaderCircle className="spin" size={17} /> : <ScanSearch size={17} />} Start harvesting</button><button onClick={resumeHarvest} disabled={busy}><FolderOpen size={17} /> Resume session</button></div>
    </div>
  </section>;

  return <section className="harvest-workspace">
    <header className="harvest-header"><div><p className="eyebrow">Whole-fruit harvester</p><h2>{session.name}</h2><small>{savedCount} PNGs saved · {skippedCount} skipped · 0.25-second samples</small></div><div><button onClick={onExit} disabled={busy}>Leave session</button><button onClick={startDifferentHarvest} disabled={busy}>Start another harvest</button><button className="success" onClick={finish} disabled={busy || (!savedCount && !activeTracks.length)}><Check size={16} /> Finish and label defects</button></div></header>
    <main className="harvest-main">
      <section className="harvest-viewer">
        <div className="harvest-viewer-toolbar"><label>Video <select value={activeVideo.video_id} onChange={(event) => changeVideo(event.target.value)} disabled={busy || openTracks.length}>{session.videos.map((item) => <option key={item.video_id} value={item.video_id}>{item.file_name}</option>)}</select></label><span>{((mode === "scan" ? scanTimeMs : session.current_time_ms) / 1000).toFixed(2)}s / {(activeVideo.duration_ms / 1000).toFixed(2)}s</span><span className={openTracks.length ? "track-badge active" : "track-badge"}>{missingMaskCount ? `${missingMaskCount} track${missingMaskCount === 1 ? "" : "s"} need a mask` : activeTracks.length ? `${activeTracks.length} active track${activeTracks.length === 1 ? "" : "s"}` : "Scanning"}</span></div>
        {mode === "scan" ? <div className="harvest-video-wrap"><video ref={videoRef} src={activeVideo.media_url} controls preload="metadata" onTimeUpdate={(event) => setScanTimeMs(event.currentTarget.currentTime * 1000)} onSeeked={(event) => setScanTimeMs(event.currentTarget.currentTime * 1000)} onPause={pauseAtSample} /><canvas ref={scanOverlayRef} className="harvest-scan-mask-overlay" aria-hidden="true" />{visibleScanMasks.length > 0 && <span className="harvest-visible-mask-count">{visibleScanMasks.length} saved mask{visibleScanMasks.length === 1 ? "" : "s"} visible</span>}<div className="harvest-scan-callout"><strong>Play until you see a defective naartjie.</strong><span>Pause the video to snap to the nearest 0.25-second sample and select it. Select a saved mask on the right to review it.</span><button onClick={pauseAtSample} disabled={busy}><MousePointer2 size={16} /> Select at current time</button></div></div> : <>
          <div className="harvest-canvas-wrap"><canvas ref={canvasRef} onPointerDown={onHarvestPointerDown} onPointerUp={onHarvestPointerUp} onContextMenu={(event) => event.preventDefault()} /></div>
          <div className="harvest-prompt-tools"><button className={!selectedTrackId ? "active" : ""} onClick={startNewFruit}><Plus size={15} /> New fruit</button><button className={promptTool === "positive" ? "active" : ""} onClick={() => setPromptTool("positive")}><Plus size={15} /> Include point</button><button className={promptTool === "negative" ? "active" : ""} onClick={() => setPromptTool("negative")}><X size={15} /> Exclude point</button><button className={promptTool === "box" ? "active" : ""} onClick={() => setPromptTool("box")}><BoxSelect size={15} /> Draw fruit box</button><button onClick={findFruit} disabled={busy}><ScanSearch size={15} /> Find naartjies</button><button onClick={clearHarvestPrompt} disabled={busy || (!points.length && !proposals.length && !detections.length)}>Clear prompt</button><button data-testid="undo-harvest-mask" title="Undo the last prompt or mask edit (Ctrl/Cmd+Z)" onClick={undoHarvestEdit} disabled={busy || !harvestUndoCount}><Undo2 size={15} /> Undo mask edit <kbd>Ctrl+Z</kbd></button></div>
          {currentProposal && <div className="harvest-proposal-bar"><span>SAM option {proposalIndex + 1} of {proposals.length} · {Math.round((currentProposal.score || 0) * 100)}%</span><button onClick={() => setProposalIndex((value) => (value - 1 + proposals.length) % proposals.length)}>Previous mask</button><button onClick={() => setProposalIndex((value) => (value + 1) % proposals.length)}>Next mask</button><button className="danger" onClick={deleteProposalMask} disabled={busy}><Trash2 size={15} /> Delete this mask</button>{!selectedTrack && <button data-testid="save-single-mask" onClick={saveProposalOnly} disabled={busy}><Check size={15} /> Save mask only</button>}<button className="success" onClick={acceptProposal} disabled={busy}>{selectedTrackRecord?.status === "needs_mask" ? "Apply replacement mask" : selectedTrack ? "Apply correction" : "Accept & track fruit"}</button></div>}
          <div className="harvest-step-controls"><button data-testid="previous-harvest-sample" title="Return to the previous editable sample (Left arrow)" onClick={previousSample} disabled={busy || Number(session.current_time_ms) <= 0}><ChevronLeft size={17} /> Previous 0.25s <kbd>←</kbd></button><button title="Save and advance (Right arrow)" onClick={() => advance()} disabled={busy || Boolean(missingMaskCount)}><ChevronRight size={17} /> Save & Next 0.25s <kbd>→</kbd></button><button title="Auto-step (Space)" className={autoStep ? "active" : ""} onClick={() => setAutoStep((value) => !value)} disabled={busy || !activeTracks.length || Boolean(missingMaskCount)}>{autoStep ? "Pause auto-step" : "Auto-step tracks"} <kbd>Space</kbd></button><button onClick={() => advance({ skip: true })} disabled={busy || !selectedTrack || selectedTrack.status === "needs_mask"}>Skip selected this sample</button><button onClick={() => advance({ end: true })} disabled={busy || !selectedTrack || selectedTrack.status === "needs_mask"}>Save & end selected track</button><button onClick={returnToScan} disabled={busy || Boolean(openTracks.length)}>Return to video scan</button></div>
        </>}
        {progress && <div className="harvest-progress"><LoaderCircle className="spin" size={17} /><span>{progress}</span>{currentJobId && <button onClick={cancelJob}>Cancel</button>}</div>}
      </section>
      <aside className="harvest-tracks"><h3>Fruit masks & tracks</h3><p>Select an item to show its saved mask, correct it, or delete it. Single masks are saved once and never tracked.</p>{session.tracks.length ? session.tracks.map((track) => <button key={track.id} className={`${selectedTrackId === track.id ? "selected " : ""}track-status-${track.status}`} onClick={() => selectHarvestTrack(track)}><span className="track-color" style={{ background: track.color }} /><span><strong>{track.id}</strong><small>{track.capture_mode === "single" ? "single mask" : track.status === "needs_mask" ? "needs replacement mask" : track.status} · {track.saved_count} saved · {track.skipped_count} skipped</small></span></button>) : <div className="empty-track-list">Pause the video and select the first defective fruit.</div>}<div className={`harvest-track-actions${selectedTrackRecord ? "" : " unselected"}`}><button className="danger" data-testid="delete-harvest-mask" onClick={deleteCurrentTrackMask} disabled={busy || !selectedTrackRecord?.mask_url || selectedTrackRecord.status !== "active"}><Trash2 size={15} /> Delete current mask</button><button className="danger" data-testid="delete-harvest-track" onClick={deleteSelectedTrack} disabled={busy || !selectedTrackRecord}><Trash2 size={15} /> Delete selected track</button><small>{!selectedTrackRecord ? "Select an item above to enable these actions." : selectedTrackRecord.status === "needs_mask" ? "Draw and apply a replacement mask, or delete the paused track." : "Deleted masks and tracks are kept in recovery backups."}</small></div><div className="harvest-output-summary"><strong>Output</strong><span>{savedCount} transparent PNGs</span><span>{skippedCount} questionable samples skipped</span><span>Defects remain completely unlabelled</span></div></aside>
    </main>
  </section>;
}

function ProjectLauncher({ onOpen, setMessage }) {
  const [sourceDir, setSourceDir] = useState("");
  const [projectDir, setProjectDir] = useState("dataset/annotations");
  const [projectName, setProjectName] = useState("Image dataset");
  const [annotationMode, setAnnotationMode] = useState("segmentation");
  const [classNames, setClassNames] = useState(DEFAULT_SEGMENTATION_CLASSES);
  const [busy, setBusy] = useState(false);
  const [foregroundDir, setForegroundDir] = useState("");
  const [backgroundDir, setBackgroundDir] = useState("");
  const [outputDir, setOutputDir] = useState("dataset/synthetic");
  const [maxObjectsPerImage, setMaxObjectsPerImage] = useState(6);
  const [totalImages, setTotalImages] = useState(100);
  const [seed, setSeed] = useState(42);
  const [minScale, setMinScale] = useState(10);
  const [maxScale, setMaxScale] = useState(17);
  const [maxRotation, setMaxRotation] = useState(360);
  const [overlapChance, setOverlapChance] = useState(30);
  const [maxOverlap, setMaxOverlap] = useState(20);
  const [generating, setGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(null);
  const [generation, setGeneration] = useState(null);

  const browse = async (setter, title) => {
    try {
      const data = await api("/api/dialog/directory", { method: "POST", body: JSON.stringify({ title }) });
      if (data.path) setter(data.path);
    } catch (error) { setMessage(error.message, "error"); }
  };
  const submit = async (kind) => {
    if (!projectDir.trim() || (kind === "create" && !sourceDir.trim())) {
      setMessage("Select the required folders first.", "error");
      return;
    }
    setBusy(true);
    try {
      const classes = classNames.split(",").map((value) => value.trim()).filter(Boolean);
      if (kind === "create" && !classes.length) throw new Error("Enter at least one class name.");
      const data = await api(`/api/project/${kind}`, {
        method: "POST",
        body: JSON.stringify(kind === "create"
          ? { source_dir: sourceDir.trim(), project_dir: projectDir.trim(), project_name: projectName.trim() || "Image dataset", annotation_mode: annotationMode, class_names: classes }
          : { project_dir: projectDir.trim() })
      });
      onOpen(data.project, data.summary);
    } catch (error) { setMessage(error.message, "error"); }
    finally { setBusy(false); }
  };
  const generate = async () => {
    if (!foregroundDir.trim() || !backgroundDir.trim() || !outputDir.trim()) {
      setMessage("Select foreground, background, and output folders first.", "error");
      return;
    }
    const settingsError = validateSyntheticSettings({
      objectsPerImage: maxObjectsPerImage,
      totalImages,
      seed,
      minScale,
      maxScale,
      maxRotation,
      overlapChance,
      maxOverlap
    });
    if (settingsError) {
      setMessage(settingsError, "error");
      return;
    }
    setGenerating(true);
    setGeneration(null);
    setGenerationProgress({ state: "queued", completed: 0, total: Number(totalImages), percent: 0, stage: "queued", message: "Starting generation…" });
    try {
      const started = await api("/api/synthetic/jobs", {
        method: "POST",
        body: JSON.stringify({
          foreground_dir: foregroundDir.trim(),
          background_dir: backgroundDir.trim(),
          output_dir: outputDir.trim(),
          objects_per_image: Number(maxObjectsPerImage),
          total_images: Number(totalImages),
          seed: Number(seed),
          min_scale_percent: Number(minScale),
          max_scale_percent: Number(maxScale),
          max_rotation_degrees: Number(maxRotation),
          overlap_probability_percent: Number(overlapChance),
          max_overlap_percent: Number(maxOverlap)
        })
      });
      let data = null;
      while (!data) {
        const status = await api(`/api/synthetic/jobs/${started.job_id}`);
        setGenerationProgress(status);
        if (status.state === "complete") data = status.result;
        else if (status.state === "failed") throw new Error(status.error || "Synthetic generation failed.");
        else await new Promise((resolve) => window.setTimeout(resolve, 300));
      }
      setGeneration(data);
      const warningText = data.warnings?.length ? ` ${data.warnings.join(" ")}` : "";
      setMessage(`Generated ${data.counts.generated_images} image(s) with ${data.counts.placed_instances} object(s), ${data.counts.overlapping_instances} overlapping object(s), and ${data.counts.plain_background_images} plain background(s).${warningText}`, data.warnings?.length ? "warning" : "success");
    } catch (error) {
      setGenerationProgress((current) => ({ ...current, state: "failed", stage: "failed", message: error.message }));
      setMessage(error.message, "error");
    }
    finally { setGenerating(false); }
  };
  const openGenerated = async () => {
    if (!generation?.project_dir) return;
    setGenerating(true);
    try {
      const data = await api("/api/project/open", { method: "POST", body: JSON.stringify({ project_dir: generation.project_dir }) });
      onOpen(data.project, data.summary);
    } catch (error) { setMessage(error.message, "error"); }
    finally { setGenerating(false); }
  };
  return (
    <section className="project-launcher">
      <div className="launcher-stack">
        <div className="launcher-card">
          <p className="eyebrow">Local, lossless annotation</p>
          <h1>Data Labeller</h1>
          <p>Create a focused labeling project from an existing image folder. Source images stay untouched and annotations are saved separately.</p>
          <label><span>Main image folder</span><div className="path-row"><input value={sourceDir} onChange={(event) => setSourceDir(event.target.value)} placeholder="D:\Jano\raw_images" /><button onClick={() => browse(setSourceDir, "Select main image folder")}><FolderOpen size={17} /> Browse</button></div></label>
          <label><span>Annotation project folder</span><div className="path-row"><input value={projectDir} onChange={(event) => setProjectDir(event.target.value)} /><button onClick={() => browse(setProjectDir, "Select annotation project folder")}><FolderOpen size={17} /> Browse</button></div></label>
          <label><span>Project name</span><input value={projectName} onChange={(event) => setProjectName(event.target.value)} /></label>
          <div className="mode-picker"><button className={annotationMode === "detection" ? "active" : ""} onClick={() => { setAnnotationMode("detection"); setClassNames("Object"); }}><BoxSelect size={20} /> Object detection<small>Bounding boxes</small></button><button className={annotationMode === "segmentation" ? "active" : ""} onClick={() => { setAnnotationMode("segmentation"); setClassNames(DEFAULT_SEGMENTATION_CLASSES); }}><Brush size={20} /> Instance segmentation<small>SAM + precise masks</small></button><button className={annotationMode === "classification" ? "active" : ""} onClick={() => { setAnnotationMode("classification"); setClassNames("Good, Defective"); }}><Tag size={20} /> Classification<small>One image label</small></button></div>
          <label><span>{annotationMode === "classification" ? "Image classes" : annotationMode === "segmentation" ? "Instance classes" : "Object classes"}, separated by commas</span><input value={classNames} onChange={(event) => setClassNames(event.target.value)} placeholder={annotationMode === "classification" ? "Good, Defective" : annotationMode === "segmentation" ? DEFAULT_SEGMENTATION_CLASSES : "Fruit"} /></label>
          <div className="launcher-actions">
            <button className="primary-action" onClick={() => submit("create")} disabled={busy}>{busy ? <LoaderCircle className="spin" size={17} /> : <Plus size={17} />} Create / rescan project</button>
            <button onClick={() => submit("open")} disabled={busy}><FolderOpen size={17} /> Open project</button>
          </div>
        </div>

        <div className="launcher-card synthetic-card" data-testid="synthetic-generator">
          <p className="eyebrow">Synthetic data</p>
          <div className="launcher-title-row"><div><h2>Generate composite images</h2><p>Create an exact number of annotated images, including controlled fruit overlap, and a ready-to-train YOLO detection dataset.</p></div><Sparkles size={28} /></div>
          <label><span>Foregrounds folder (transparent PNG, WebP, TIFF)</span><div className="path-row"><input value={foregroundDir} onChange={(event) => setForegroundDir(event.target.value)} placeholder="dataset\exports\synthetic_assets\foregrounds" /><button onClick={() => browse(setForegroundDir, "Select foregrounds folder")}><FolderOpen size={17} /> Browse</button></div></label>
          <label><span>Backgrounds folder</span><div className="path-row"><input value={backgroundDir} onChange={(event) => setBackgroundDir(event.target.value)} placeholder="dataset\exports\synthetic_assets\backgrounds" /><button onClick={() => browse(setBackgroundDir, "Select backgrounds folder")}><FolderOpen size={17} /> Browse</button></div></label>
          <label><span>Output parent folder</span><div className="path-row"><input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} /><button onClick={() => browse(setOutputDir, "Select synthetic output parent folder")}><FolderOpen size={17} /> Browse</button></div></label>
          <div className="synthetic-settings">
            <label><span>Maximum fruits per image (1–{SYNTHETIC_LIMITS.maxObjectsPerImage})</span><input type="number" min={SYNTHETIC_LIMITS.minObjectsPerImage} max={SYNTHETIC_LIMITS.maxObjectsPerImage} value={maxObjectsPerImage} onChange={(event) => setMaxObjectsPerImage(event.target.value)} /></label>
            <label><span>Total images to generate</span><input data-testid="synthetic-total-images" type="number" min="2" max="100000" value={totalImages} onChange={(event) => setTotalImages(event.target.value)} /></label>
            <label><span>Random seed</span><input type="number" value={seed} onChange={(event) => setSeed(event.target.value)} /></label>
            <label><span>Minimum scale (%)</span><input type="number" min="0.1" step="0.1" value={minScale} onChange={(event) => setMinScale(event.target.value)} /></label>
            <label><span>Maximum scale (%)</span><input type="number" min="0.1" step="0.1" value={maxScale} onChange={(event) => setMaxScale(event.target.value)} /></label>
            <label><span>Rotation range (0–degrees)</span><input type="number" min="0" max="360" value={maxRotation} onChange={(event) => setMaxRotation(event.target.value)} /></label>
            <label><span>Overlap chance (%)</span><input data-testid="synthetic-overlap-chance" type="number" min="0" max="100" step="1" value={overlapChance} onChange={(event) => setOverlapChance(event.target.value)} /></label>
            <label><span>Maximum overlap per fruit (%)</span><input data-testid="synthetic-max-overlap" type="number" min="1" max="80" step="1" value={maxOverlap} onChange={(event) => setMaxOverlap(event.target.value)} /></label>
          </div>
          <small>The maximum is fruit pasted into one composite, not the number of foreground files. Foregrounds are split into disjoint 80% training and 20% validation pools, then every file in each pool is shuffled and used before reuse. Each image receives a random count, including plain negative backgrounds. By default, 30% of eligible placements deliberately overlap while covering no more than 20% of any fruit.</small>
          <div className="launcher-actions">
            <button onClick={generate} disabled={generating}>{generating ? <LoaderCircle className="spin" size={17} /> : <Sparkles size={17} />} Generate synthetic data</button>
            {generation && <button onClick={openGenerated} disabled={generating}><FolderOpen size={17} /> Open generated project</button>}
          </div>
          {generating && generationProgress && <div className="synthetic-progress" data-testid="synthetic-progress" role="status" aria-live="polite">
            <div className="progress-meta"><strong>{generationProgress.message}</strong><span>{generationProgress.percent}%</span></div>
            <div className="progress-track" role="progressbar" aria-label="Synthetic dataset generation" aria-valuemin="0" aria-valuemax="100" aria-valuenow={generationProgress.percent}>
              <span className="progress-fill" style={{ width: `${generationProgress.percent}%` }} />
            </div>
            <small>{generationProgress.completed} of {generationProgress.total} composite images · {generationProgress.stage}</small>
          </div>}
          {generation && <div className="generation-result"><Check size={18} /><div><strong>{generation.counts.generated_images} images · {generation.counts.placed_instances} boxes and masks · {generation.counts.overlapping_instances} overlapping objects · {generation.counts.plain_background_images} plain backgrounds{generation.counts.capacity_limited_images ? ` · ${generation.counts.capacity_limited_images} capacity-limited images` : ""}</strong><small>YOLO: {generation.detection_yaml}</small></div></div>}
        </div>
      </div>
    </section>
  );
}

const blurTypes = [
  ["horizontal", "Horizontal conveyor"],
  ["diagonal", "Slight diagonal"],
  ["tangential", "Tangential / curved"],
  ["spatially_varying", "Spatially varying"]
];

function SyntheticBlurDataset({ setMessage }) {
  const [sourceDir, setSourceDir] = useState("");
  const [outputDir, setOutputDir] = useState("dataset/deblur_dataset");
  const [size, setSize] = useState(256);
  const [variants, setVariants] = useState(4);
  const [seed, setSeed] = useState(42);
  const [splits, setSplits] = useState({ train: 70, val: 20, test: 10 });
  const [enabled, setEnabled] = useState(Object.fromEntries(blurTypes.map(([name]) => [name, true])));
  const [probabilities, setProbabilities] = useState({ horizontal: 0.35, diagonal: 0.25, tangential: 0.2, spatially_varying: 0.2 });
  const [moderateProbability, setModerateProbability] = useState(0.65);
  const [moderate, setModerate] = useState({ path_length: [5, 13], curvature: [0.02, 0.16], exposure: [-0.025, 0.025], brightness: [-0.018, 0.018], contrast: [0.97, 1.03], noise_sigma: [0.001, 0.005] });
  const [severe, setSevere] = useState({ path_length: [15, 31], curvature: [0.12, 0.42], exposure: [-0.04, 0.04], brightness: [-0.025, 0.025], contrast: [0.95, 1.05], noise_sigma: [0.002, 0.009] });
  const [resume, setResume] = useState(false);
  const [preview, setPreview] = useState(null);
  const [progress, setProgress] = useState(null);
  const [busy, setBusy] = useState(false);
  const jobIdRef = useRef(null);

  const browse = async (setter, title) => {
    try {
      const data = await api("/api/dialog/directory", { method: "POST", body: JSON.stringify({ title }) });
      if (data.path) setter(data.path);
    } catch (error) { setMessage(error.message, "error"); }
  };
  const config = () => ({
    size: Number(size), variants: Number(variants), seed: Number(seed),
    ratios: { train: Number(splits.train) / 100, val: Number(splits.val) / 100, test: Number(splits.test) / 100 },
    enabled_blur_types: blurTypes.filter(([name]) => enabled[name]).map(([name]) => name),
    blur_probabilities: Object.fromEntries(Object.entries(probabilities).map(([name, value]) => [name, Number(value)])),
    moderate_probability: Number(moderateProbability), moderate, severe
  });
  const request = (extra = {}) => ({ source_dir: sourceDir.trim(), output_dir: outputDir.trim(), config: config(), resume, ...extra });
  const updateRange = (setter, values, key, index, value) => setter({ ...values, [key]: values[key].map((item, itemIndex) => itemIndex === index ? Number(value) : item) });
  const runPreview = async () => {
    if (!sourceDir.trim()) return setMessage("Select a sharp-image source folder first.", "error");
    setBusy(true);
    try {
      const data = await api("/api/deblur/preview", { method: "POST", body: JSON.stringify(request()) });
      setPreview(data); setMessage("Synthetic blur preview is ready.", "success");
    } catch (error) { setMessage(`Preview failed: ${error.message}`, "error"); }
    finally { setBusy(false); }
  };
  const generate = async () => {
    if (!sourceDir.trim() || !outputDir.trim()) return setMessage("Select source and output folders first.", "error");
    setBusy(true); setProgress({ state: "queued", completed: 0, total: 0, percent: 0, message: "Starting paired generation." });
    try {
      const started = await api("/api/deblur/jobs", { method: "POST", body: JSON.stringify(request()) });
      jobIdRef.current = started.job_id;
      while (jobIdRef.current === started.job_id) {
        const status = await api(`/api/deblur/jobs/${started.job_id}`);
        setProgress(status);
        if (["complete", "complete_with_errors", "cancelled"].includes(status.state)) {
          const failures = status.result?.failure_count || 0;
          setMessage(`${status.result?.generated_pairs || 0} paired image(s) written.${failures ? ` ${failures} file failure(s) were recorded.` : ""}`, failures ? "warning" : "success");
          break;
        }
        if (status.state === "failed") throw new Error(status.error || "Generation failed.");
        await new Promise((resolve) => window.setTimeout(resolve, 300));
      }
    } catch (error) { setMessage(`Generation failed: ${error.message}`, "error"); }
    finally { jobIdRef.current = null; setBusy(false); }
  };
  const cancel = async () => {
    if (!jobIdRef.current) return;
    try { await api(`/api/deblur/jobs/${jobIdRef.current}/cancel`, { method: "POST", body: "{}" }); }
    catch (error) { setMessage(`Cancel failed: ${error.message}`, "error"); }
  };
  const validate = async () => {
    if (!outputDir.trim()) return;
    setBusy(true);
    try {
      const data = await api("/api/deblur/validate", { method: "POST", body: JSON.stringify(request()) });
      setMessage(`Dataset validation ${data.valid ? "passed" : "failed"}: ${data.pair_count || 0} pair(s), ${data.errors.length} error(s), ${data.warnings.length} warning(s).`, data.valid ? "success" : "error");
    } catch (error) { setMessage(error.message, "error"); }
    finally { setBusy(false); }
  };
  const RangeRow = ({ title, values, setter }) => <div className="range-table">
    <strong>{title}</strong>
    {[ ["path_length", "Path length", 1], ["curvature", "Curvature", 0.01], ["exposure", "Exposure EV", 0.005], ["brightness", "Brightness", 0.005], ["contrast", "Contrast", 0.01], ["noise_sigma", "Noise sigma", 0.001] ].map(([key, label, step]) => <React.Fragment key={key}>
      <span>{label}</span><input type="number" step={step} value={values[key][0]} onChange={(event) => updateRange(setter, values, key, 0, event.target.value)} /><input type="number" step={step} value={values[key][1]} onChange={(event) => updateRange(setter, values, key, 1, event.target.value)} />
    </React.Fragment>)}
  </div>;
  return <section className="deblur-page">
    <div className="deblur-heading"><div><p className="eyebrow">Paired restoration data</p><h1>Synthetic Blur Dataset</h1><p>Generate aligned sharp targets and physically motivated motion-blurred inputs. Source identities and duplicates are grouped before splitting.</p></div><Sparkles size={34} /></div>
    <div className="deblur-grid">
      <div className="launcher-card deblur-config">
        <h2>Dataset</h2>
        <label><span>Sharp-image source folder</span><div className="path-row"><input value={sourceDir} onChange={(event) => setSourceDir(event.target.value)} /><button onClick={() => browse(setSourceDir, "Select sharp-image source folder")}><FolderOpen size={17} /> Browse</button></div></label>
        <label><span>Output folder</span><div className="path-row"><input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} /><button onClick={() => browse(setOutputDir, "Select paired dataset output folder")}><FolderOpen size={17} /> Browse</button></div></label>
        <div className="synthetic-settings">
          <label><span>Image size</span><input type="number" min="32" value={size} onChange={(event) => setSize(event.target.value)} /></label>
          <label><span>Variants / source</span><input type="number" min="1" value={variants} onChange={(event) => setVariants(event.target.value)} /></label>
          <label><span>Seed</span><input type="number" value={seed} onChange={(event) => setSeed(event.target.value)} /></label>
          {Object.entries(splits).map(([name, value]) => <label key={name}><span>{name === "val" ? "Validation" : name[0].toUpperCase() + name.slice(1)} (%)</span><input type="number" min="0" max="100" value={value} onChange={(event) => setSplits({ ...splits, [name]: event.target.value })} /></label>)}
        </div>
        <h2>Blur mix</h2>
        <div className="blur-type-grid">{blurTypes.map(([name, label]) => <div key={name} className="blur-type-row"><label className="check-row"><input type="checkbox" checked={enabled[name]} onChange={(event) => setEnabled({ ...enabled, [name]: event.target.checked })} /> {label}</label><input aria-label={`${label} probability`} type="number" min="0" step="0.05" value={probabilities[name]} onChange={(event) => setProbabilities({ ...probabilities, [name]: event.target.value })} /></div>)}</div>
        <label><span>Moderate blur probability</span><input type="number" min="0" max="1" step="0.05" value={moderateProbability} onChange={(event) => setModerateProbability(event.target.value)} /></label>
        <div className="range-pair"><RangeRow title="Moderate ranges" values={moderate} setter={setModerate} /><RangeRow title="Severe ranges" values={severe} setter={setSevere} /></div>
        <label className="check-row"><input type="checkbox" checked={resume} onChange={(event) => setResume(event.target.checked)} /> Resume and validate an interrupted generation</label>
        <div className="launcher-actions"><button onClick={runPreview} disabled={busy}><Eye size={17} /> Preview</button><button onClick={generate} disabled={busy}><Sparkles size={17} /> Generate</button><button onClick={cancel} disabled={!jobIdRef.current}><X size={17} /> Cancel</button><button onClick={validate} disabled={busy}><ListChecks size={17} /> Validate</button></div>
        {progress && <div className="synthetic-progress"><div className="progress-meta"><strong>{progress.message}</strong><span>{progress.percent || 0}%</span></div><div className="progress-track"><span className="progress-fill" style={{ width: `${progress.percent || 0}%` }} /></div><small>{progress.completed || 0} of {progress.total || 0} · {progress.stage || progress.state}</small>{progress.error && <small className="error-text">{progress.error}</small>}</div>}
      </div>
      <div className="launcher-card deblur-preview"><h2>Preview</h2>
        {!preview && <div className="preview-placeholder">Choose a source folder and generate a deterministic preview.</div>}
        {preview && <><div className="preview-images"><figure><img src={preview.sharp_image} alt="Sharp target preview" /><figcaption>Sharp · Laplacian {preview.sharp_laplacian_variance.toFixed(1)}</figcaption></figure><figure><img src={preview.blurred_image} alt="Blurred input preview" /><figcaption>Blurred · Laplacian {preview.blurred_laplacian_variance.toFixed(1)}</figcaption></figure><figure><img src={preview.difference_image} alt="Difference preview" /><figcaption>Amplified difference</figcaption></figure></div>
          <div className="preview-summary"><strong>{preview.total_pairs} total pairs</strong><span>Train {preview.split_counts.train} · Val {preview.split_counts.val} · Test {preview.split_counts.test}</span><span>Estimated {Math.max(0.1, preview.estimated_storage_bytes / 1048576).toFixed(1)} MiB</span><span>{preview.blur_parameters.blur_type} · {preview.blur_parameters.severity} · path {preview.blur_parameters.path_length}px · angle {preview.blur_parameters.angle_degrees.toFixed(1)}°</span><small>{preview.source_path}</small></div>
        </>}
      </div>
    </div>
  </section>;
}

function TopBar({ project, currentIndex, imageCount, saveState, navigationBusy, page, onPage, onChangeProject, onSave, onValidate, onSplit, onDataset, onHelp, onImages, onPrevious, onNext, onProperties, onHarvest, onBackToHarvest }) {
  const [toolsOpen, setToolsOpen] = useState(false);
  const opened = Boolean(project);
  const mode = project?.annotation_mode || "mixed";
  return (
    <header className="topbar">
      <div className="brand"><strong>{project?.project_name || "Data Labeller"}</strong>{opened && <><span className="task-badge">{mode}</span><span className={`save-state ${saveState}`}>{saveState === "saving" ? "Saving…" : saveState === "dirty" ? "Unsaved" : saveState === "error" ? "Save error" : "Saved"}</span></>}</div>
      {opened && page === "annotation" && <nav className="workflow-steps" aria-label="Dataset workflow"><span className="complete"><Check size={14} /> Import</span><span className="active">2 Label</span><button onClick={onDataset}>3 Review & export</button></nav>}
      <div className="top-actions">
        {opened && page === "annotation" && <>
          <IconButton title="Browse all frames" onClick={onImages}><FolderOpen size={18} /></IconButton>
          <nav className="frame-navigation" aria-label="Frame navigation">
            <button className="frame-nav-button" data-testid="previous-frame" title="Save and open previous frame (Left arrow)" onClick={onPrevious} disabled={navigationBusy || currentIndex <= 0}><ChevronLeft size={17} /><span>Previous</span></button>
            <button className="image-position" onClick={onImages} title="Browse all frames">{imageCount ? `Frame ${currentIndex + 1} of ${imageCount}` : "No frames"}</button>
            <button className="frame-nav-button" data-testid="next-frame" title="Save and open next frame (Right arrow)" onClick={onNext} disabled={navigationBusy || currentIndex >= imageCount - 1}><span>{navigationBusy ? "Saving…" : "Next"}</span><ChevronRight size={17} /></button>
          </nav>
          <IconButton title="Annotation properties" onClick={onProperties}><ListChecks size={18} /></IconButton><IconButton title="Save now" onClick={onSave}><Save size={18} /></IconButton>{project?.source_kind === "harvested_foregrounds" && <button className="back-to-harvest-button" data-testid="back-to-harvesting" onClick={onBackToHarvest} disabled={navigationBusy}><ScanSearch size={17} /> Back to harvesting</button>}<button className="review-button" onClick={onDataset}><Download size={17} /> Review & export</button>
        </>}
        <div className="tools-menu"><button aria-expanded={toolsOpen} onClick={() => setToolsOpen((value) => !value)}>Tools <ChevronRight className={toolsOpen ? "rotate" : ""} size={16} /></button>{toolsOpen && <div className="tools-popover" role="menu">
          {opened && <><button onClick={() => { setToolsOpen(false); onValidate(); }}>Validate project</button><button onClick={() => { setToolsOpen(false); onSplit(); }}>Split settings</button><button onClick={() => { setToolsOpen(false); onDataset(); }}>Classes and export</button><button onClick={() => { setToolsOpen(false); onChangeProject(); }}>Change project</button></>}
          <button onClick={() => { setToolsOpen(false); onHarvest(); }}>{project?.source_kind === "harvested_foregrounds" ? "Back to harvesting" : "Harvest defective fruit from video"}</button>
          <button onClick={() => { setToolsOpen(false); onPage("advanced"); }}>Image & synthetic tools</button>
          <button onClick={() => { setToolsOpen(false); onPage("deblur"); }}>Synthetic blur dataset</button>
          <button onClick={() => { setToolsOpen(false); onHelp(); }}>Keyboard shortcuts</button>
          {page !== "annotation" && <button onClick={() => { setToolsOpen(false); onPage("annotation"); }}>Back to labeling</button>}
        </div>}</div>
      </div>
    </header>
  );
}

function ImageSidebar({ images, currentImage, currentIndex, navigationBusy, onSelect, onPrev, onNext, onBulkNoInstances, onClose }) {
  const rowHeight = 76;
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const [split, setSplit] = useState("all");
  const [warningOnly, setWarningOnly] = useState(false);
  const [scrollTop, setScrollTop] = useState(0);
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [bulkBusy, setBulkBusy] = useState(false);
  const listRef = useRef(null);
  const lastSelectedIdRef = useRef(null);
  const positionedCurrentRef = useRef(false);
  const filtered = useMemo(() => images.filter((image) => {
    const term = search.trim().toLowerCase();
    return (!term || `${image.file_name} ${image.relative_path} ${image.fruit_id} ${image.view_id}`.toLowerCase().includes(term))
      && (status === "all" || image.status === status)
      && (split === "all" || image.split === split)
      && (!warningOnly || image.has_warning);
  }), [images, search, split, status, warningOnly]);
  useEffect(() => setSelectedIds((current) => retainExistingSelections(current, images)), [images]);
  useLayoutEffect(() => {
    const list = listRef.current;
    if (!list || positionedCurrentRef.current || !currentImage) return;
    const activeIndex = filtered.findIndex((image) => String(image.id) === String(currentImage.id));
    if (activeIndex < 0) return;
    const centeredTop = activeIndex * rowHeight - (list.clientHeight - rowHeight) / 2;
    const maximumTop = Math.max(0, filtered.length * rowHeight - list.clientHeight);
    const nextScrollTop = Math.max(0, Math.min(maximumTop, centeredTop));
    list.scrollTop = nextScrollTop;
    setScrollTop(nextScrollTop);
    positionedCurrentRef.current = true;
  }, [currentImage, filtered]);
  const orderedIds = useMemo(() => filtered.map((image) => String(image.id)), [filtered]);
  const selectFiltered = () => setSelectedIds((current) => new Set([...current, ...orderedIds]));
  const toggleSelection = (event, imageId) => {
    const checked = event.currentTarget.checked;
    setSelectedIds((current) => updateImageSelection(
      current,
      imageId,
      checked,
      event.shiftKey,
      lastSelectedIdRef.current,
      orderedIds,
    ));
    lastSelectedIdRef.current = String(imageId);
  };
  const bulkNoInstances = async () => {
    const ids = [...selectedIds];
    if (!ids.length || !window.confirm(`Mark ${ids.length} selected image${ids.length === 1 ? "" : "s"} as No instances? Existing masks and boxes will be removed and kept in recoverable backups.`)) return;
    setBulkBusy(true);
    try {
      const completed = await onBulkNoInstances(ids);
      if (completed) {
        setSelectedIds(new Set());
        lastSelectedIdRef.current = null;
      }
    } finally {
      setBulkBusy(false);
    }
  };
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - 5);
  const visible = filtered.slice(start, start + 28);
  return (
    <aside className="sidebar left drawer" aria-label="Image browser">
      <div className="drawer-heading"><div><strong>Images</strong><small>{filtered.length} of {images.length}</small></div><IconButton title="Close image browser" onClick={onClose}><X size={17} /></IconButton></div>
      <div className="panel-head"><span>{currentImage?.source_video_name || "Dataset"}</span><span>{currentImage?.status || "unlabelled"}</span></div>
      <input className="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search filename, fruit or view…" />
      <div className="filter-grid">
        <select value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">All statuses</option><option value="unlabelled">Unlabelled</option><option value="draft">Draft</option><option value="review">Review</option><option value="approved">Approved</option><option value="excluded">Excluded</option></select>
        <select value={split} onChange={(event) => setSplit(event.target.value)}><option value="all">All splits</option><option value="train">Train</option><option value="val">Validation</option><option value="test">Test</option></select>
        <label className="check-row"><input type="checkbox" checked={warningOnly} onChange={(event) => setWarningOnly(event.target.checked)} /> Warnings</label>
      </div>
      <div className="bulk-select-bar">
        <div><strong>{selectedIds.size} selected</strong><small>Checkboxes support Shift-click ranges</small></div>
        <div className="bulk-select-actions">
          <button className="compact" onClick={selectFiltered} disabled={!filtered.length || bulkBusy}>Select filtered</button>
          <button className="compact" onClick={() => setSelectedIds(new Set())} disabled={!selectedIds.size || bulkBusy}>Clear</button>
        </div>
        <button className="danger bulk-negative" onClick={bulkNoInstances} disabled={!selectedIds.size || bulkBusy || navigationBusy}>
          {bulkBusy ? <><LoaderCircle className="spin" size={16} /> Updating…</> : <><X size={16} /> No instances</>}
        </button>
      </div>
      <div className="nav-row"><button title="Previous image (Left arrow)" onClick={onPrev} disabled={navigationBusy || currentIndex <= 0}><ChevronLeft size={17} /> Prev <kbd>←</kbd></button><button data-testid="next-image" title="Next image (Right arrow)" onClick={onNext} disabled={navigationBusy || currentIndex >= images.length - 1}>{navigationBusy ? <><LoaderCircle className="spin" size={17} /> Saving…</> : <>Next <kbd>→</kbd> <ChevronRight size={17} /></>}</button></div>
      <div ref={listRef} className="virtual-list" onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}>
        <div style={{ height: filtered.length * rowHeight, position: "relative" }}>
          {visible.map((image, offset) => (
            <div key={image.id} className={`${String(image.id) === String(currentImage?.id) ? "image-row active" : "image-row"}${selectedIds.has(String(image.id)) ? " selected" : ""}`} style={{ position: "absolute", top: (start + offset) * rowHeight, height: rowHeight - 4 }}>
              <label className="image-select" title="Select image"><input type="checkbox" checked={selectedIds.has(String(image.id))} readOnly onClick={(event) => toggleSelection(event, image.id)} aria-label={`Select ${image.file_name}`} /></label>
              <button className="image-row-main" onClick={() => onSelect(image.id)}>
                <img src={`/api/thumbnails/${image.id}?v=${image.revision || 0}`} loading="lazy" alt="" />
                <span><strong>{image.file_name}</strong><small>{image.status} · {image.split || "unsplit"}{image.sample_time_ms != null ? ` · ${(image.sample_time_ms / 1000).toFixed(1)}s` : ""}</small><small>{image.classification_class_name || image.source_video_name || image.fruit_id || "image source"}</small></span>
              </button>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

function drawColoredMask(target, overlay, mask, transform, image, color, opacity) {
  const width = overlay.width;
  const height = overlay.height;
  const context = overlay.getContext("2d");
  context.clearRect(0, 0, width, height);
  context.drawImage(mask, transform.offsetX, transform.offsetY, image.width * transform.scale, image.height * transform.scale);
  context.globalCompositeOperation = "source-in";
  context.fillStyle = color;
  context.fillRect(0, 0, width, height);
  context.globalCompositeOperation = "source-over";
  target.save();
  target.globalAlpha = opacity;
  target.drawImage(overlay, 0, 0);
  target.restore();
}

function AnnotationCanvas({
  image,
  instances,
  masksRef,
  maskRevision,
  selectedId,
  setSelectedId,
  tool,
  setTool,
  brushSize,
  setBrushSize,
  opacity,
  setOpacity,
  showMasks,
  setShowMasks,
  prompts,
  setPrompts,
  proposalCanvas,
  boxProposals,
  pushHistory,
  markDirty,
  undo,
  redo,
  clearSelectedMask,
  deleteSelected,
  registerFit,
  canUndo,
  canRedo,
  annotationMode = "mixed",
  onEditImage,
  onOpenDetectionAssist,
  onOpenSegmentationAssist,
  onCreateBox,
  onPreviewBox,
  onCreateMask
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const overlayRef = useRef(document.createElement("canvas"));
  const viewRef = useRef({ scale: 1, offsetX: 0, offsetY: 0 });
  const fitScaleRef = useRef(1);
  const dragRef = useRef(null);
  const polygonRef = useRef([]);
  const cursorRef = useRef(null);
  const spaceRef = useRef(false);
  const [redrawTick, requestDraw] = useState(0);
  const [zoomPercent, setZoomPercent] = useState(100);
  const selected = instances.find((item) => String(item.id) === String(selectedId));
  const imageWidth = Number(image.width);
  const imageHeight = Number(image.height);
  const hasDetectionTools = annotationMode === "detection" || annotationMode === "mixed";
  const hasSegmentationTools = annotationMode === "segmentation" || annotationMode === "mixed";

  const canvasPoint = useCallback((event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }, []);
  const imagePoint = useCallback((event) => clampImagePoint(screenToImage(canvasPoint(event), viewRef.current), Number(image.width), Number(image.height)), [canvasPoint, image]);
  const fit = useCallback(() => {
    if (!image || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const transform = fitImageTransform(rect.width, rect.height, imageWidth, imageHeight);
    fitScaleRef.current = transform.scale;
    viewRef.current = transform;
    setZoomPercent(100);
    requestDraw((value) => value + 1);
  }, [imageHeight, imageWidth]);
  useEffect(() => { registerFit(fit); }, [fit, registerFit]);

  useEffect(() => {
    const loaded = new Image();
    imageRef.current = null;
    loaded.onload = () => { imageRef.current = loaded; fit(); };
    loaded.src = `/api/image-file/${image.id}?v=${image.revision || 0}`;
    return () => { imageRef.current = null; };
  }, [fit, image.id, image.revision]);
  useEffect(() => {
    const resize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      overlayRef.current.width = Math.max(1, Math.floor(rect.width));
      overlayRef.current.height = Math.max(1, Math.floor(rect.height));
      requestDraw((value) => value + 1);
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);
  useEffect(() => {
    const down = (event) => { if (event.code === "Space" && !shouldIgnoreShortcut(event.target)) { spaceRef.current = true; event.preventDefault(); } };
    const up = (event) => { if (event.code === "Space") spaceRef.current = false; };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, rect.width, rect.height);
    context.fillStyle = "#090c10";
    context.fillRect(0, 0, rect.width, rect.height);
    if (!imageRef.current) return;
    const view = viewRef.current;
    context.drawImage(imageRef.current, view.offsetX, view.offsetY, image.width * view.scale, image.height * view.scale);
    if (showMasks) {
      for (const instance of instances) {
        if (!instance.visible) continue;
        const mask = masksRef.current.get(String(instance.id));
        if (mask) drawColoredMask(context, overlayRef.current, mask, view, image, instance.color || "#5aa7d8", String(instance.id) === String(selectedId) ? Math.min(0.85, opacity + 0.2) : opacity);
      }
    }
    for (const instance of instances.filter((item) => item.visible && isFruit(item) && item.bbox)) {
      const [x, y] = imageToScreen([instance.bbox[0], instance.bbox[1]], view);
      context.strokeStyle = instance.color || "#42b883";
      context.lineWidth = String(instance.id) === String(selectedId) ? 3 : 2;
      context.setLineDash(String(instance.id) === String(selectedId) ? [] : [7, 4]);
      context.strokeRect(x, y, instance.bbox[2] * view.scale, instance.bbox[3] * view.scale);
      context.setLineDash([]);
      if (String(instance.id) === String(selectedId) && tool === "select" && !instance.locked) {
        context.fillStyle = "#ffffff";
        context.strokeStyle = instance.color || "#42b883";
        context.lineWidth = 2;
        const handleSize = 16;
        for (const [, handleImageX, handleImageY] of boxHandlePoints(instance.bbox)) {
          const [handleX, handleY] = imageToScreen([handleImageX, handleImageY], view);
          context.fillRect(handleX - handleSize / 2, handleY - handleSize / 2, handleSize, handleSize);
          context.strokeRect(handleX - handleSize / 2, handleY - handleSize / 2, handleSize, handleSize);
        }
      }
    }
    for (const proposal of boxProposals) {
      const [x, y] = imageToScreen([proposal.bbox[0], proposal.bbox[1]], view);
      const width = proposal.bbox[2] * view.scale;
      const height = proposal.bbox[3] * view.scale;
      context.strokeStyle = "#ffd166";
      context.lineWidth = 2.5;
      context.setLineDash([9, 5]);
      context.strokeRect(x, y, width, height);
      context.setLineDash([]);
      const label = `${proposal.label || "object"} ${Math.round(Number(proposal.score || 0) * 100)}%`;
      context.font = "600 12px system-ui, sans-serif";
      const labelWidth = context.measureText(label).width + 12;
      const labelY = Math.max(2, y - 24);
      context.fillStyle = "#ffd166";
      context.fillRect(x, labelY, labelWidth, 22);
      context.fillStyle = "#17120a";
      context.fillText(label, x + 6, labelY + 15);
    }
    if (dragRef.current?.type === "draw_box") {
      const box = dragRef.current.box;
      const [x, y] = imageToScreen([box[0], box[1]], view);
      context.strokeStyle = "#ffffff"; context.lineWidth = 2; context.setLineDash([7, 4]);
      context.strokeRect(x, y, box[2] * view.scale, box[3] * view.scale); context.setLineDash([]);
    }
    if (proposalCanvas) drawColoredMask(context, overlayRef.current, proposalCanvas, view, image, "#52d6ff", 0.48);
    for (const point of prompts.points) {
      const [x, y] = imageToScreen([point.x, point.y], view);
      context.beginPath();
      context.fillStyle = point.label ? "#39d98a" : "#ff5d6c";
      context.strokeStyle = "#ffffff";
      context.lineWidth = 2;
      context.arc(x, y, 6, 0, Math.PI * 2);
      context.fill(); context.stroke();
    }
    const activeBox = dragRef.current?.type === "sam_box" ? dragRef.current.box : prompts.box;
    if (activeBox) {
      const [x1, y1] = imageToScreen([activeBox[0], activeBox[1]], view);
      const [x2, y2] = imageToScreen([activeBox[2], activeBox[3]], view);
      context.strokeStyle = "#52d6ff"; context.lineWidth = 2; context.setLineDash([6, 4]);
      context.strokeRect(x1, y1, x2 - x1, y2 - y1); context.setLineDash([]);
    }
    if (cursorRef.current && ["add", "erase"].includes(tool)) {
      context.beginPath(); context.strokeStyle = tool === "add" ? "#ffffff" : "#ff6675"; context.lineWidth = 1.5;
      context.arc(cursorRef.current[0], cursorRef.current[1], brushSize * view.scale / 2, 0, Math.PI * 2); context.stroke();
    }
    if (polygonRef.current.length) {
      context.beginPath();
      polygonRef.current.forEach((point, index) => { const [x, y] = imageToScreen(point, view); if (index) context.lineTo(x, y); else context.moveTo(x, y); });
      context.strokeStyle = "#ffffff"; context.lineWidth = 2; context.stroke();
    }
  }, [boxProposals, brushSize, image, instances, maskRevision, opacity, prompts, proposalCanvas, redrawTick, selectedId, showMasks, tool]);

  const paintPoints = (points, mode) => {
    const mask = masksRef.current.get(String(selectedId));
    if (!mask || selected?.locked) return;
    const context = mask.getContext("2d");
    context.save();
    context.globalCompositeOperation = mode === "erase" ? "destination-out" : "source-over";
    context.fillStyle = "#ffffff";
    for (const point of points) {
      context.beginPath(); context.arc(point[0], point[1], brushSize / 2, 0, Math.PI * 2); context.fill();
    }
    context.restore();
    requestDraw((value) => value + 1);
  };
  const selectableAtPoint = (pixel) => instances.filter((item) => item.visible).reverse().find((item) => {
    if (isFruit(item) && item.bbox) return boxContainsPoint(item.bbox, pixel);
    const mask = masksRef.current.get(String(item.id));
    return mask?.getContext("2d", { willReadFrequently: true }).getImageData(Math.round(pixel[0]), Math.round(pixel[1]), 1, 1).data[3] > 0;
  });
  const updateSelectionCursor = (pixel) => {
    const selectedBox = selected?.visible && isFruit(selected) && selected.bbox && !selected.locked ? selected : null;
    const handle = selectedBox ? hitBoxHandle(selectedBox.bbox, pixel, viewRef.current.scale) : null;
    if (handle) canvasRef.current.style.cursor = BOX_RESIZE_CURSORS[handle];
    else if ((selectedBox && boxContainsPoint(selectedBox.bbox, pixel)) || instances.some((item) => item.visible && isFruit(item) && item.bbox && boxContainsPoint(item.bbox, pixel))) canvasRef.current.style.cursor = "move";
    else canvasRef.current.style.cursor = "default";
  };
  const onPointerDown = (event) => {
    if (event.button !== 0 && event.button !== 1) return;
    canvasRef.current.setPointerCapture(event.pointerId);
    const screen = canvasPoint(event);
    if (tool === "pan" || spaceRef.current || event.button === 1) {
      dragRef.current = { type: "pan", screen, offsetX: viewRef.current.offsetX, offsetY: viewRef.current.offsetY };
    } else if (tool === "draw_box") {
      const point = imagePoint(event);
      dragRef.current = { type: "draw_box", start: point, box: [point[0], point[1], 0, 0] };
    } else if (["add", "erase"].includes(tool)) {
      if (!selectedId || selected?.locked) return;
      pushHistory();
      const point = imagePoint(event);
      dragRef.current = { type: "brush", mode: tool, samples: [point] };
      paintPoints([point], tool);
    } else if (tool === "sam_positive" || tool === "sam_negative") {
      const point = imagePoint(event);
      setPrompts((current) => ({ ...current, points: [...current.points, { x: point[0], y: point[1], label: tool === "sam_positive" ? 1 : 0 }] }));
    } else if (tool === "sam_box") {
      const point = imagePoint(event);
      dragRef.current = { type: "sam_box", start: point, box: [point[0], point[1], point[0], point[1]] };
    } else if (tool === "polygon") {
      const mask = masksRef.current.get(String(selectedId));
      if (!mask || selected?.locked) return;
      const point = imagePoint(event);
      if (event.detail >= 2 && polygonRef.current.length >= 2) {
        pushHistory();
        const points = [...polygonRef.current, point];
        const context = mask.getContext("2d");
        context.fillStyle = "#ffffff"; context.beginPath();
        points.forEach(([x, y], index) => { if (index) context.lineTo(x, y); else context.moveTo(x, y); });
        context.closePath(); context.fill(); polygonRef.current = []; markDirty(true);
      } else polygonRef.current.push(point);
    } else if (tool === "select") {
      const visible = instances.filter((item) => item.visible).reverse();
      const pixel = imagePoint(event);
      const selectedBox = visible.find((item) => String(item.id) === String(selectedId) && isFruit(item) && item.bbox && !item.locked);
      const selectedHandle = selectedBox ? hitBoxHandle(selectedBox.bbox, pixel, viewRef.current.scale) : null;
      const hit = selectedHandle ? selectedBox : selectableAtPoint(pixel);
      setSelectedId(hit?.id || null);
      if (hit && isFruit(hit) && hit.bbox && !hit.locked) {
        const [x, y, width, height] = hit.bbox.map(Number);
        const handle = selectedHandle || (String(hit.id) === String(selectedId) ? hitBoxHandle(hit.bbox, pixel, viewRef.current.scale) : null);
        dragRef.current = { type: handle ? "resize_box" : "move_box", id: hit.id, handle, start: pixel, original: [x, y, width, height], changed: false };
        pushHistory();
      }
    }
    requestDraw((value) => value + 1);
  };
  const onPointerMove = (event) => {
    const screen = canvasPoint(event);
    cursorRef.current = screen;
    const drag = dragRef.current;
    if (!drag) {
      if (tool === "select") updateSelectionCursor(imagePoint(event));
      requestDraw((value) => value + 1);
      return;
    }
    if (drag.type === "pan") {
      viewRef.current = { ...viewRef.current, offsetX: drag.offsetX + screen[0] - drag.screen[0], offsetY: drag.offsetY + screen[1] - drag.screen[1] };
    } else if (drag.type === "brush") {
      const point = imagePoint(event);
      const previous = drag.samples[drag.samples.length - 1];
      const interpolated = interpolateStroke([previous, point], Math.max(0.5, brushSize * 0.2)).slice(1);
      drag.samples.push(point);
      paintPoints(interpolated, drag.mode);
    } else if (drag.type === "draw_box") {
      const point = imagePoint(event);
      drag.box = [Math.min(drag.start[0], point[0]), Math.min(drag.start[1], point[1]), Math.abs(point[0] - drag.start[0]), Math.abs(point[1] - drag.start[1])];
    } else if (drag.type === "move_box") {
      const point = imagePoint(event);
      if (!drag.changed && Math.hypot(point[0] - drag.start[0], point[1] - drag.start[1]) >= 0.5) {
        drag.changed = true;
        markDirty(true);
      }
      onPreviewBox(drag.id, moveBox(drag.original, [point[0] - drag.start[0], point[1] - drag.start[1]], image.width, image.height));
    } else if (drag.type === "resize_box") {
      const point = imagePoint(event);
      if (!drag.changed && Math.hypot(point[0] - drag.start[0], point[1] - drag.start[1]) >= 0.5) {
        drag.changed = true;
        markDirty(true);
      }
      onPreviewBox(drag.id, resizeBox(drag.original, drag.handle, point, image.width, image.height));
    } else if (drag.type === "sam_box") {
      const point = imagePoint(event);
      drag.box = [drag.start[0], drag.start[1], point[0], point[1]];
    }
    requestDraw((value) => value + 1);
  };
  const onPointerUp = (event) => {
    const drag = dragRef.current;
    if (drag?.type === "brush") markDirty(true);
    if (drag?.type === "draw_box" && drag.box[2] >= 2 && drag.box[3] >= 2) onCreateBox(drag.box);
    if (drag?.type === "sam_box") {
      const box = [Math.min(drag.box[0], drag.box[2]), Math.min(drag.box[1], drag.box[3]), Math.max(drag.box[0], drag.box[2]), Math.max(drag.box[1], drag.box[3])];
      if (box[2] - box[0] > 2 && box[3] - box[1] > 2) setPrompts((current) => ({ ...current, box }));
    }
    dragRef.current = null;
    try { canvasRef.current.releasePointerCapture(event.pointerId); } catch { /* pointer already released */ }
    if (tool === "select") updateSelectionCursor(imagePoint(event));
    requestDraw((value) => value + 1);
  };
  const zoomBy = (multiplier, screenPoint = null) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const anchor = screenPoint || [rect.width / 2, rect.height / 2];
    const minimumScale = fitScaleRef.current * 0.25;
    const maximumScale = fitScaleRef.current * 16;
    viewRef.current = zoomTransformAtPoint(viewRef.current, anchor, multiplier, minimumScale, maximumScale);
    setZoomPercent(Math.round((viewRef.current.scale / fitScaleRef.current) * 100));
    requestDraw((value) => value + 1);
  };
  const onWheel = (event) => {
    event.preventDefault();
    zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12, canvasPoint(event));
  };
  return (
    <section className="canvas-shell">
      <div className="toolbar">
        {annotationMode !== "classification" && <IconButton title="Select annotation" active={tool === "select"} onClick={() => setTool("select")}><MousePointer2 size={18} /></IconButton>}
        {hasDetectionTools && <IconButton title="Draw bounding box" active={tool === "draw_box"} onClick={() => setTool("draw_box")}><BoxSelect size={18} /></IconButton>}
        {hasDetectionTools && <button className="ai-box-button" onClick={onOpenDetectionAssist}><Sparkles size={16} /> AI boxes</button>}
        {annotationMode === "detection" && <span className="box-edit-hint">Select a box · drag inside to move · drag a handle to resize</span>}
        {hasSegmentationTools && <><button className="ai-mask-button" onClick={onOpenSegmentationAssist}><Sparkles size={16} /> SAM mask</button><button className="new-mask-button" onClick={onCreateMask}><Plus size={16} /> Manual mask</button><IconButton title="Add brush (A)" active={tool === "add"} disabled={!selected || isFruit(selected)} onClick={() => setTool("add")}><Brush size={18} /></IconButton><IconButton title="Erase brush (E)" active={tool === "erase"} disabled={!selected || isFruit(selected)} onClick={() => setTool("erase")}><Eraser size={18} /></IconButton><IconButton title="Polygon (double-click to close)" active={tool === "polygon"} disabled={!selected || isFruit(selected)} onClick={() => { polygonRef.current = []; setTool("polygon"); }}><ScissorsLineDashed size={18} /></IconButton></>}
        {annotationMode === "classification" && <span className="classification-hint"><Tag size={16} /> Choose the image class in Properties</span>}
        <IconButton title="Edit source image" onClick={onEditImage}><Pencil size={18} /></IconButton>
        <IconButton title="Pan (Space-drag)" active={tool === "pan"} onClick={() => setTool("pan")}><Hand size={18} /></IconButton>
        <div className="canvas-zoom-controls" aria-label="Frame zoom controls">
          <IconButton title="Zoom out" onClick={() => zoomBy(0.8)} disabled={zoomPercent <= 25}><ZoomOut size={18} /></IconButton>
          <output aria-live="polite" title="Current zoom">{zoomPercent}%</output>
          <IconButton title="Zoom in" onClick={() => zoomBy(1.25)} disabled={zoomPercent >= 1600}><ZoomIn size={18} /></IconButton>
          <IconButton title="Fit frame (F)" onClick={fit} disabled={zoomPercent === 100}><Maximize size={18} /></IconButton>
        </div>
        {annotationMode !== "classification" && <div className="canvas-history-controls" aria-label="Annotation edit history"><button className="undo-mask-button" data-testid="undo-mask" title="Undo the last mask or annotation edit (Ctrl/Cmd+Z)" onClick={undo} disabled={!canUndo}><Undo2 size={18} /> Undo {hasSegmentationTools ? "mask edit" : "edit"}</button><IconButton title="Redo annotation edit (Ctrl/Cmd+Shift+Z)" onClick={redo} disabled={!canRedo}><Redo2 size={18} /></IconButton></div>}
        {hasSegmentationTools && <IconButton title={isFruit(selected) ? "Boxes have no mask pixels to clear" : "Clear selected mask pixels"} disabled={!selected || isFruit(selected)} onClick={clearSelectedMask}><ScanSearch size={18} /></IconButton>}
        {annotationMode !== "classification" && <IconButton title="Delete selected annotation (Delete)" disabled={!selected} onClick={deleteSelected}><Trash2 size={18} /></IconButton>}
        {hasSegmentationTools && <><IconButton title={showMasks ? "Hide masks" : "Show masks"} active={showMasks} onClick={() => setShowMasks((value) => !value)}>{showMasks ? <Eye size={18} /> : <EyeOff size={18} />}</IconButton><label className="range-control"><span>Brush {brushSize}px <span className="brush-shortcuts"><kbd>[</kbd><kbd>]</kbd></span></span><input type="range" min="2" max="200" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} /></label><label className="range-control"><span>Opacity</span><input type="range" min="0" max="0.9" step="0.05" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} /></label></>}
      </div>
      <canvas ref={canvasRef} style={{ cursor: tool === "select" ? "default" : tool === "pan" ? "grab" : "crosshair" }} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onPointerLeave={() => { cursorRef.current = null; if (canvasRef.current) canvasRef.current.style.cursor = tool === "select" ? "default" : tool === "pan" ? "grab" : "crosshair"; requestDraw((value) => value + 1); }} onWheel={onWheel} onContextMenu={(event) => event.preventDefault()} />
    </section>
  );
}

function SamPanel({ provider, classes, classUuid, setClassUuid, prompts, setPrompts, tool, setTool, proposals, proposalIndex, setProposalIndex, onLoad, onPredict, onAccept, onReject, busy }) {
  const textSupported = provider?.capabilities?.text !== false;
  const pointSupported = provider?.capabilities?.positive_points !== false;
  const hasPrompt = (pointSupported && prompts.points.length > 0) || Boolean(prompts.box) || (textSupported && Boolean(prompts.text.trim()));
  const unavailable = provider?.state === "unavailable" || provider?.state === "failed";
  const ready = provider?.state === "ready";
  const resolvedClassUuid = classes.some((item) => item.uuid === classUuid) ? classUuid : (classes[0]?.uuid || "");
  const providerLabel = provider?.provider === "sam2" ? "SAM 2.1" : provider?.provider === "sam3" ? "SAM 3" : provider?.provider === "mock" ? "Mock preview" : "SAM";
  return (
    <section className="sam-panel" aria-label="SAM initial mask assistant">
      <div className="panel-head"><span><Sparkles size={14} /> SAM initial mask</span><span className={`provider-state ${provider?.state}`}>{providerLabel} · {provider?.state || "unknown"}</span></div>
      <p className="assist-intro">Prompt one visible instance, preview the result in cyan, then accept it as an editable mask. Nothing is saved until you accept.</p>
      <div className="sam-step"><span>1</span><div><strong>Choose a class</strong><small>The accepted mask becomes one instance of this class.</small></div></div>
      <label className="field"><span>Save accepted mask as</span><select value={resolvedClassUuid} onChange={(event) => setClassUuid(event.target.value)}>{classes.map((item) => <option key={item.uuid} value={item.uuid}>{item.display_name} · class {item.export_index}</option>)}</select></label>
      <div className="sam-step"><span>2</span><div><strong>Prompt the object on the image</strong><small>{pointSupported ? "A positive point is fastest. Add negative points to remove nearby regions, or draw a tight box." : "Draw a tight box, enter a text concept, or combine both."}</small></div></div>
      <div className="sam-tools">
        {pointSupported && <button className={tool === "sam_positive" ? "active" : ""} onClick={() => setTool("sam_positive")}><Plus size={16} /> Include point</button>}
        {provider?.capabilities?.negative_points && <button className={tool === "sam_negative" ? "active" : ""} onClick={() => setTool("sam_negative")}><X size={16} /> Exclude point</button>}
        <button className={tool === "sam_box" ? "active" : ""} onClick={() => setTool("sam_box")}><BoxSelect size={16} /> Tight box</button>
      </div>
      {textSupported && <label className="field"><span>Text concept (optional)</span><input value={prompts.text} onChange={(event) => setPrompts((current) => ({ ...current, text: event.target.value }))} placeholder="e.g. citrus canker lesion" /></label>}
      <div className="prompt-summary"><span>{prompts.points.filter((point) => point.label).length} include</span><span>{prompts.points.filter((point) => !point.label).length} exclude</span><span>{prompts.box ? "box ready" : "no box"}</span><button className="compact" onClick={() => setPrompts(blankPrompts())} disabled={!prompts.points.length && !prompts.box && !prompts.text}>Clear</button></div>
      <div className="sam-step"><span>3</span><div><strong>Preview and refine</strong><small>Compare alternatives before accepting. Use brush and eraser for final edge cleanup.</small></div></div>
      <div className="sam-actions">
        <button onClick={onLoad} disabled={busy || ready}>{provider?.state === "loading" ? <LoaderCircle className="spin" size={16} /> : null} {ready ? `${providerLabel} ready` : `Load ${providerLabel}`}</button>
        <button className="primary-action" onClick={onPredict} disabled={busy || !ready || !hasPrompt}>{busy ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />} Preview mask</button>
      </div>
      {!ready && provider?.state === "idle" && <small className="model-note">Load once per session. Image embeddings are cached while you refine prompts.</small>}
      {unavailable && <small className="model-error">{provider?.error || "SAM is unavailable. You can continue with manual masks."}</small>}
      {provider?.official_repository && <small className="model-note"><a href={provider.official_repository} target="_blank" rel="noreferrer">Official {providerLabel}</a>{provider?.model_id ? ` · ${provider.model_id}` : ""}{provider?.device ? ` · ${provider.device}` : ""}{provider?.peak_vram_mb ? ` · ${Math.round(provider.peak_vram_mb)} MB peak VRAM` : ""}</small>}
      {proposals.length > 0 && <div className="sam-proposals"><label className="field"><span>Preview alternative</span><select value={proposalIndex} onChange={(event) => setProposalIndex(Number(event.target.value))}>{proposals.map((item, index) => <option key={item.id} value={index}>Option {index + 1} · {Math.round(Number(item.score) * 100)}% score</option>)}</select></label><div className="sam-actions"><button className="accept-mask" onClick={onAccept}><Check size={16} /> Accept and edit mask</button><button onClick={onReject}><X size={16} /> Dismiss preview</button></div></div>}
    </section>
  );
}

function DetectionAssistPanel({ provider, classes, prompt, setPrompt, confidence, setConfidence, maxDetections, setMaxDetections, classUuid, setClassUuid, proposals, busy, onLoad, onPredict, onAccept, onAcceptAll, onReject }) {
  const labels = parseDetectionLabels(prompt);
  const unavailable = provider?.state === "unavailable" || provider?.state === "failed";
  return (
    <section className="detection-assist" aria-label="AI bounding box assistant">
      <div className="panel-head"><span>AI box assistant</span><span className={`provider-state ${provider?.state}`}>{provider?.state || "unknown"}</span></div>
      <p className="assist-intro">Describe the objects to find. Grounding DINO previews boxes; you decide which ones become editable annotations.</p>
      <label className="field"><span>Objects to find</span><textarea rows="2" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="orange, naartjie, citrus fruit" /></label>
      <label className="field"><span>Save accepted boxes as</span><select value={classUuid || classes[0]?.uuid || ""} onChange={(event) => setClassUuid(event.target.value)}>{classes.map((item) => <option key={item.uuid} value={item.uuid}>{item.display_name}</option>)}</select></label>
      <label className="range-control detection-confidence"><span>Minimum confidence {Math.round(confidence * 100)}%</span><input aria-label="Minimum detection confidence" type="range" min="0.1" max="0.8" step="0.05" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} /></label>
      <label className="range-control detection-confidence"><span>Maximum boxes <output>{maxDetections}</output></span><input aria-label="Maximum proposed boxes" type="range" min="1" max="100" step="1" value={maxDetections} onChange={(event) => setMaxDetections(Number(event.target.value))} /></label>
      <div className="sam-actions">
        <button onClick={onLoad} disabled={busy || provider?.state === "ready"}>{provider?.state === "loading" ? <LoaderCircle className="spin" size={16} /> : null} Load detector</button>
        <button onClick={onPredict} disabled={busy || unavailable || provider?.state !== "ready" || !labels.length}>{busy ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />} Find boxes</button>
      </div>
      {provider?.state === "idle" && <small className="model-note">First load downloads and caches the model. Detection then runs locally on this computer.</small>}
      {unavailable && <small className="model-error">{provider?.error || "The detector is unavailable."}</small>}
      {provider?.model_url && <small className="model-note"><a href={provider.model_url} target="_blank" rel="noreferrer">{provider.model_id}</a> · {provider.license} · {provider.device || "auto"}</small>}
      {proposals.length > 0 && <div className="detection-proposals">
        <div className="proposal-summary"><strong>{proposals.length} preview{proposals.length === 1 ? "" : "s"}</strong><button className="compact" onClick={onReject}>Clear</button></div>
        <div className="proposal-list">{proposals.map((proposal, index) => <div className="proposal-row" key={proposal.id || index}><span><strong>{proposal.label || "object"}</strong><small>{Math.round(Number(proposal.score || 0) * 100)}% confidence</small></span><button className="compact" onClick={() => onAccept(proposal)}><Check size={14} /> Use</button></div>)}</div>
        <button className="accept-all" onClick={onAcceptAll}><Check size={16} /> Use all {proposals.length} boxes</button>
      </div>}
      {provider?.state === "ready" && !busy && !proposals.length && <small className="model-note">No preview boxes yet. Adjust the prompt or confidence if the first search misses fruit.</small>}
    </section>
  );
}

function ClassificationPanel({ classes, image, onSelect }) {
  return (
    <section className="classification-panel" aria-label="Image class">
      <div className="panel-head"><span>Image class</span><span>one label per frame</span></div>
      <p className="assist-intro">Choose the class that best describes the whole image. Number keys 1–9 select classes quickly.</p>
      <div className="class-choice-grid">
        {classes.map((item, index) => {
          const active = String(image.classification_class_uuid || "") === String(item.uuid);
          return <button key={item.uuid} className={active ? "class-choice active" : "class-choice"} onClick={() => onSelect(item)}><span className="class-swatch" style={{ background: item.color }} /><span><strong>{item.display_name}</strong><small>Key {index + 1} · class {item.export_index}</small></span>{active && <Check size={17} />}</button>;
        })}
      </div>
    </section>
  );
}

function Inspector({ project, image, updateImage, instances, selectedId, setSelectedId, updateInstance, newInstance, onDeleteSelected, onClearSelectedMask, onClearAnnotations, onDeriveBox, providerProps, detectionProps, summary, onClose, onMarkDone, onMarkEmpty }) {
  const selected = instances.find((item) => String(item.id) === String(selectedId));
  const fruitInstances = instances.filter(isFruit);
  const mode = project.annotation_mode || "mixed";
  const classes = selected?.annotation_kind === "fruit_box" ? project.detection_classes : project.segmentation_classes;
  const updateClass = (classUuid) => {
    const item = classes.find((entry) => entry.uuid === classUuid);
    updateInstance({ class_uuid: item.uuid, export_index: item.export_index, training_class_id: item.export_index, category_id: item.export_index, class_name: item.name, name: item.display_name, display_label: item.display_name, color: item.color });
  };
  if (mode === "classification") {
    const selectClass = (item) => updateImage({
      classification_class_uuid: item.uuid,
      classification_class_name: item.name,
      classification_export_index: item.export_index,
      status: "draft",
      reviewed: false,
      excluded: false
    });
    return (
      <aside className="sidebar right drawer" aria-label="Classification properties">
        <div className="drawer-heading"><div><strong>Properties</strong><small>classification</small></div><IconButton title="Close classification properties" onClick={onClose}><X size={17} /></IconButton></div>
        <div className="completion-actions single"><button className="success" title="Approve label (Y)" onClick={onMarkDone} disabled={!image.classification_class_uuid}><Check size={16} /> Approve label <kbd>Y</kbd></button></div>
        <ClassificationPanel classes={project.classification_classes || []} image={image} onSelect={selectClass} />
        <div className="metadata-grid classification-metadata">
          <label className="field"><span>Status</span><select value={image.status} onChange={(event) => updateImage({ status: event.target.value })}><option value="unlabelled">Unlabelled</option><option value="draft">Draft</option><option value="review">Review</option><option value="approved">Approved</option><option value="excluded">Excluded</option></select></label>
          {image.source_video_name && <div className="source-meta"><span>{image.source_video_name}</span><small>{(Number(image.sample_time_ms || 0) / 1000).toFixed(1)} seconds · frame {image.source_frame_index}</small></div>}
          <label className="field"><span>Split</span><input value={image.split || "Not assigned"} readOnly /></label>
          <label className="check-row"><input type="checkbox" checked={Boolean(image.excluded)} onChange={(event) => updateImage({ excluded: event.target.checked, status: event.target.checked ? "excluded" : (image.classification_class_uuid ? "draft" : "unlabelled") })} /> Exclude / quarantine</label>
          {image.excluded && <textarea value={image.exclusion_reason || ""} onChange={(event) => updateImage({ exclusion_reason: event.target.value }, true)} placeholder="Reversible exclusion reason" />}
        </div>
        {summary && <div className="summary-text"><span>{summary.image_count} images · {Object.values(summary.classification_class_counts || {}).reduce((total, count) => total + count, 0)} labelled</span><span>{summary.approved_count || 0} approved · {summary.warning_count || 0} warnings</span></div>}
      </aside>
    );
  }
  return (
    <aside className="sidebar right drawer" aria-label="Annotation properties">
      <div className="drawer-heading"><div><strong>Properties</strong><small>{mode === "mixed" ? "Mixed annotation" : mode}</small></div><IconButton title="Close annotation properties" onClick={onClose}><X size={17} /></IconButton></div>
      <div className="completion-actions"><button className="success" title="Mark image done (Y)" onClick={onMarkDone}><Check size={16} /> Mark image done <kbd>Y</kbd></button><button title="No instances (N)" onClick={onMarkEmpty}>No instances <kbd>N</kbd></button></div>
      <div className="metadata-grid">
        <label className="field"><span>Status</span><select value={image.status} onChange={(event) => updateImage({ status: event.target.value })}><option value="unlabelled">Unlabelled</option><option value="draft">Draft</option><option value="review">Review</option><option value="approved">Approved</option><option value="excluded">Excluded</option></select></label>
        {project.source_kind !== "video_frames" && <><label className="field"><span>Optional capture group</span><input value={image.fruit_id || ""} onChange={(event) => updateImage({ fruit_id: event.target.value }, true)} placeholder="same fruit / burst group" /></label><label className="field"><span>View ID</span><input value={image.view_id || ""} onChange={(event) => updateImage({ view_id: event.target.value }, true)} placeholder="side-a" /></label></>}
        {image.source_video_name && <div className="source-meta"><span>{image.source_video_name}</span><small>{(Number(image.sample_time_ms || 0) / 1000).toFixed(1)} seconds · frame {image.source_frame_index}</small></div>}
        <label className="field"><span>Split</span><input value={image.split || "Not assigned"} readOnly /></label>
        <label className="check-row"><input type="checkbox" checked={Boolean(image.excluded)} onChange={(event) => updateImage({ excluded: event.target.checked, status: event.target.checked ? "excluded" : (instances.length ? "draft" : "unlabelled") })} /> Exclude / quarantine</label>
        {image.excluded && <textarea value={image.exclusion_reason || ""} onChange={(event) => updateImage({ exclusion_reason: event.target.value }, true)} placeholder="Reversible exclusion reason" />}
      </div>
      {(mode === "detection" || mode === "mixed") && <DetectionAssistPanel classes={project.detection_classes} {...detectionProps} />}
      {mode !== "detection" && <SamPanel {...providerProps} />}
      {mode === "segmentation" && <section className="segmentation-start"><div className="panel-head"><span>Manual mask</span><span>one mask per instance</span></div><p className="assist-intro">Choose a class to create an empty instance, then paint it with the brush or polygon tool. Use a separate mask for every visible object or defect.</p><div className="mask-class-grid">{project.segmentation_classes.map((item) => <button key={item.uuid} onClick={() => newInstance("defect_mask", null, "manual", {}, null, item.uuid)}><span className="class-swatch" style={{ background: item.color }} /><Plus size={15} /> {item.display_name}</button>)}</div></section>}
      <div className="panel-head instance-head"><span>Instances ({instances.length})</span><span>{mode !== "segmentation" && <button className="compact" onClick={() => newInstance("fruit_box")}><Plus size={15} /> Box</button>}{mode === "mixed" && <button className="compact" onClick={() => newInstance("defect_mask")}><Plus size={15} /> Mask</button>}{mode === "mixed" && <button className="compact" onClick={() => newInstance("foreground_mask")}><Plus size={15} /> Foreground</button>}</span></div>
      <div className="instance-list">
        {instances.map((instance, index) => <button key={instance.id} className={String(instance.id) === String(selectedId) ? "instance-row active" : "instance-row"} onClick={() => setSelectedId(instance.id)}><span className="color-dot" style={{ background: instance.color }} /> <span><strong>{index + 1}. {instance.name}</strong><small>{instance.source || instance.provenance}{instance.prompt_metadata?.score != null ? ` · ${Math.round(Number(instance.prompt_metadata.score) * 100)}% SAM` : ""}</small></span>{instance.locked ? <small>locked</small> : null}</button>)}
      </div>
      <div className="annotation-actions">
        <button onClick={onDeleteSelected} disabled={!selected}><Trash2 size={15} /> Delete selected</button>
        <button className="danger" onClick={onClearAnnotations} disabled={!instances.length}><Trash2 size={15} /> Clear image annotations</button>
      </div>
      {selected && <section className="instance-editor">
        <label className="field"><span>Annotation type</span><input value={selected.annotation_kind} readOnly /></label>
        {selected.annotation_kind !== "foreground_mask" && <label className="field"><span>Class → export index</span><select value={selected.class_uuid || ""} onChange={(event) => updateClass(event.target.value)}>{classes.map((item) => <option key={item.uuid} value={item.uuid}>{item.display_name} → {item.export_index}</option>)}</select></label>}
        {isFruit(selected) && <div className="form-grid bbox-fields">{["x", "y", "width", "height"].map((label, index) => <label className="field" key={label}><span>{label} (px)</span><input type="number" min="0" value={Math.round(selected.bbox?.[index] || 0)} onChange={(event) => { const bbox = [...selected.bbox]; bbox[index] = Number(event.target.value); updateInstance({ bbox }); }} /></label>)}</div>}
        {selected.annotation_kind === "defect_mask" && <label className="field"><span>Optional parent fruit UUID</span><select value={selected.parent_fruit_uuid || selected.parent_fruit_instance_id || ""} onChange={(event) => updateInstance({ parent_fruit_uuid: event.target.value || null, parent_fruit_instance_id: event.target.value || null })}><option value="">None</option>{fruitInstances.map((fruit) => <option key={fruit.id} value={fruit.id}>{fruit.name} · {fruit.id.slice(0, 8)}</option>)}</select></label>}
        {selected.annotation_kind === "foreground_mask" && <button onClick={() => onDeriveBox(selected.uuid)}><BoxSelect size={16} /> Confirm foreground and derive fruit box</button>}
        <label className="field"><span>Instance name</span><input value={selected.name} onChange={(event) => updateInstance({ name: event.target.value })} /></label>
        <label className="field"><span>Review state</span><select value={selected.review_status} onChange={(event) => updateInstance({ review_status: event.target.value })}><option value="draft">Draft</option><option value="review">Needs review</option><option value="approved">Approved</option></select></label>
        <div className="two-checks"><label className="check-row"><input type="checkbox" checked={selected.visible} onChange={(event) => updateInstance({ visible: event.target.checked })} /> Visible</label><label className="check-row"><input type="checkbox" checked={selected.locked} onChange={(event) => updateInstance({ locked: event.target.checked })} /> Locked</label></div>
        {!isFruit(selected) && <button onClick={onClearSelectedMask}><ScanSearch size={15} /> Clear selected mask pixels</button>}
      </section>}
      {summary && <div className="summary-text"><span>{summary.image_count} images · {summary.annotation_count} instances</span><span>{summary.approved_count || 0} approved · {summary.warning_count || 0} warnings</span></div>}
    </aside>
  );
}

function HelpModal({ onClose }) {
  const shortcuts = [["B", "Draw bounding box"], ["A / E", "Add / erase mask brush"], ["[ / ]", "Decrease / increase the mask or source eraser brush"], ["Mouse wheel", "Zoom into or out of the frame at the pointer"], ["Y", "Mark image done or approve its label"], ["N", "Mark the current image as No instances"], ["Delete / Backspace", "Delete selected annotation"], ["Ctrl/Cmd+Z", "Undo the last mask or annotation edit"], ["Ctrl/Cmd+Shift+Z or Ctrl/Cmd+Y", "Redo"], ["← / →", "Save and open the previous / next frame"], ["1–9", "Assign an annotation or image class"], ["S", "Save"], ["F", "Fit frame to the workspace"], ["Space-drag", "Pan while zoomed"]];
  return <Modal title="Keyboard shortcuts" onClose={onClose}><div className="shortcut-list">{shortcuts.map(([key, action]) => <React.Fragment key={key}><kbd>{key}</kbd><span>{action}</span></React.Fragment>)}</div><p>Shortcuts are disabled while typing in an input, select, or text area.</p></Modal>;
}

const IMAGE_EDIT_OPTIONS = [
  { id: "rotate_left", label: "Rotate left", icon: RotateCcw, transform: "rotate(-90deg)" },
  { id: "rotate_right", label: "Rotate right", icon: RotateCw, transform: "rotate(90deg)" },
  { id: "flip_horizontal", label: "Flip horizontally", icon: FlipHorizontal2, transform: "scaleX(-1)" },
  { id: "flip_vertical", label: "Flip vertically", icon: FlipVertical2, transform: "scaleY(-1)" },
];

function drawEraseSegment(context, from, to, size, compositeOperation = "source-over") {
  context.save();
  context.globalCompositeOperation = compositeOperation;
  context.strokeStyle = "#ffffff";
  context.fillStyle = "#ffffff";
  context.lineWidth = size;
  context.lineCap = "round";
  context.lineJoin = "round";
  if (from[0] === to[0] && from[1] === to[1]) {
    context.beginPath(); context.arc(from[0], from[1], size / 2, 0, Math.PI * 2); context.fill();
  } else {
    context.beginPath(); context.moveTo(from[0], from[1]); context.lineTo(to[0], to[1]); context.stroke();
  }
  context.restore();
}

function ImageEditorModal({ image, busy, onClose, onTransform, onErase }) {
  const [mode, setMode] = useState("erase");
  const [operation, setOperation] = useState("rotate_right");
  const [brushSize, setBrushSize] = useState(() => Math.max(4, Math.min(100, Math.round(Math.min(image.width, image.height) * 0.03))));
  const [strokes, setStrokes] = useState([]);
  const [imageReady, setImageReady] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [zoom, setZoom] = useState(1);
  const [fitScale, setFitScale] = useState(1);
  const [viewportSize, setViewportSize] = useState({ width: 1, height: 1 });
  const [panMode, setPanMode] = useState(false);
  const previewRef = useRef(null);
  const viewportRef = useRef(null);
  const eraseMaskRef = useRef(document.createElement("canvas"));
  const sourceImageRef = useRef(null);
  const activeStrokeRef = useRef(null);
  const activePanRef = useRef(null);
  const strokesRef = useRef([]);
  const selected = IMAGE_EDIT_OPTIONS.find((item) => item.id === operation);
  const swapsDimensions = operation === "rotate_left" || operation === "rotate_right";
  const maximumBrush = Math.max(20, Math.min(300, Math.round(Math.min(image.width, image.height) / 3)));

  const renderErasePreview = useCallback(() => {
    const canvas = previewRef.current;
    const source = sourceImageRef.current;
    if (!canvas || !source) return;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.globalCompositeOperation = "source-over";
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    context.globalCompositeOperation = "destination-out";
    context.drawImage(eraseMaskRef.current, 0, 0);
    context.globalCompositeOperation = "source-over";
  }, []);

  const rebuildEraseMask = useCallback((nextStrokes) => {
    const mask = eraseMaskRef.current;
    const context = mask.getContext("2d");
    context.clearRect(0, 0, mask.width, mask.height);
    for (const stroke of nextStrokes) {
      if (stroke.points.length === 1) drawEraseSegment(context, stroke.points[0], stroke.points[0], stroke.size);
      for (let index = 1; index < stroke.points.length; index += 1) {
        drawEraseSegment(context, stroke.points[index - 1], stroke.points[index], stroke.size);
      }
    }
    renderErasePreview();
  }, [renderErasePreview]);

  useEffect(() => {
    let cancelled = false;
    setImageReady(false);
    setLoadError("");
    setZoom(1);
    setPanMode(false);
    strokesRef.current = [];
    setStrokes([]);
    const loaded = new Image();
    loaded.onload = () => {
      if (cancelled) return;
      sourceImageRef.current = loaded;
      setImageReady(true);
      const preview = previewRef.current;
      if (!preview) return;
      preview.width = Number(image.width);
      preview.height = Number(image.height);
      eraseMaskRef.current.width = Number(image.width);
      eraseMaskRef.current.height = Number(image.height);
      renderErasePreview();
    };
    loaded.onerror = () => { if (!cancelled) setLoadError("The source image could not be loaded for editing."); };
    loaded.src = `/api/image-file/${image.id}?v=${image.revision || 0}`;
    return () => { cancelled = true; activeStrokeRef.current = null; sourceImageRef.current = null; };
  }, [image.height, image.id, image.revision, image.width, renderErasePreview]);
  useEffect(() => {
    if (mode !== "erase" || !viewportRef.current) return;
    const viewport = viewportRef.current;
    const updateFitScale = () => {
      const width = Math.max(1, viewport.clientWidth);
      const height = Math.max(1, viewport.clientHeight);
      setViewportSize({ width, height });
      setFitScale(Math.min(width / Number(image.width), height / Number(image.height)) * 0.96);
    };
    updateFitScale();
    const observer = new ResizeObserver(updateFitScale);
    observer.observe(viewport);
    return () => observer.disconnect();
  }, [image.height, image.width, mode]);
  useEffect(() => {
    if (mode !== "erase" || !sourceImageRef.current || !previewRef.current) return;
    previewRef.current.width = Number(image.width);
    previewRef.current.height = Number(image.height);
    eraseMaskRef.current.width = Number(image.width);
    eraseMaskRef.current.height = Number(image.height);
    rebuildEraseMask(strokesRef.current);
  }, [image.height, image.width, mode, rebuildEraseMask]);
  useEffect(() => {
    const handler = (event) => {
      const rangeHasFocus = event.target?.matches?.('input[type="range"]');
      if (busy || mode !== "erase" || (shouldIgnoreShortcut(event.target) && !rangeHasFocus)) return;
      if (event.key === "[") {
        event.preventDefault();
        setBrushSize((value) => Math.max(1, value - 2));
      } else if (event.key === "]") {
        event.preventDefault();
        setBrushSize((value) => Math.min(maximumBrush, value + 2));
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [busy, maximumBrush, mode]);

  const imagePoint = (event) => {
    const canvas = previewRef.current;
    const rect = canvas.getBoundingClientRect();
    return [
      Math.max(0, Math.min(canvas.width - 1, (event.clientX - rect.left) * canvas.width / rect.width)),
      Math.max(0, Math.min(canvas.height - 1, (event.clientY - rect.top) * canvas.height / rect.height)),
    ];
  };
  const paintEraseSegment = (from, to) => {
    drawEraseSegment(eraseMaskRef.current.getContext("2d"), from, to, brushSize);
    drawEraseSegment(previewRef.current.getContext("2d"), from, to, brushSize, "destination-out");
  };
  const beginErase = (event) => {
    if (busy || mode !== "erase" || !imageReady) return;
    if (panMode || event.button === 1) {
      const viewport = viewportRef.current;
      event.preventDefault();
      previewRef.current.setPointerCapture(event.pointerId);
      activePanRef.current = { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY, scrollLeft: viewport.scrollLeft, scrollTop: viewport.scrollTop };
      return;
    }
    if (event.button !== 0) return;
    event.preventDefault();
    previewRef.current.setPointerCapture(event.pointerId);
    const point = imagePoint(event);
    activeStrokeRef.current = { pointerId: event.pointerId, size: brushSize, points: [point] };
    paintEraseSegment(point, point);
  };
  const continueErase = (event) => {
    const pan = activePanRef.current;
    if (pan && pan.pointerId === event.pointerId) {
      event.preventDefault();
      viewportRef.current.scrollLeft = pan.scrollLeft - (event.clientX - pan.clientX);
      viewportRef.current.scrollTop = pan.scrollTop - (event.clientY - pan.clientY);
      return;
    }
    const stroke = activeStrokeRef.current;
    if (!stroke || stroke.pointerId !== event.pointerId) return;
    event.preventDefault();
    const point = imagePoint(event);
    const previous = stroke.points[stroke.points.length - 1];
    stroke.points.push(point);
    drawEraseSegment(eraseMaskRef.current.getContext("2d"), previous, point, stroke.size);
    drawEraseSegment(previewRef.current.getContext("2d"), previous, point, stroke.size, "destination-out");
  };
  const finishErase = (event) => {
    const pan = activePanRef.current;
    if (pan && pan.pointerId === event.pointerId) {
      activePanRef.current = null;
      try { previewRef.current.releasePointerCapture(event.pointerId); } catch { /* capture was already released */ }
      return;
    }
    const stroke = activeStrokeRef.current;
    if (!stroke || stroke.pointerId !== event.pointerId) return;
    activeStrokeRef.current = null;
    const next = [...strokesRef.current, { size: stroke.size, points: stroke.points }];
    strokesRef.current = next;
    setStrokes(next);
    try { previewRef.current.releasePointerCapture(event.pointerId); } catch { /* capture was already released */ }
  };
  const undoErase = () => {
    const next = strokesRef.current.slice(0, -1);
    strokesRef.current = next;
    setStrokes(next);
    rebuildEraseMask(next);
  };
  const resetErase = () => {
    strokesRef.current = [];
    setStrokes([]);
    rebuildEraseMask([]);
  };
  const changeZoom = (nextZoom) => {
    const viewport = viewportRef.current;
    const canvas = previewRef.current;
    const clamped = Math.max(0.5, Math.min(12, nextZoom));
    if (!viewport || !canvas || clamped === zoom) return;
    const centerX = (viewport.scrollLeft + viewport.clientWidth / 2 - canvas.offsetLeft) / Math.max(1, canvas.offsetWidth);
    const centerY = (viewport.scrollTop + viewport.clientHeight / 2 - canvas.offsetTop) / Math.max(1, canvas.offsetHeight);
    setZoom(clamped);
    requestAnimationFrame(() => {
      const updatedCanvas = previewRef.current;
      const updatedViewport = viewportRef.current;
      if (!updatedCanvas || !updatedViewport) return;
      updatedViewport.scrollLeft = updatedCanvas.offsetLeft + centerX * updatedCanvas.offsetWidth - updatedViewport.clientWidth / 2;
      updatedViewport.scrollTop = updatedCanvas.offsetTop + centerY * updatedCanvas.offsetHeight - updatedViewport.clientHeight / 2;
    });
  };
  const onEraseWheel = (event) => {
    event.preventDefault();
    changeZoom(zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2));
  };
  const canvasDisplayWidth = Math.max(1, image.width * fitScale * zoom);
  const canvasDisplayHeight = Math.max(1, image.height * fitScale * zoom);

  return (
    <Modal title="Edit source pixels" onClose={busy ? () => {} : onClose} wide>
      <div className="image-edit-tabs" role="tablist"><button role="tab" aria-selected={mode === "erase"} className={mode === "erase" ? "active" : ""} onClick={() => setMode("erase")} disabled={busy}><Eraser size={18} /> Erase pixels</button><button role="tab" aria-selected={mode === "transform"} className={mode === "transform" ? "active" : ""} onClick={() => setMode("transform")} disabled={busy}><RotateCw size={18} /> Rotate & flip</button></div>
      {mode === "erase" ? <>
        <p className="image-edit-instruction">Paint over the leftover GrabCut background. Use the mouse wheel or zoom buttons for edge detail, then Pan to move around the enlarged image.</p>
        <div className="source-erase-toolbar"><label><span><span>Brush <span className="brush-shortcuts"><kbd>[</kbd><kbd>]</kbd></span></span><output>{brushSize}px</output></span><input type="range" min="1" max={maximumBrush} value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} disabled={busy} /></label><div className="image-zoom-controls" aria-label="Eraser zoom controls"><button title="Zoom out" aria-label="Zoom out" onClick={() => changeZoom(zoom / 1.25)} disabled={busy || zoom <= 0.5}><ZoomOut size={16} /></button><output>{Math.round(zoom * 100)}%</output><button title="Zoom in" aria-label="Zoom in" onClick={() => changeZoom(zoom * 1.25)} disabled={busy || zoom >= 12}><ZoomIn size={16} /></button><button title="Fit image" aria-label="Fit image" onClick={() => changeZoom(1)} disabled={busy || zoom === 1}><Maximize size={16} /></button></div><button className={panMode ? "active" : ""} onClick={() => setPanMode((value) => !value)} disabled={busy}><Hand size={16} /> Pan</button><button onClick={undoErase} disabled={busy || !strokes.length}><Undo2 size={16} /> Undo</button><button onClick={resetErase} disabled={busy || !strokes.length}><X size={16} /> Reset</button></div>
        <div ref={viewportRef} className="image-edit-preview pixel-eraser" onWheel={onEraseWheel}>
          {!imageReady && !loadError && <LoaderCircle className="spin image-edit-loader" size={28} />}
          {loadError && <span className="model-error">{loadError}</span>}
          <div className="image-erase-stage" style={{ width: Math.max(viewportSize.width, canvasDisplayWidth), height: Math.max(viewportSize.height, canvasDisplayHeight) }}><canvas ref={previewRef} className={`${imageReady ? "image-erase-canvas ready" : "image-erase-canvas"}${panMode ? " pan" : ""}`} style={{ width: canvasDisplayWidth, height: canvasDisplayHeight, left: Math.max(0, (viewportSize.width - canvasDisplayWidth) / 2), top: Math.max(0, (viewportSize.height - canvasDisplayHeight) / 2) }} aria-label="Source pixel eraser" onPointerDown={beginErase} onPointerMove={continueErase} onPointerUp={finishErase} onPointerCancel={finishErase} /></div>
        </div>
        <p className="image-edit-note"><strong>This changes the source pixels, not the annotation mask.</strong> The original file is backed up first. JPEG/BMP sources are converted to PNG so erased pixels remain transparent.</p>
      </> : <>
        <div className="image-edit-preview">
          <img className={swapsDimensions ? "rotated" : ""} src={`/api/image-file/${image.id}?v=${image.revision || 0}`} alt={`Preview of ${image.file_name}`} style={{ transform: selected.transform }} />
        </div>
        <div className="image-edit-dimensions"><span>Current: {image.width} × {image.height}px</span><span>After: {swapsDimensions ? `${image.height} × ${image.width}` : `${image.width} × ${image.height}`}px</span></div>
        <div className="image-edit-options">{IMAGE_EDIT_OPTIONS.map((item) => { const EditIcon = item.icon; return <button key={item.id} className={operation === item.id ? "active" : ""} aria-pressed={operation === item.id} onClick={() => setOperation(item.id)} disabled={busy}><EditIcon size={20} /><span>{item.label}</span></button>; })}</div>
        <p className="image-edit-note"><strong>This edits the source image.</strong> Saved boxes and masks are transformed to remain aligned. The original file is copied into the project’s recoverable backups first.</p>
      </>}
      <div className="modal-actions"><button onClick={onClose} disabled={busy}>Cancel</button>{mode === "erase" ? <button className="primary-action" onClick={() => onErase(eraseMaskRef.current.toDataURL("image/png"))} disabled={busy || !imageReady || !strokes.length}>{busy ? <><LoaderCircle className="spin" size={17} /> Applying…</> : <><Eraser size={17} /> Remove painted pixels</>}</button> : <button className="primary-action" onClick={() => onTransform(operation)} disabled={busy}>{busy ? <><LoaderCircle className="spin" size={17} /> Applying…</> : <><Pencil size={17} /> Apply {selected.label.toLowerCase()}</>}</button>}</div>
    </Modal>
  );
}

function DatasetModal({ project, onClose, runOperation, onComplete }) {
  const [outputDir, setOutputDir] = useState(`${project.project_dir}\\exports`);
  const [allowWarnings, setAllowWarnings] = useState(false);
  const [detectionClasses, setDetectionClasses] = useState(clone(project.detection_classes));
  const [segmentationClasses, setSegmentationClasses] = useState(clone(project.segmentation_classes));
  const [classificationClasses, setClassificationClasses] = useState(clone(project.classification_classes || []));
  const mode = project.annotation_mode || "mixed";
  const formats = mode === "detection" ? ["yolo26_detection", "dfine_detection"] : mode === "segmentation" ? ["yolo26_segmentation"] : mode === "classification" ? ["image_classification"] : null;
  const editor = (title, values, setter) => <section><h3>{title}</h3>{values.map((item, index) => <div className="form-grid" key={item.uuid}><label className="field"><span>Name · index {index}</span><input value={item.name} onChange={(event) => setter(values.map((entry, position) => position === index ? { ...entry, name: event.target.value, export_index: position, id: position } : entry))} /></label><label className="field"><span>Display label</span><input value={item.display_name} onChange={(event) => setter(values.map((entry, position) => position === index ? { ...entry, display_name: event.target.value } : entry))} /></label></div>)}<button onClick={() => setter([...values, { uuid: crypto.randomUUID(), id: values.length, export_index: values.length, name: `class_${values.length + 1}`, display_name: `Class ${values.length + 1}`, color: "#5aa7d8", meaning: null }])}><Plus size={15} /> Add class</button></section>;
  const saveClasses = async () => {
    const settings = mode === "detection" ? { detection_classes: detectionClasses } : mode === "segmentation" ? { segmentation_classes: segmentationClasses } : mode === "classification" ? { classification_classes: classificationClasses } : { detection_classes: detectionClasses, segmentation_classes: segmentationClasses };
    const result = await runOperation("Class settings", () => api("/api/project/settings", { method: "PATCH", body: JSON.stringify(settings) }));
    if (result) await onComplete();
  };
  const approved = project.images.filter((image) => image.status === "approved").length;
  const train = project.images.filter((image) => image.split === "train" && !image.excluded).length;
  const val = project.images.filter((image) => image.split === "val" && !image.excluded).length;
  const test = project.images.filter((image) => image.split === "test" && !image.excluded).length;
  return <Modal title="Review and export" onClose={onClose} wide><div className="review-summary"><div><strong>{approved}</strong><span>approved of {project.images.length}</span></div><div><strong>{train}</strong><span>training frames</span></div><div><strong>{val}</strong><span>validation frames</span></div><div><strong>{test}</strong><span>test frames</span></div><div><strong>{mode}</strong><span>project task</span></div></div><p>Validation runs before export. Hard errors block the dataset; acknowledged warnings remain in the portable manifest.</p>{(mode === "detection" || mode === "mixed") && editor("Detection classes", detectionClasses, setDetectionClasses)}{(mode === "segmentation" || mode === "mixed") && editor("Segmentation classes", segmentationClasses, setSegmentationClasses)}{mode === "classification" && editor("Classification classes", classificationClasses, setClassificationClasses)}<div className="modal-actions"><button onClick={saveClasses}><Save size={17} /> Save classes</button></div><hr /><p>{mode === "classification" ? "Exports a ready-to-train ImageFolder dataset for Ultralytics and torchvision." : formats ? `Exports ${formats.map((value) => value.replaceAll("_", " ")).join(" and ")}.` : "Exports the complete legacy detection, segmentation, D-FINE, and synthetic-assets bundle."}</p><div className="form-grid"><label className="field"><span>Export parent folder</span><input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} /></label></div><label className="check-row warning"><input type="checkbox" checked={allowWarnings} onChange={(event) => setAllowWarnings(event.target.checked)} /> I have reviewed and accept validation warnings</label><div className="modal-actions"><button className="primary-action" onClick={() => runOperation("Training export", () => api("/api/export/dataset", { method: "POST", body: JSON.stringify({ output_dir: outputDir, minimum_iou: 0.98, simplification_pixels: 0.5, precision: 6, allow_warnings: allowWarnings, formats }) }))}><Download size={17} /> Validate and export dataset</button></div></Modal>;
}

function SplitModal({ project, onClose, runOperation, onComplete }) {
  const defaults = project.split || {};
  const [train, setTrain] = useState(defaults.ratios?.train ?? 0.7);
  const [val, setVal] = useState(defaults.ratios?.val ?? 0.2);
  const [test, setTest] = useState(defaults.ratios?.test ?? 0.1);
  const [seed, setSeed] = useState(defaults.seed ?? 42);
  const [version, setVersion] = useState(defaults.dataset_version || "0.1.0");
  const [pilotMode, setPilotMode] = useState(false);
  const [freezeTest, setFreezeTest] = useState(defaults.manifest_json ? Boolean(defaults.test_frozen) : true);
  const [confirmFrozen, setConfirmFrozen] = useState(false);
  return <Modal title="Dataset split" onClose={onClose} wide>
    <p>The split is balanced independently for every source class. All camera views inferred to belong to the same physical naartjie stay together, preventing leakage between training, validation, and testing.</p>
    <div className="held-out-note"><strong>Keep test data untouched</strong><span>The test split is reserved for one final evaluation after training. Tune the model only with training and validation data.</span></div>
    <div className="form-grid"><label className="field"><span>Train ratio</span><input type="number" min="0" max="1" step="0.01" value={train} onChange={(event) => setTrain(Number(event.target.value))} /></label><label className="field"><span>Validation ratio</span><input type="number" min="0" max="1" step="0.01" value={val} onChange={(event) => setVal(Number(event.target.value))} /></label><label className="field"><span>Test ratio</span><input type="number" min="0" max="1" step="0.01" value={test} onChange={(event) => setTest(Number(event.target.value))} /></label><label className="field"><span>Seed</span><input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label><label className="field"><span>Dataset version</span><input value={version} onChange={(event) => setVersion(event.target.value)} /></label></div>
    <label className="check-row"><input type="checkbox" checked={freezeTest} onChange={(event) => setFreezeTest(event.target.checked)} /> Freeze held-out test membership</label>
    <label className="check-row warning"><input type="checkbox" checked={confirmFrozen} onChange={(event) => setConfirmFrozen(event.target.checked)} /> Confirm changing a previously frozen test split (requires a new version)</label>
    <label className="check-row"><input type="checkbox" checked={pilotMode} onChange={(event) => setPilotMode(event.target.checked)} /> Record compatibility pilot flag</label>
    <div className="modal-actions"><button onClick={async () => { const result = await runOperation("Dataset split", () => api("/api/split", { method: "POST", body: JSON.stringify({ ratios: { train, val, test }, seed, dataset_version: version, freeze_test: freezeTest, confirm_frozen_change: confirmFrozen, pilot_mode: pilotMode }) })); if (result) await onComplete(); }}><ScissorsLineDashed size={17} /> Create and freeze split</button></div>
  </Modal>;
}

function App() {
  const [page, setPage] = useState(() => shouldResumeHarvestOnLaunch(window.localStorage) ? "harvest" : "annotation");
  const [project, setProject] = useState(null);
  const [images, setImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentImage, setCurrentImage] = useState(null);
  const [instances, setInstances] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [tool, setTool] = useState("add");
  const [brushSize, setBrushSize] = useState(28);
  const [opacity, setOpacity] = useState(0.4);
  const [showMasks, setShowMasks] = useState(true);
  const [maskRevision, setMaskRevision] = useState(0);
  const [historyState, setHistoryState] = useState({ undo: 0, redo: 0 });
  const [saveState, setSaveState] = useState("saved");
  const [summary, setSummary] = useState(null);
  const [message, setMessageState] = useState(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [imageEditorOpen, setImageEditorOpen] = useState(false);
  const [imageEditBusy, setImageEditBusy] = useState(false);
  const [datasetOpen, setDatasetOpen] = useState(false);
  const [splitOpen, setSplitOpen] = useState(false);
  const [provider, setProvider] = useState(null);
  const [detector, setDetector] = useState(null);
  const [detectionPrompt, setDetectionPrompt] = useState(DEFAULT_DETECTION_PROMPT);
  const [detectionConfidence, setDetectionConfidence] = useState(0.3);
  const [detectionMaxDetections, setDetectionMaxDetections] = useState(25);
  const [detectionClassUuid, setDetectionClassUuid] = useState("");
  const [segmentationClassUuid, setSegmentationClassUuid] = useState("");
  const [boxProposals, setBoxProposals] = useState([]);
  const [detectionBusy, setDetectionBusy] = useState(false);
  const [prompts, setPrompts] = useState(blankPrompts());
  const [proposals, setProposals] = useState([]);
  const [proposalPrompts, setProposalPrompts] = useState(null);
  const [proposalIndex, setProposalIndex] = useState(0);
  const [proposalCanvas, setProposalCanvas] = useState(null);
  const [proposalBusy, setProposalBusy] = useState(false);
  const [navigationBusy, setNavigationBusy] = useState(false);
  const [imagesOpen, setImagesOpen] = useState(false);
  const [propertiesOpen, setPropertiesOpen] = useState(false);
  const masksRef = useRef(new Map());
  const historyRef = useRef([]);
  const redoRef = useRef([]);
  const saveTimerRef = useRef(null);
  const saveChainRef = useRef(Promise.resolve());
  const pointerActiveRef = useRef(false);
  const editVersionRef = useRef(0);
  const dirtyRef = useRef(false);
  const loadTokenRef = useRef(0);
  const requestIdRef = useRef(null);
  const detectionRequestIdRef = useRef(null);
  const fitRef = useRef(() => {});
  const currentImageRef = useRef(null);
  const instancesRef = useRef([]);
  const projectRef = useRef(null);
  const currentIndexRef = useRef(0);
  useEffect(() => { currentImageRef.current = currentImage; instancesRef.current = instances; projectRef.current = project; currentIndexRef.current = currentIndex; }, [currentImage, currentIndex, instances, project]);
  const setMessage = useCallback((text, type = "info") => setMessageState({ text, type }), []);

  const openLoadedProject = useCallback((nextProject, nextSummary) => {
    const nextImages = nextProject.images || [];
    const lastIndex = initialImageIndex(nextImages, nextProject.last_image_id);
    projectRef.current = nextProject;
    currentIndexRef.current = lastIndex;
    currentImageRef.current = null;
    instancesRef.current = [];
    setProject(nextProject); setImages(nextImages); setSummary(nextSummary);
    setCurrentImage(null); setInstances([]); setSelectedId(null); setCurrentIndex(lastIndex);
    setTool(nextProject.annotation_mode === "detection" ? "draw_box" : nextProject.annotation_mode === "classification" ? "pan" : "add");
    setDetectionClassUuid(nextProject.detection_classes?.[0]?.uuid || "");
    setSegmentationClassUuid(nextProject.segmentation_classes?.[0]?.uuid || "");
    setBoxProposals([]);
    setPage("annotation");
    setImagesOpen(false); setPropertiesOpen(["segmentation", "classification"].includes(nextProject.annotation_mode));
  }, []);
  useEffect(() => {
    Promise.all([api("/api/state"), api("/api/segmentation/status"), api("/api/detection/status")]).then(([state, segmentation, detection]) => {
      setProvider(segmentation);
      setDetector(detection);
      if (state.opened) openLoadedProject(state.project, state.summary);
    }).catch((error) => setMessage(error.message, "error"));
  }, [openLoadedProject, setMessage]);

  const loadCurrent = useCallback(async () => {
    if (!project || !images.length) { setCurrentImage(null); setInstances([]); setNavigationBusy(false); return; }
    const token = ++loadTokenRef.current;
    const data = await api(`/api/images/${images[currentIndex].id}`);
    if (token !== loadTokenRef.current) return;
    await api(`/api/images/${data.image.id}/activate`, { method: "POST", body: "{}" });
    const maskEntries = await Promise.all((data.annotations || []).filter((instance) => instance.mask_url).map(async (instance) => [String(instance.id), await maskCanvasFromSource(instance.mask_url, data.image.width, data.image.height)]));
    if (token !== loadTokenRef.current) return;
    masksRef.current = new Map(maskEntries);
    instancesRef.current = data.annotations || [];
    projectRef.current = patchProjectImage(projectRef.current, data.image.id, data.image, true);
    setCurrentImage(data.image); setInstances(instancesRef.current); setSelectedId(data.annotations?.[0]?.id || null);
    setImages((current) => patchImageCollection(current, data.image.id, data.image));
    setProject((current) => patchProjectImage(current, data.image.id, data.image, true));
    historyRef.current = []; redoRef.current = []; setHistoryState({ undo: 0, redo: 0 }); dirtyRef.current = false; setMaskRevision((value) => value + 1); setSaveState("saved");
    setPrompts(blankPrompts()); setProposals([]); setProposalPrompts(null); setProposalCanvas(null); setBoxProposals([]);
    setNavigationBusy(false);
  }, [currentIndex, images, project]);
  useEffect(() => { loadCurrent().catch((error) => { setNavigationBusy(false); setMessage(error.message, "error"); }); }, [currentIndex, project]); // eslint-disable-line react-hooks/exhaustive-deps

  const applyCurrentImagePatch = useCallback((patch) => {
    const current = currentImageRef.current;
    if (!current) return null;
    const next = { ...current, ...patch };
    currentImageRef.current = next;
    projectRef.current = patchProjectImage(projectRef.current, next.id, patch);
    setCurrentImage(next);
    setImages((items) => patchImageCollection(items, next.id, patch));
    setProject((current) => patchProjectImage(current, next.id, patch));
    return next;
  }, []);

  const snapshot = useCallback(() => ({
    instances: clone(instancesRef.current),
    image: currentImageRef.current ? { status: currentImageRef.current.status, reviewed: currentImageRef.current.reviewed, excluded: currentImageRef.current.excluded, exclusion_reason: currentImageRef.current.exclusion_reason, fruit_id: currentImageRef.current.fruit_id, view_id: currentImageRef.current.view_id, classification_class_uuid: currentImageRef.current.classification_class_uuid, classification_class_name: currentImageRef.current.classification_class_name, classification_export_index: currentImageRef.current.classification_export_index } : null,
    masks: Object.fromEntries([...masksRef.current.entries()].map(([id, canvas]) => [id, canvas.toDataURL("image/png")]))
  }), []);
  const restore = useCallback(async (state) => {
    const image = currentImageRef.current;
    const entries = await Promise.all(Object.entries(state.masks).map(async ([id, source]) => [id, await maskCanvasFromSource(source, image.width, image.height)]));
    masksRef.current = new Map(entries); instancesRef.current = state.instances; setInstances(state.instances); applyCurrentImagePatch(state.image);
    editVersionRef.current += 1; dirtyRef.current = true; setMaskRevision((value) => value + 1); setSaveState("dirty");
  }, [applyCurrentImagePatch]);
  const pushHistory = useCallback(() => {
    historyRef.current = [...historyRef.current.slice(-59), snapshot()];
    redoRef.current = [];
    setHistoryState({ undo: historyRef.current.length, redo: 0 });
  }, [snapshot]);

  const performSave = useCallback(async () => {
    const image = currentImageRef.current;
    if (!projectRef.current || !image || !dirtyRef.current) return;
    clearTimeout(saveTimerRef.current);
    const capturedVersion = editVersionRef.current;
    setSaveState("saving");
    const payload = {
      expected_revision: image.revision,
      image: { status: image.status, reviewed: image.status === "approved", excluded: Boolean(image.excluded), exclusion_reason: image.exclusion_reason || "", fruit_id: image.fruit_id || "", view_id: image.view_id || "", split: image.split || null, classification_class_uuid: image.classification_class_uuid || null },
      annotations: instancesRef.current.map((instance) => ({ ...instance, mask_url: undefined, image_id: undefined, mask_path: undefined, mask_data: masksRef.current.get(String(instance.id))?.toDataURL("image/png") }))
    };
    try {
      const data = await api(`/api/images/${image.id}`, { method: "PUT", body: JSON.stringify(payload) });
      const reconciled = reconcileSaveResponse({
        localImage: currentImageRef.current,
        localInstances: instancesRef.current,
        serverImage: data.image,
        serverAnnotations: data.annotations,
        hasNewerEdits: editVersionRef.current !== capturedVersion
      });
      currentImageRef.current = reconciled.image;
      instancesRef.current = reconciled.instances;
      projectRef.current = patchProjectImage(projectRef.current, reconciled.image.id, reconciled.image, true);
      setCurrentImage(reconciled.image); setInstances(reconciled.instances); setSummary(data.summary);
      setImages((current) => patchImageCollection(current, reconciled.image.id, reconciled.image));
      setProject((current) => patchProjectImage(current, reconciled.image.id, reconciled.image, true));
      dirtyRef.current = reconciled.dirty;
      setSaveState(reconciled.dirty ? "dirty" : "saved");
    } catch (error) { setSaveState("error"); setMessage(`Save failed: ${error.message}`, "error"); throw error; }
  }, [setMessage]);
  const saveCurrent = useCallback(() => {
    saveChainRef.current = saveChainRef.current.catch(() => {}).then(performSave);
    return saveChainRef.current;
  }, [performSave]);
  const openHarvest = useCallback(async () => {
    const currentProject = projectRef.current;
    const returning = currentProject?.source_kind === "harvested_foregrounds";
    if (returning && !window.confirm("Save your current defect labels and reopen the original harvest? Existing labels stay attached to the harvested PNGs when you finish harvesting again.")) return;
    setNavigationBusy(true);
    try {
      if (currentProject) await saveCurrent();
      if (returning) {
        const data = await api("/api/harvest/sessions/reopen-current-project", { method: "POST", body: "{}" });
        rememberHarvestSession(window.localStorage, {
          sessionDir: data.session_dir,
          sessionId: data.session.session_id,
          name: data.session.name,
          updatedAt: data.session.updated_at,
        });
        setMessage(`Returned to ${data.session.name}. Your existing defect labels remain saved.`, "success");
      }
      setPage("harvest");
    } catch (error) {
      setMessage(`Could not return to harvesting: ${error.message}`, "error");
    } finally {
      setNavigationBusy(false);
    }
  }, [saveCurrent, setMessage]);
  const scheduleAutosave = useCallback(() => {
    clearTimeout(saveTimerRef.current);
    if (!dirtyRef.current || pointerActiveRef.current) return;
    saveTimerRef.current = window.setTimeout(() => {
      saveTimerRef.current = null;
      saveCurrent();
    }, AUTOSAVE_IDLE_MS);
  }, [saveCurrent]);
  useEffect(() => {
    const beginPointerInteraction = () => {
      pointerActiveRef.current = true;
      clearTimeout(saveTimerRef.current);
    };
    const finishPointerInteraction = () => {
      pointerActiveRef.current = false;
      scheduleAutosave();
    };
    window.addEventListener("pointerdown", beginPointerInteraction, true);
    window.addEventListener("pointerup", finishPointerInteraction, true);
    window.addEventListener("pointercancel", finishPointerInteraction, true);
    window.addEventListener("blur", finishPointerInteraction);
    return () => {
      window.removeEventListener("pointerdown", beginPointerInteraction, true);
      window.removeEventListener("pointerup", finishPointerInteraction, true);
      window.removeEventListener("pointercancel", finishPointerInteraction, true);
      window.removeEventListener("blur", finishPointerInteraction);
      clearTimeout(saveTimerRef.current);
    };
  }, [scheduleAutosave]);
  const markDirty = useCallback((manualChange = false) => {
    if (manualChange) {
      setInstances((current) => {
        const next = current.map((instance) => {
          if (String(instance.id) !== String(selectedId)) return instance;
          const manuallyAdjusted = ["sam2", "sam3", "mock", "grounding_dino"].includes(instance.provenance);
          const provenance = manuallyAdjusted ? `${instance.provenance}_plus_manual` : instance.provenance;
          return { ...instance, source: manuallyAdjusted ? provenance : instance.source, provenance, review_status: "draft" };
        });
        instancesRef.current = next;
        return next;
      });
      if (currentImageRef.current?.status === "approved" || currentImageRef.current?.reviewed) applyCurrentImagePatch({ status: "draft", reviewed: false });
    }
    editVersionRef.current += 1; dirtyRef.current = true; setMaskRevision((value) => value + 1); setSaveState("dirty");
    scheduleAutosave();
  }, [applyCurrentImagePatch, scheduleAutosave, selectedId]);

  const undo = useCallback(async () => {
    if (!historyRef.current.length) return;
    redoRef.current.push(snapshot());
    const previous = historyRef.current.pop();
    await restore(previous);
    setHistoryState({ undo: historyRef.current.length, redo: redoRef.current.length });
    markDirty();
  }, [markDirty, restore, snapshot]);
  const redo = useCallback(async () => {
    if (!redoRef.current.length) return;
    historyRef.current.push(snapshot());
    const next = redoRef.current.pop();
    await restore(next);
    setHistoryState({ undo: historyRef.current.length, redo: redoRef.current.length });
    markDirty();
  }, [markDirty, restore, snapshot]);
  const changeImage = useCallback(async (idOrIndex) => {
    const index = typeof idOrIndex === "number" ? idOrIndex : images.findIndex((item) => String(item.id) === String(idOrIndex));
    const targetIndex = Math.max(0, Math.min(images.length - 1, index));
    if (targetIndex === currentIndexRef.current) return;
    setNavigationBusy(true);
    try {
      await saveCurrent();
      currentIndexRef.current = targetIndex;
      setCurrentIndex(targetIndex);
    } catch {
      setNavigationBusy(false);
    }
  }, [images, saveCurrent]);

  const updateImage = useCallback((patch, quiet = false) => { if (!quiet) pushHistory(); applyCurrentImagePatch(patch); markDirty(); }, [applyCurrentImagePatch, markDirty, pushHistory]);
  const newInstance = useCallback((kind, initialMask = null, provenance = "manual", promptMetadata = {}, bbox = null, classUuid = null) => {
    if (!currentImageRef.current) return;
    pushHistory();
    const id = crypto.randomUUID();
    const availableClasses = kind === "fruit_box" ? projectRef.current.detection_classes : projectRef.current.segmentation_classes;
    const classItem = availableClasses.find((item) => item.uuid === classUuid) || availableClasses[0];
    const base = { id, uuid: id, type: kind, annotation_kind: kind, source: provenance, provenance, prompt_metadata: promptMetadata, review_status: "draft", visible: true, locked: false, notes: "", parent_fruit_uuid: null, parent_fruit_instance_id: null };
    const instance = kind === "foreground_mask"
      ? { ...base, class_uuid: null, export_index: null, training_class_id: null, category_id: null, display_label: "Synthetic foreground", name: "Synthetic foreground", color: "#45c2e8" }
      : { ...base, class_uuid: classItem.uuid, class_name: classItem.name, export_index: classItem.export_index, training_class_id: classItem.export_index, category_id: classItem.export_index, display_label: classItem.display_name, name: classItem.display_name, color: classItem.color };
    if (kind === "fruit_box") {
      const { width, height } = currentImageRef.current;
      instance.bbox = bbox || [Math.round(width * 0.2), Math.round(height * 0.2), Math.round(width * 0.6), Math.round(height * 0.6)];
    } else {
      masksRef.current.set(id, initialMask ? cloneMaskCanvas(initialMask) : blankMaskCanvas(currentImageRef.current.width, currentImageRef.current.height));
    }
    setInstances((current) => { const next = [...current, instance]; instancesRef.current = next; return next; }); setSelectedId(id); setPropertiesOpen(true); applyCurrentImagePatch({ status: "draft", reviewed: false }); markDirty(); setTool(kind === "fruit_box" ? "select" : "add");
  }, [applyCurrentImagePatch, markDirty, pushHistory]);
  const previewBox = useCallback((id, bbox) => {
    setInstances((current) => {
      const next = current.map((instance) => String(instance.id) === String(id) ? { ...instance, bbox } : instance);
      instancesRef.current = next;
      return next;
    });
  }, []);
  const updateInstance = useCallback((patch) => {
    if (!selectedId) return; pushHistory();
    setInstances((current) => { const next = current.map((instance) => String(instance.id) === String(selectedId) ? { ...instance, ...patch, review_status: "draft" } : instance); instancesRef.current = next; return next; });
    applyCurrentImagePatch({ status: "draft", reviewed: false });
    markDirty();
  }, [applyCurrentImagePatch, markDirty, pushHistory, selectedId]);
  const clearSelectedMask = useCallback(() => {
    if (!selectedId || !window.confirm("Clear every pixel in the selected mask? You can still undo this action.")) return;
    pushHistory(); masksRef.current.get(String(selectedId))?.getContext("2d").clearRect(0, 0, currentImageRef.current.width, currentImageRef.current.height); markDirty(true);
  }, [markDirty, pushHistory, selectedId]);
  const deleteSelected = useCallback(() => {
    if (!selectedId || !window.confirm("Delete this instance? Its mask will be moved to the recoverable backup folder.")) return;
    pushHistory(); masksRef.current.delete(String(selectedId)); setInstances((current) => { const next = current.filter((item) => String(item.id) !== String(selectedId)); instancesRef.current = next; return next; }); setSelectedId(null); applyCurrentImagePatch({ status: "draft", reviewed: false }); markDirty();
  }, [applyCurrentImagePatch, markDirty, pushHistory, selectedId]);
  const clearAnnotations = useCallback(() => {
    const count = instancesRef.current.length;
    if (!count || !window.confirm(`Clear all ${count} annotation${count === 1 ? "" : "s"} from this image? This can be undone, and saved mask files are moved to recoverable backups.`)) return;
    pushHistory();
    masksRef.current = new Map();
    instancesRef.current = [];
    setInstances([]);
    setSelectedId(null);
    applyCurrentImagePatch({ status: "unlabelled", reviewed: false });
    markDirty();
  }, [applyCurrentImagePatch, markDirty, pushHistory]);
  const markDone = useCallback(() => {
    const mode = projectRef.current?.annotation_mode;
    if (mode === "classification" && !currentImageRef.current?.classification_class_uuid) {
      setMessage("Choose an image class before approving this frame.", "error");
      return;
    }
    if (mode !== "classification" && !instancesRef.current.length) {
      setMessage("This image has no instances. Add a mask, or use ‘No instances’ to confirm it as a valid negative.", "error");
      return;
    }
    const emptyMask = instancesRef.current.find((instance) => !isFruit(instance) && !maskCanvasHasPixels(masksRef.current.get(String(instance.id))));
    if (emptyMask) {
      setSelectedId(emptyMask.id);
      setPropertiesOpen(true);
      setTool("add");
      setMessage("The selected mask is empty. Paint it, accept a SAM preview, or delete the empty instance before marking this image done.", "error");
      return;
    }
    pushHistory();
    if (mode !== "classification") {
      setInstances((current) => {
        const next = current.map((instance) => ({ ...instance, review_status: "approved" }));
        instancesRef.current = next;
        return next;
      });
    }
    applyCurrentImagePatch({ status: "approved", reviewed: true, excluded: false });
    markDirty();
  }, [applyCurrentImagePatch, markDirty, pushHistory, setMessage]);
  const markEmpty = useCallback(() => {
    const count = instancesRef.current.length;
    if (count && !window.confirm(`Remove ${count} annotation${count === 1 ? "" : "s"} and mark this as a valid negative image?`)) return;
    pushHistory();
    masksRef.current = new Map();
    instancesRef.current = [];
    setInstances([]); setSelectedId(null);
    applyCurrentImagePatch({ status: "approved", reviewed: true, excluded: false });
    markDirty();
  }, [applyCurrentImagePatch, markDirty, pushHistory]);
  const applyImageEdit = useCallback(async ({ operation = null, eraseMaskData = null }) => {
    setImageEditBusy(true);
    setNavigationBusy(true);
    try {
      await saveCurrent();
      const activeImage = currentImageRef.current;
      if (!activeImage) return;
      const erasingPixels = Boolean(eraseMaskData);
      const endpoint = erasingPixels ? "erase-pixels" : "edit";
      const request = erasingPixels
        ? { erase_mask_data: eraseMaskData, expected_revision: activeImage.revision }
        : { operation, expected_revision: activeImage.revision };
      const data = await api(`/api/images/${activeImage.id}/${endpoint}`, {
        method: "POST",
        body: JSON.stringify(request),
      });
      const maskEntries = await Promise.all((data.annotations || []).filter((instance) => instance.mask_url).map(async (instance) => [
        String(instance.id),
        await maskCanvasFromSource(`${instance.mask_url}?v=${data.image.revision}`, data.image.width, data.image.height),
      ]));
      loadTokenRef.current += 1;
      clearTimeout(saveTimerRef.current);
      masksRef.current = new Map(maskEntries);
      instancesRef.current = data.annotations || [];
      currentImageRef.current = data.image;
      historyRef.current = [];
      redoRef.current = [];
      setHistoryState({ undo: 0, redo: 0 });
      dirtyRef.current = false;
      setCurrentImage(data.image);
      setInstances(instancesRef.current);
      setSelectedId(data.annotations?.[0]?.id || null);
      setImages((current) => patchImageCollection(current, data.image.id, data.image));
      setProject((current) => current ? { ...current, images: patchImageCollection(current.images || [], data.image.id, data.image) } : current);
      setSummary(data.summary);
      setPrompts(blankPrompts());
      setProposals([]);
      setProposalPrompts(null);
      setProposalCanvas(null);
      setBoxProposals([]);
      setMaskRevision((value) => value + 1);
      setSaveState("saved");
      setImageEditorOpen(false);
      const resultMessage = erasingPixels
        ? `${data.erased_pixel_count.toLocaleString()} source pixel${data.erased_pixel_count === 1 ? "" : "s"} made transparent${data.converted_to_png ? "; the source was converted to PNG" : ""}.`
        : "Source image transformed. Boxes and masks stayed aligned.";
      setMessage(`${resultMessage} Backup: ${data.backup_path}`, "success");
    } catch (error) {
      setMessage(`Image edit failed: ${error.message}`, "error");
    } finally {
      setImageEditBusy(false);
      setNavigationBusy(false);
    }
  }, [saveCurrent, setMessage]);
  const bulkMarkEmpty = useCallback(async (imageIds) => {
    setNavigationBusy(true);
    try {
      await saveCurrent();
      const latestCurrent = currentImageRef.current;
      const selectedImageIds = new Set(imageIds.map(String));
      const expectedRevisions = Object.fromEntries(
        images
          .filter((image) => selectedImageIds.has(String(image.id)))
          .map((image) => [String(image.id), Number(image.revision || 0)])
      );
      if (latestCurrent && selectedImageIds.has(String(latestCurrent.id))) {
        expectedRevisions[String(latestCurrent.id)] = Number(latestCurrent.revision || 0);
      }
      const data = await api("/api/images/bulk-no-instances", {
        method: "POST",
        body: JSON.stringify({ image_ids: imageIds, expected_revisions: expectedRevisions }),
      });
      const updates = new Map(data.images.map((image) => [String(image.id), image]));
      const mergeUpdates = (items) => items.map((image) => updates.get(String(image.id)) || image);
      setImages(mergeUpdates);
      setProject((current) => current ? { ...current, images: mergeUpdates(current.images || []) } : current);
      const updatedCurrent = latestCurrent && updates.get(String(latestCurrent.id));
      if (updatedCurrent) {
        clearTimeout(saveTimerRef.current);
        currentImageRef.current = updatedCurrent;
        instancesRef.current = [];
        masksRef.current = new Map();
        historyRef.current = [];
        redoRef.current = [];
        setHistoryState({ undo: 0, redo: 0 });
        dirtyRef.current = false;
        setCurrentImage(updatedCurrent);
        setInstances([]);
        setSelectedId(null);
        setPrompts(blankPrompts());
        setProposals([]);
        setProposalPrompts(null);
        setProposalCanvas(null);
        setBoxProposals([]);
        setMaskRevision((value) => value + 1);
        setSaveState("saved");
      }
      setSummary(data.summary);
      setMessage(`${data.selected_count} image${data.selected_count === 1 ? "" : "s"} marked as valid No instances (${data.changed_count} changed).`, "success");
      return true;
    } catch (error) {
      setMessage(`Bulk No instances failed: ${error.message}`, "error");
      return false;
    } finally {
      setNavigationBusy(false);
    }
  }, [images, saveCurrent, setMessage]);
  const changeProject = useCallback(async () => {
    if (!window.confirm("Save this project and return to the folder picker? Your project and annotations stay on disk.")) return;
    try {
      await saveCurrent();
      await api("/api/project/close", { method: "POST", body: "{}" });
      loadTokenRef.current += 1;
      clearTimeout(saveTimerRef.current);
      masksRef.current = new Map();
      historyRef.current = [];
      redoRef.current = [];
      setHistoryState({ undo: 0, redo: 0 });
      dirtyRef.current = false;
      setProject(null);
      setImages([]);
      setSummary(null);
      setCurrentImage(null);
      setInstances([]);
      setSelectedId(null);
      setCurrentIndex(0);
      setSaveState("saved");
      setBoxProposals([]);
      setSegmentationClassUuid("");
      setImagesOpen(false); setPropertiesOpen(false); setPage("annotation");
      setMessage("Choose videos or open an existing Data Labeller project.", "success");
    } catch (error) {
      setMessage(`Could not change project: ${error.message}`, "error");
    }
  }, [saveCurrent, setMessage]);

  const runOperation = useCallback(async (name, operation) => {
    setMessage(`${name}…`); try { const result = await operation(); setMessage(`${name} complete.\n${JSON.stringify(result, null, 2)}`, "success"); return result; } catch (error) { setMessage(`${name} failed: ${error.message}`, "error"); return null; }
  }, [setMessage]);
  const refreshProject = useCallback(async () => {
    const state = await api("/api/state");
    if (!state.opened) return;
    setProject(state.project); setImages(state.project.images || []); setSummary(state.summary);
    setSegmentationClassUuid((current) => state.project.segmentation_classes?.some((item) => item.uuid === current) ? current : (state.project.segmentation_classes?.[0]?.uuid || ""));
    setCurrentImage((current) => current ? { ...current, ...(state.project.images || []).find((item) => String(item.id) === String(current.id)) } : current);
  }, []);
  const toggleImages = useCallback(async () => {
    if (imagesOpen) {
      setImagesOpen(false);
      return;
    }
    setNavigationBusy(true);
    try {
      await saveCurrent();
      const data = await api("/api/images");
      const activeId = currentImageRef.current?.id;
      const authoritativeCurrent = activeId
        ? data.images.find((image) => String(image.id) === String(activeId))
        : null;
      if (projectRef.current) {
        projectRef.current = { ...projectRef.current, images: data.images || [], last_image_id: data.last_image_id };
      }
      setImages(data.images || []);
      setProject((current) => current ? { ...current, images: data.images || [], last_image_id: data.last_image_id } : current);
      if (authoritativeCurrent) {
        currentImageRef.current = { ...currentImageRef.current, ...authoritativeCurrent };
        setCurrentImage((current) => current ? { ...current, ...authoritativeCurrent } : current);
      }
      setImagesOpen(true);
    } catch (error) {
      setMessage(`Could not refresh the image list: ${error.message}`, "error");
    } finally {
      setNavigationBusy(false);
    }
  }, [imagesOpen, saveCurrent, setMessage]);
  const loadProvider = useCallback(async () => {
    setProposalBusy(true);
    setProvider((current) => ({ ...current, state: "loading", error: null }));
    setMessage("Loading the SAM initial-mask model. The first load can take a minute; manual tools remain available.");
    try {
      const status = await api("/api/segmentation/load", { method: "POST", body: "{}" });
      setProvider(status);
      setMessage(`${status.provider === "sam2" ? "SAM 2.1" : status.provider === "sam3" ? "SAM 3" : "Segmentation assistant"} is ready on ${status.device || "this computer"}.`, "success");
    } catch (error) {
      setMessage(`Could not load the SAM assistant: ${error.message} Manual masks are still available.`, "error");
      api("/api/segmentation/status").then(setProvider).catch(() => {});
    } finally {
      setProposalBusy(false);
    }
  }, [setMessage]);
  const loadDetector = useCallback(async () => {
    setDetectionBusy(true);
    setDetector((current) => ({ ...current, state: "loading", error: null }));
    setMessage("Loading the AI box detector. The first load may download and cache model files.");
    try {
      const status = await api("/api/detection/load", { method: "POST", body: "{}" });
      setDetector(status);
      setMessage(`AI box detector ready on ${status.device || "this computer"}.`, "success");
    } catch (error) {
      setMessage(`Could not load the AI box detector: ${error.message}`, "error");
      api("/api/detection/status").then(setDetector).catch(() => {});
    } finally {
      setDetectionBusy(false);
    }
  }, [setMessage]);
  const predictBoxes = useCallback(async () => {
    if (!currentImageRef.current) return;
    const labels = parseDetectionLabels(detectionPrompt);
    if (!labels.length) return setMessage("Enter at least one object description, such as orange or naartjie.", "error");
    setDetectionBusy(true);
    const requestId = crypto.randomUUID();
    detectionRequestIdRef.current = requestId;
    try {
      const data = await api("/api/detection/predict", { method: "POST", body: JSON.stringify({
        image_id: currentImageRef.current.id,
        request_id: requestId,
        labels,
        confidence: detectionConfidence,
        text_threshold: 0.25,
        max_detections: detectionMaxDetections
      }) });
      if (detectionRequestIdRef.current !== data.request_id || String(currentImageRef.current?.id) !== String(data.image_id)) return;
      setDetector(data.status);
      setBoxProposals(data.proposals || []);
      setMessage(
        data.proposals?.length
          ? `Found ${data.proposals.length} suggested box${data.proposals.length === 1 ? "" : "es"}. Review the dashed previews before using them.`
          : "No boxes matched. Try a simpler prompt or lower the confidence.",
        data.proposals?.length ? "success" : "warning"
      );
    } catch (error) {
      setMessage(`AI box search failed: ${error.message}`, "error");
      api("/api/detection/status").then(setDetector).catch(() => {});
    } finally {
      setDetectionBusy(false);
    }
  }, [detectionConfidence, detectionMaxDetections, detectionPrompt, setMessage]);
  const acceptDetectionBoxes = useCallback((accepted) => {
    if (!accepted.length || !currentImageRef.current) return;
    const classItem = projectRef.current.detection_classes.find((item) => item.uuid === detectionClassUuid) || projectRef.current.detection_classes[0];
    const labels = parseDetectionLabels(detectionPrompt);
    const provenance = detector?.provider === "grounding_dino" ? "grounding_dino" : "mock";
    pushHistory();
    const additions = accepted.map((proposal) => {
      const id = crypto.randomUUID();
      return {
        id,
        uuid: id,
        type: "fruit_box",
        annotation_kind: "fruit_box",
        source: provenance,
        provenance,
        prompt_metadata: detectionPromptMetadata(proposal, labels),
        review_status: "draft",
        visible: true,
        locked: false,
        notes: "",
        parent_fruit_uuid: null,
        parent_fruit_instance_id: null,
        class_uuid: classItem.uuid,
        class_name: classItem.name,
        export_index: classItem.export_index,
        training_class_id: classItem.export_index,
        category_id: classItem.export_index,
        display_label: classItem.display_name,
        name: classItem.display_name,
        color: classItem.color,
        bbox: proposal.bbox.map(Number)
      };
    });
    setInstances((current) => { const next = [...current, ...additions]; instancesRef.current = next; return next; });
    setSelectedId(additions.at(-1).id);
    applyCurrentImagePatch({ status: "draft", reviewed: false });
    const acceptedIds = new Set(accepted.map((proposal) => proposal.id));
    setBoxProposals((current) => current.filter((proposal) => !acceptedIds.has(proposal.id)));
    setTool("select");
    setPropertiesOpen(true);
    markDirty();
  }, [applyCurrentImagePatch, detectionClassUuid, detectionPrompt, detector, markDirty, pushHistory]);
  const predict = useCallback(async () => {
    if (!currentImage) return;
    setProposalBusy(true); const requestId = crypto.randomUUID(); requestIdRef.current = requestId;
    try {
      const requestPrompts = { ...prompts, text: provider?.capabilities?.text === false ? "" : prompts.text };
      const data = await api("/api/segmentation/predict", { method: "POST", body: JSON.stringify({ image_id: currentImage.id, request_id: requestId, ...requestPrompts }) });
      if (requestIdRef.current !== data.request_id || String(currentImageRef.current.id) !== String(data.image_id)) return;
      setProvider(data.status); setProposals(data.proposals); setProposalIndex(0);
      setProposalPrompts(requestPrompts);
      setProposalCanvas(data.proposals[0] ? await maskCanvasFromSource(data.proposals[0].mask_data, currentImage.width, currentImage.height) : null);
      setMessage(`SAM created ${data.proposals.length} mask option${data.proposals.length === 1 ? "" : "s"}. Review the cyan preview before accepting it.`, "success");
    } catch (error) {
      setMessage(`Proposal failed: ${error.message}`, "error");
      api("/api/segmentation/status").then(setProvider).catch(() => {});
    }
    finally { setProposalBusy(false); }
  }, [currentImage, prompts, provider, setMessage]);
  useEffect(() => {
    const selected = proposals[proposalIndex];
    if (!selected || !currentImage) { if (!selected) setProposalCanvas(null); return; }
    maskCanvasFromSource(selected.mask_data, currentImage.width, currentImage.height).then(setProposalCanvas).catch((error) => setMessage(error.message, "error"));
  }, [proposalIndex]); // eslint-disable-line react-hooks/exhaustive-deps
  const acceptProposal = useCallback(() => {
    if (!proposalCanvas) return;
    const provenance = ["sam2", "sam3"].includes(provider?.provider) ? provider.provider : "mock";
    const acceptedPrompts = proposalPrompts || prompts;
    newInstance("defect_mask", proposalCanvas, provenance, { points: acceptedPrompts.points, box: acceptedPrompts.box, text: provider?.capabilities?.text ? acceptedPrompts.text : "", score: proposals[proposalIndex]?.score, accepted_as_draft: true }, null, segmentationClassUuid);
    setProposals([]); setProposalPrompts(null); setProposalCanvas(null); setPrompts(blankPrompts());
    setMessage("SAM mask accepted as an editable draft. Refine the boundary with Add or Erase, then mark the image done.", "success");
  }, [newInstance, prompts, proposalCanvas, proposalIndex, proposalPrompts, proposals, provider, segmentationClassUuid, setMessage]);
  const rejectProposal = useCallback(() => { setProposals([]); setProposalPrompts(null); setProposalCanvas(null); }, []);
  const deriveBox = useCallback(async (foregroundUuid) => {
    await saveCurrent();
    await runOperation("Derive fruit box", () => api(`/api/images/${currentImageRef.current.id}/derive-fruit-box`, { method: "POST", body: JSON.stringify({ foreground_uuid: foregroundUuid, detection_class_uuid: projectRef.current.detection_classes[0].uuid }) }));
    await loadCurrent();
  }, [loadCurrent, runOperation, saveCurrent]);

  useEffect(() => {
    const handler = (event) => {
      if (page !== "annotation") return;
      const brushRangeHasFocus = (event.key === "[" || event.key === "]") && event.target?.matches?.('input[type="range"]');
      if ((shouldIgnoreShortcut(event.target) && !brushRangeHasFocus) || helpOpen || imageEditorOpen || datasetOpen || splitOpen) return;
      const key = event.key.toLowerCase(); const command = event.ctrlKey || event.metaKey;
      if (command && key === "z") { event.preventDefault(); event.shiftKey ? redo() : undo(); }
      else if (command && key === "y") { event.preventDefault(); redo(); }
      else if (command && key === "s") { event.preventDefault(); saveCurrent(); }
      else if (command) return;
      else if (key === "b" && ["detection", "mixed"].includes(projectRef.current?.annotation_mode)) setTool("draw_box");
      else if (key === "a" && ["segmentation", "mixed"].includes(projectRef.current?.annotation_mode)) setTool("add"); else if (key === "e" && ["segmentation", "mixed"].includes(projectRef.current?.annotation_mode)) setTool("erase");
      else if (event.key === "[") { event.preventDefault(); setBrushSize((value) => Math.max(2, value - 2)); } else if (event.key === "]") { event.preventDefault(); setBrushSize((value) => Math.min(200, value + 2)); }
      else if (key === "y" && !navigationBusy && !event.repeat) { event.preventDefault(); markDone(); }
      else if (key === "n" && projectRef.current?.annotation_mode !== "classification" && !navigationBusy && !event.repeat) { event.preventDefault(); markEmpty(); }
      else if (event.key === "ArrowLeft" && !navigationBusy && !event.repeat) { event.preventDefault(); changeImage(currentIndexRef.current - 1); }
      else if (event.key === "ArrowRight" && !navigationBusy && !event.repeat) { event.preventDefault(); changeImage(currentIndexRef.current + 1); }
      else if (["delete", "backspace"].includes(key) && selectedId) { event.preventDefault(); deleteSelected(); }
      else if (/^[1-9]$/.test(event.key)) {
        if (projectRef.current?.annotation_mode === "classification") {
          const item = projectRef.current.classification_classes?.[Number(event.key) - 1];
          if (item) updateImage({ classification_class_uuid: item.uuid, classification_class_name: item.name, classification_export_index: item.export_index, status: "draft", reviewed: false, excluded: false });
        } else {
          const selected = instancesRef.current.find((item) => String(item.id) === String(selectedId));
          const classes = isFruit(selected) ? projectRef.current.detection_classes : projectRef.current.segmentation_classes;
          const item = classes?.[Number(event.key) - 1];
          if (selected && item && selected.annotation_kind !== "foreground_mask") updateInstance({ class_uuid: item.uuid, export_index: item.export_index, training_class_id: item.export_index, category_id: item.export_index, name: item.display_name, display_label: item.display_name, color: item.color });
        }
      }
      else if (key === "s") { event.preventDefault(); saveCurrent(); } else if (key === "f") fitRef.current();
    };
    window.addEventListener("keydown", handler); return () => window.removeEventListener("keydown", handler);
  }, [changeImage, datasetOpen, deleteSelected, helpOpen, imageEditorOpen, markDone, markEmpty, navigationBusy, page, redo, saveCurrent, selectedId, splitOpen, undo, updateImage, updateInstance]);
  useEffect(() => { const before = (event) => { if (["dirty", "saving", "error"].includes(saveState)) { event.preventDefault(); event.returnValue = ""; } }; window.addEventListener("beforeunload", before); return () => window.removeEventListener("beforeunload", before); }, [saveState]);

  const providerProps = { provider, classes: project?.segmentation_classes || [], classUuid: segmentationClassUuid, setClassUuid: setSegmentationClassUuid, prompts, setPrompts, tool, setTool, proposals, proposalIndex, setProposalIndex, onLoad: loadProvider, onPredict: predict, onAccept: acceptProposal, onReject: rejectProposal, busy: proposalBusy };
  const detectionProps = {
    provider: detector,
    prompt: detectionPrompt,
    setPrompt: setDetectionPrompt,
    confidence: detectionConfidence,
    setConfidence: setDetectionConfidence,
    maxDetections: detectionMaxDetections,
    setMaxDetections: setDetectionMaxDetections,
    classUuid: detectionClassUuid,
    setClassUuid: setDetectionClassUuid,
    proposals: boxProposals,
    busy: detectionBusy,
    onLoad: loadDetector,
    onPredict: predictBoxes,
    onAccept: (proposal) => acceptDetectionBoxes([proposal]),
    onAcceptAll: () => acceptDetectionBoxes(boxProposals),
    onReject: () => setBoxProposals([])
  };
  const selectAnnotation = useCallback((id) => { setSelectedId(id); if (id) setPropertiesOpen(true); }, []);
  return <>
    <TopBar project={project} currentIndex={currentIndex} imageCount={images.length} saveState={saveState} navigationBusy={navigationBusy} page={page} onPage={setPage} onChangeProject={changeProject} onSave={saveCurrent} onValidate={async () => { const result = await runOperation("Validation", async () => { await saveCurrent(); return api("/api/validate", { method: "POST", body: "{}" }); }); if (result) await refreshProject(); }} onSplit={async () => { await saveCurrent(); setSplitOpen(true); }} onDataset={async () => { await saveCurrent(); setDatasetOpen(true); }} onHelp={() => setHelpOpen(true)} onImages={toggleImages} onPrevious={() => changeImage(currentIndex - 1)} onNext={() => changeImage(currentIndex + 1)} onProperties={() => setPropertiesOpen((value) => !value)} onHarvest={openHarvest} onBackToHarvest={openHarvest} />
    {page === "deblur" && <SyntheticBlurDataset setMessage={setMessage} />}
    {page === "advanced" && <ProjectLauncher onOpen={openLoadedProject} setMessage={setMessage} />}
    {page === "harvest" && <HarvestWorkspace currentClasses={project?.segmentation_classes || []} onOpenProject={openLoadedProject} onExit={() => setPage("annotation")} setMessage={setMessage} />}
    {page === "annotation" && !project && <HomeLauncher onOpen={openLoadedProject} setMessage={setMessage} onAdvanced={() => setPage("advanced")} onHarvest={() => setPage("harvest")} />}
    {page === "annotation" && project && currentImage && <main className="workspace canvas-first">
      {imagesOpen && <ImageSidebar images={images} currentImage={currentImage} currentIndex={currentIndex} navigationBusy={navigationBusy} onClose={() => setImagesOpen(false)} onSelect={(id) => { changeImage(id); setImagesOpen(false); }} onPrev={() => changeImage(currentIndex - 1)} onNext={() => changeImage(currentIndex + 1)} onBulkNoInstances={bulkMarkEmpty} />}
      <AnnotationCanvas image={currentImage} instances={instances} masksRef={masksRef} maskRevision={maskRevision} selectedId={selectedId} setSelectedId={selectAnnotation} tool={tool} setTool={setTool} brushSize={brushSize} setBrushSize={setBrushSize} opacity={opacity} setOpacity={setOpacity} showMasks={showMasks} setShowMasks={setShowMasks} prompts={prompts} setPrompts={setPrompts} proposalCanvas={proposalCanvas} boxProposals={boxProposals} pushHistory={pushHistory} markDirty={markDirty} undo={undo} redo={redo} canUndo={historyState.undo > 0} canRedo={historyState.redo > 0} clearSelectedMask={clearSelectedMask} deleteSelected={deleteSelected} registerFit={(fit) => { fitRef.current = fit; }} annotationMode={project.annotation_mode || "mixed"} onEditImage={() => setImageEditorOpen(true)} onOpenDetectionAssist={() => setPropertiesOpen(true)} onOpenSegmentationAssist={() => { setPropertiesOpen(true); setTool(provider?.capabilities?.positive_points === false ? "sam_box" : "sam_positive"); }} onCreateBox={(bbox) => newInstance("fruit_box", null, "manual", {}, bbox)} onPreviewBox={previewBox} onCreateMask={() => newInstance("defect_mask", null, "manual", {}, null, segmentationClassUuid)} />
      {propertiesOpen && <Inspector project={project} image={currentImage} updateImage={updateImage} instances={instances} selectedId={selectedId} setSelectedId={selectAnnotation} updateInstance={updateInstance} newInstance={newInstance} onDeleteSelected={deleteSelected} onClearSelectedMask={clearSelectedMask} onClearAnnotations={clearAnnotations} onDeriveBox={deriveBox} providerProps={providerProps} detectionProps={detectionProps} summary={summary} onClose={() => setPropertiesOpen(false)} onMarkDone={markDone} onMarkEmpty={markEmpty} />}
    </main>}
    {page === "annotation" && project && !currentImage && <section className="empty-state"><CircleAlert size={24} /> No supported images are loaded in this project.</section>}
    {message && <div className={`toast ${message.type}`}><pre>{message.text}</pre><IconButton title="Dismiss" onClick={() => setMessageState(null)}><X size={16} /></IconButton></div>}
    {helpOpen && <HelpModal onClose={() => setHelpOpen(false)} />}
    {imageEditorOpen && currentImage && <ImageEditorModal image={currentImage} busy={imageEditBusy} onClose={() => setImageEditorOpen(false)} onTransform={(operation) => applyImageEdit({ operation })} onErase={(eraseMaskData) => applyImageEdit({ eraseMaskData })} />}
    {datasetOpen && <DatasetModal project={project} onClose={() => setDatasetOpen(false)} runOperation={runOperation} onComplete={refreshProject} />}
    {splitOpen && <SplitModal project={project} onClose={() => setSplitOpen(false)} runOperation={runOperation} onComplete={refreshProject} />}
  </>;
}

createRoot(document.getElementById("root")).render(<App />);
