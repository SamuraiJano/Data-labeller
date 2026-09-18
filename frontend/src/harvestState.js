export const HARVEST_STEP_MS = 250;

export function snapHarvestTime(timeMs, durationMs = Number.POSITIVE_INFINITY) {
  const snapped = Math.max(0, Math.round(Number(timeMs) / HARVEST_STEP_MS) * HARVEST_STEP_MS);
  const last = Number.isFinite(Number(durationMs))
    ? Math.max(0, Math.floor((Number(durationMs) - 1e-6) / HARVEST_STEP_MS) * HARVEST_STEP_MS)
    : snapped;
  return Math.min(snapped, last);
}

export function containedFrameRect(containerWidth, containerHeight, frameWidth, frameHeight) {
  const safeContainerWidth = Math.max(0, Number(containerWidth) || 0);
  const safeContainerHeight = Math.max(0, Number(containerHeight) || 0);
  const safeFrameWidth = Math.max(1, Number(frameWidth) || 1);
  const safeFrameHeight = Math.max(1, Number(frameHeight) || 1);
  const scale = Math.min(safeContainerWidth / safeFrameWidth, safeContainerHeight / safeFrameHeight);
  const width = safeFrameWidth * scale;
  const height = safeFrameHeight * scale;
  return {
    x: (safeContainerWidth - width) / 2,
    y: (safeContainerHeight - height) / 2,
    width,
    height,
  };
}

export function activeHarvestTracks(session) {
  if (!session) return [];
  return (session.tracks || []).filter((track) => track.status === "active" && track.video_id === session.current_video_id);
}

export function openHarvestTracks(session) {
  if (!session) return [];
  return (session.tracks || []).filter((track) => ["active", "needs_mask"].includes(track.status) && track.video_id === session.current_video_id);
}

export function visibleHarvestMasks(session, videoId, timeMs) {
  if (!session || !videoId) return [];
  const playhead = Number(timeMs);
  const sampleTimeMs = snapHarvestTime(playhead);
  return (session.tracks || []).filter((track) => (
    track.video_id === videoId
    && Boolean(track.mask_url)
    && Number.isFinite(playhead)
    && Number(track.current_time_ms) === sampleTimeMs
  ));
}

export function harvestModeAfterLoad(session) {
  return openHarvestTracks(session).length ? "track" : "scan";
}
