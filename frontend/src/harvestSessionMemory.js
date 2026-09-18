export const HARVEST_SESSION_STORAGE_KEY = "data-labeller.last-harvest-session.v1";

export function readRememberedHarvestSession(storage) {
  if (!storage?.getItem) return null;
  try {
    const raw = storage.getItem(HARVEST_SESSION_STORAGE_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw);
    if (!value || typeof value.sessionDir !== "string" || !value.sessionDir.trim()) return null;
    return {
      sessionDir: value.sessionDir,
      sessionId: typeof value.sessionId === "string" ? value.sessionId : "",
      name: typeof value.name === "string" ? value.name : "",
      updatedAt: typeof value.updatedAt === "string" ? value.updatedAt : ""
    };
  } catch {
    return null;
  }
}

export function rememberHarvestSession(storage, { sessionDir, sessionId = "", name = "", updatedAt = "" }) {
  if (!storage?.setItem || typeof sessionDir !== "string" || !sessionDir.trim()) return false;
  try {
    storage.setItem(HARVEST_SESSION_STORAGE_KEY, JSON.stringify({ sessionDir, sessionId, name, updatedAt }));
    return true;
  } catch {
    return false;
  }
}

export function forgetRememberedHarvestSession(storage) {
  if (!storage?.removeItem) return false;
  try {
    storage.removeItem(HARVEST_SESSION_STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}

export function shouldResumeHarvestOnLaunch(storage) {
  return Boolean(readRememberedHarvestSession(storage));
}
