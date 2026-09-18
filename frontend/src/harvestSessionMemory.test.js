import test from "node:test";
import assert from "node:assert/strict";
import { forgetRememberedHarvestSession, HARVEST_SESSION_STORAGE_KEY, readRememberedHarvestSession, rememberHarvestSession, shouldResumeHarvestOnLaunch } from "./harvestSessionMemory.js";

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
    values
  };
}

test("the last harvest session can be remembered across launches", () => {
  const storage = memoryStorage();
  assert.equal(shouldResumeHarvestOnLaunch(storage), false);
  assert.equal(rememberHarvestSession(storage, {
    sessionDir: "C:/datasets/harvest-one",
    sessionId: "session-1",
    name: "Defective naartjies",
    updatedAt: "2026-09-13T10:00:00Z"
  }), true);
  assert.deepEqual(readRememberedHarvestSession(storage), {
    sessionDir: "C:/datasets/harvest-one",
    sessionId: "session-1",
    name: "Defective naartjies",
    updatedAt: "2026-09-13T10:00:00Z"
  });
  assert.equal(shouldResumeHarvestOnLaunch(storage), true);
});

test("finished or invalid remembered sessions can be forgotten safely", () => {
  const storage = memoryStorage();
  storage.setItem(HARVEST_SESSION_STORAGE_KEY, "not-json");
  assert.equal(readRememberedHarvestSession(storage), null);
  storage.setItem(HARVEST_SESSION_STORAGE_KEY, JSON.stringify({ sessionDir: "C:/datasets/old" }));
  assert.equal(forgetRememberedHarvestSession(storage), true);
  assert.equal(readRememberedHarvestSession(storage), null);
});
