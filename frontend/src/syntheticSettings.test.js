import test from "node:test";
import assert from "node:assert/strict";
import { validateSyntheticSettings } from "./syntheticSettings.js";

const validSettings = {
  objectsPerImage: 6,
  totalImages: 100,
  seed: 42,
  minScale: 10,
  maxScale: 17,
  maxRotation: 360,
  overlapChance: 30,
  maxOverlap: 20
};

test("synthetic settings accept the documented defaults", () => {
  assert.equal(validateSyntheticSettings(validSettings), null);
});

test("synthetic settings accept foreground scales above 100 percent", () => {
  assert.equal(validateSyntheticSettings({ ...validSettings, minScale: 125, maxScale: 250 }), null);
});

test("synthetic settings accept 216 objects and explain the 300-object limit", () => {
  assert.equal(validateSyntheticSettings({ ...validSettings, objectsPerImage: 216 }), null);
  const message = validateSyntheticSettings({ ...validSettings, objectsPerImage: 301 });
  assert.match(message, /whole number from 1 to 300/);
  assert.match(message, /not the number of foreground files/);
});
