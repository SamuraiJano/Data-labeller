import test from "node:test";
import assert from "node:assert/strict";
import { boxContainsPoint, boxHandlePoints, clampImagePoint, fitImageTransform, hitBoxHandle, imageToScreen, interpolateStroke, moveBox, resizeBox, screenToImage, shouldIgnoreShortcut, zoomTransformAtPoint } from "./canvasMath.js";

test("coordinate mapping round trips through pan and zoom", () => {
  for (const transform of [
    { scale: 0.25, offsetX: 19, offsetY: -4 },
    { scale: 1, offsetX: 0, offsetY: 0 },
    { scale: 7.5, offsetX: -830, offsetY: 221 }
  ]) {
    const point = [637.25, 311.75];
    const roundTrip = screenToImage(imageToScreen(point, transform), transform);
    assert.ok(Math.abs(roundTrip[0] - point[0]) < 1e-9);
    assert.ok(Math.abs(roundTrip[1] - point[1]) < 1e-9);
  }
});

test("fit and zoom keep the image centered and the pointer anchored", () => {
  const fitted = fitImageTransform(1200, 800, 1920, 1080);
  assert.equal(fitted.scale, 0.6);
  assert.deepEqual(fitted, { scale: 0.6, offsetX: 24, offsetY: 76 });

  const pointer = [420, 310];
  const imagePointBefore = screenToImage(pointer, fitted);
  const zoomed = zoomTransformAtPoint(fitted, pointer, 2, fitted.scale * 0.25, fitted.scale * 16);
  assert.equal(zoomed.scale, 1.2);
  assert.deepEqual(screenToImage(pointer, zoomed), imagePointBefore);

  assert.equal(zoomTransformAtPoint(zoomed, pointer, 100, fitted.scale * 0.25, fitted.scale * 16).scale, fitted.scale * 16);
  assert.equal(zoomTransformAtPoint(fitted, pointer, 0.001, fitted.scale * 0.25, fitted.scale * 16).scale, fitted.scale * 0.25);
});

test("stroke interpolation prevents gaps and clamps to image pixels", () => {
  const points = interpolateStroke([[1, 1], [21, 1], [21, 19]], 3);
  assert.ok(points.length > 10);
  for (let index = 1; index < points.length; index += 1) {
    assert.ok(Math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1]) <= 3.01);
  }
  assert.deepEqual(clampImagePoint([-5, 120], 80, 60), [0, 59]);
});

test("shortcuts ignore text fields", () => {
  assert.equal(shouldIgnoreShortcut({ tagName: "INPUT", isContentEditable: false }), true);
  assert.equal(shouldIgnoreShortcut({ tagName: "CANVAS", isContentEditable: false }), false);
});

test("bounding boxes move and resize without leaving image bounds", () => {
  assert.deepEqual(moveBox([10, 10, 40, 30], [-50, 100], 100, 80), [0, 50, 40, 30]);
  assert.deepEqual(moveBox([10, 10, 40, 30], [90, -30], 100, 80), [60, 0, 40, 30]);
  assert.deepEqual(resizeBox([10, 10, 40, 30], "nw", [-12, -8], 100, 80), [0, 0, 50, 40]);
  assert.deepEqual(resizeBox([10, 10, 40, 30], "se", [140, 120], 100, 80), [10, 10, 90, 70]);
  assert.deepEqual(resizeBox([10, 10, 40, 30], "nw", [49, 39], 100, 80), [48, 38, 2, 2]);
});

test("bounding boxes expose eight generous resize targets", () => {
  const box = [10, 20, 80, 60];
  assert.deepEqual(boxHandlePoints(box), [
    ["nw", 10, 20], ["n", 50, 20], ["ne", 90, 20], ["e", 90, 50],
    ["se", 90, 80], ["s", 50, 80], ["sw", 10, 80], ["w", 10, 50]
  ]);
  assert.equal(hitBoxHandle(box, [7, 17], 1), "nw");
  assert.equal(hitBoxHandle(box, [93, 50], 1), "e");
  assert.equal(hitBoxHandle(box, [50, 23], 2), "n");
  assert.equal(boxContainsPoint(box, [45, 45]), true);
  assert.equal(boxContainsPoint(box, [2, 45]), false);
});
