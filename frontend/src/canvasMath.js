export function imageToScreen([x, y], transform) {
  return [x * transform.scale + transform.offsetX, y * transform.scale + transform.offsetY];
}

export function screenToImage([x, y], transform) {
  if (!Number.isFinite(transform.scale) || transform.scale <= 0) throw new Error("Scale must be positive and finite");
  return [(x - transform.offsetX) / transform.scale, (y - transform.offsetY) / transform.scale];
}

export function fitImageTransform(viewportWidth, viewportHeight, imageWidth, imageHeight, padding = 0.96) {
  const values = [viewportWidth, viewportHeight, imageWidth, imageHeight, padding].map(Number);
  if (values.some((value) => !Number.isFinite(value) || value <= 0)) throw new Error("Fit dimensions and padding must be positive and finite");
  const [viewWidth, viewHeight, sourceWidth, sourceHeight, inset] = values;
  const scale = Math.min(viewWidth / sourceWidth, viewHeight / sourceHeight) * inset;
  return {
    scale,
    offsetX: (viewWidth - sourceWidth * scale) / 2,
    offsetY: (viewHeight - sourceHeight * scale) / 2
  };
}

export function zoomTransformAtPoint(transform, screenPoint, multiplier, minimumScale, maximumScale) {
  const factor = Number(multiplier);
  const minimum = Number(minimumScale);
  const maximum = Number(maximumScale);
  if (!Number.isFinite(factor) || factor <= 0) throw new Error("Zoom multiplier must be positive and finite");
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum <= 0 || maximum < minimum) throw new Error("Zoom limits are invalid");
  const anchor = screenToImage(screenPoint, transform);
  const scale = Math.max(minimum, Math.min(maximum, transform.scale * factor));
  return {
    scale,
    offsetX: screenPoint[0] - anchor[0] * scale,
    offsetY: screenPoint[1] - anchor[1] * scale
  };
}

export function clampImagePoint([x, y], width, height) {
  return [Math.max(0, Math.min(width - 1, x)), Math.max(0, Math.min(height - 1, y))];
}

export function interpolateStroke(samples, spacing) {
  if (!samples.length) return [];
  const step = Math.max(0.25, Number(spacing) || 1);
  const result = [samples[0]];
  for (let index = 1; index < samples.length; index += 1) {
    const start = samples[index - 1];
    const end = samples[index];
    const distance = Math.hypot(end[0] - start[0], end[1] - start[1]);
    const segments = Math.max(1, Math.ceil(distance / step));
    for (let part = 1; part <= segments; part += 1) {
      const ratio = part / segments;
      result.push([start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio]);
    }
  }
  return result;
}

export function moveBox([x, y, width, height], [deltaX, deltaY], imageWidth, imageHeight) {
  return [
    Math.max(0, Math.min(imageWidth - width, x + deltaX)),
    Math.max(0, Math.min(imageHeight - height, y + deltaY)),
    width,
    height
  ];
}

export function resizeBox([x, y, width, height], handle, [pointX, pointY], imageWidth, imageHeight, minSize = 2) {
  const right = x + width;
  const bottom = y + height;
  let left = x;
  let top = y;
  let nextRight = right;
  let nextBottom = bottom;
  if (handle.includes("w")) left = Math.max(0, Math.min(pointX, right - minSize));
  if (handle.includes("e")) nextRight = Math.min(imageWidth, Math.max(pointX, x + minSize));
  if (handle.includes("n")) top = Math.max(0, Math.min(pointY, bottom - minSize));
  if (handle.includes("s")) nextBottom = Math.min(imageHeight, Math.max(pointY, y + minSize));
  return [left, top, nextRight - left, nextBottom - top];
}

export function boxHandlePoints([x, y, width, height]) {
  const right = x + width;
  const bottom = y + height;
  const middleX = x + width / 2;
  const middleY = y + height / 2;
  return [
    ["nw", x, y], ["n", middleX, y], ["ne", right, y],
    ["e", right, middleY], ["se", right, bottom], ["s", middleX, bottom],
    ["sw", x, bottom], ["w", x, middleY]
  ];
}

export function hitBoxHandle(box, [pointX, pointY], scale = 1, hitRadiusPixels = 18) {
  const threshold = hitRadiusPixels / Math.max(Number(scale) || 1, 0.001);
  let nearest = null;
  for (const [handle, handleX, handleY] of boxHandlePoints(box)) {
    const distance = Math.hypot(pointX - handleX, pointY - handleY);
    if (distance <= threshold && (!nearest || distance < nearest.distance)) nearest = { handle, distance };
  }
  return nearest?.handle || null;
}

export function boxContainsPoint([x, y, width, height], [pointX, pointY]) {
  return pointX >= x && pointX <= x + width && pointY >= y && pointY <= y + height;
}

export function shouldIgnoreShortcut(target) {
  if (!target) return false;
  const tag = String(target.tagName || "").toLowerCase();
  return target.isContentEditable || tag === "input" || tag === "textarea" || tag === "select";
}
