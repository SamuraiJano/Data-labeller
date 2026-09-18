export const SYNTHETIC_LIMITS = Object.freeze({
  minObjectsPerImage: 1,
  maxObjectsPerImage: 300,
  minTotalImages: 2,
  maxTotalImages: 100000,
  maxRotationDegrees: 360,
  maxOverlapPercent: 80
});

const inRange = (value, minimum, maximum) => Number.isFinite(value) && value >= minimum && value <= maximum;

export function validateSyntheticSettings(settings) {
  const objects = Number(settings.objectsPerImage);
  if (!Number.isInteger(objects) || !inRange(objects, SYNTHETIC_LIMITS.minObjectsPerImage, SYNTHETIC_LIMITS.maxObjectsPerImage)) {
    return `Maximum objects per image must be a whole number from 1 to ${SYNTHETIC_LIMITS.maxObjectsPerImage}. This is the fruit pasted into one composite, not the number of foreground files; every foreground file is used automatically before reuse.`;
  }

  const total = Number(settings.totalImages);
  if (!Number.isInteger(total) || !inRange(total, SYNTHETIC_LIMITS.minTotalImages, SYNTHETIC_LIMITS.maxTotalImages)) {
    return "Total images must be a whole number from 2 to 100,000.";
  }

  if (!Number.isInteger(Number(settings.seed))) return "Random seed must be a whole number.";
  const minimumScale = Number(settings.minScale);
  const maximumScale = Number(settings.maxScale);
  if (!Number.isFinite(minimumScale) || minimumScale <= 0
      || !Number.isFinite(maximumScale) || maximumScale <= 0) {
    return "Foreground scale percentages must be greater than 0.";
  }
  if (minimumScale > maximumScale) return "Minimum scale cannot be greater than maximum scale.";

  if (!inRange(Number(settings.maxRotation), 0, SYNTHETIC_LIMITS.maxRotationDegrees)) {
    return "Rotation range must be between 0 and 360 degrees.";
  }
  if (!inRange(Number(settings.overlapChance), 0, 100)) {
    return "Overlap chance must be between 0% and 100%.";
  }
  if (!inRange(Number(settings.maxOverlap), Number.MIN_VALUE, SYNTHETIC_LIMITS.maxOverlapPercent)) {
    return "Maximum overlap per fruit must be greater than 0% and no more than 80%.";
  }
  return null;
}
