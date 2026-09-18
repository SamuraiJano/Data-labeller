export const AUTOSAVE_IDLE_MS = 3000;

export function patchImageCollection(images, imageId, patch) {
  return images.map((image) => String(image.id) === String(imageId) ? { ...image, ...patch } : image);
}

export function initialImageIndex(images, lastImageId) {
  if (!images.length || lastImageId == null || lastImageId === "") return 0;
  const index = images.findIndex((image) => String(image.id) === String(lastImageId));
  return index >= 0 ? index : 0;
}

export function patchProjectImage(project, imageId, patch, makeCurrent = false) {
  if (!project) return project;
  return {
    ...project,
    ...(makeCurrent ? { last_image_id: String(imageId) } : {}),
    images: patchImageCollection(project.images || [], imageId, patch),
  };
}

const EDITABLE_IMAGE_FIELDS = [
  "status",
  "reviewed",
  "excluded",
  "exclusion_reason",
  "fruit_id",
  "view_id",
  "split"
];

export function reconcileSaveResponse({
  localImage,
  localInstances,
  serverImage,
  serverAnnotations,
  hasNewerEdits
}) {
  if (!hasNewerEdits) {
    return { image: serverImage, instances: serverAnnotations, dirty: false };
  }
  const editable = Object.fromEntries(
    EDITABLE_IMAGE_FIELDS
      .filter((field) => Object.hasOwn(localImage || {}, field))
      .map((field) => [field, localImage[field]])
  );
  return {
    image: { ...serverImage, ...editable },
    instances: localInstances,
    dirty: true
  };
}
