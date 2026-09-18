export function updateImageSelection(selectedIds, imageId, checked, shiftKey, lastSelectedId, orderedIds) {
  const next = new Set(selectedIds);
  const currentIndex = orderedIds.findIndex((id) => String(id) === String(imageId));
  const previousIndex = orderedIds.findIndex((id) => String(id) === String(lastSelectedId));
  const targets = shiftKey && previousIndex >= 0 && currentIndex >= 0
    ? orderedIds.slice(Math.min(previousIndex, currentIndex), Math.max(previousIndex, currentIndex) + 1)
    : [imageId];
  targets.forEach((id) => checked ? next.add(String(id)) : next.delete(String(id)));
  return next;
}

export function retainExistingSelections(selectedIds, images) {
  const existing = new Set(images.map((image) => String(image.id)));
  return new Set([...selectedIds].filter((id) => existing.has(String(id))));
}
