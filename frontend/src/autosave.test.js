import test from "node:test";
import assert from "node:assert/strict";
import { AUTOSAVE_IDLE_MS, initialImageIndex, patchImageCollection, patchProjectImage, reconcileSaveResponse } from "./autosave.js";

test("autosave waits for three seconds of inactivity", () => {
  assert.equal(AUTOSAVE_IDLE_MS, 3000);
});

test("an optimistic image status patch updates the matching sidebar row only", () => {
  const images = [
    { id: "image-1", status: "unlabelled", reviewed: false },
    { id: "image-2", status: "unlabelled", reviewed: false }
  ];
  const result = patchImageCollection(images, "image-1", { status: "approved", reviewed: true });
  assert.deepEqual(result[0], { id: "image-1", status: "approved", reviewed: true });
  assert.deepEqual(result[1], images[1]);
});

test("the saved UUID selects the last working image when a project reopens", () => {
  const images = [{ id: "image-1" }, { id: "image-2" }, { id: "image-3" }];
  assert.equal(initialImageIndex(images, "image-2"), 1);
  assert.equal(initialImageIndex(images, "missing-image"), 0);
  assert.equal(initialImageIndex(images, null), 0);
});

test("an image patch keeps project metadata and the sidebar collection synchronized", () => {
  const project = {
    project_name: "Defects",
    last_image_id: "image-1",
    images: [
      { id: "image-1", status: "approved" },
      { id: "image-2", status: "unlabelled" },
    ],
  };
  const result = patchProjectImage(project, "image-2", { status: "approved", reviewed: true }, true);
  assert.equal(result.project_name, "Defects");
  assert.equal(result.last_image_id, "image-2");
  assert.deepEqual(result.images[1], { id: "image-2", status: "approved", reviewed: true });
});

test("an older save response cannot replace newer live box edits", () => {
  const newerBox = { id: "box-1", bbox: [30, 25, 90, 70] };
  const result = reconcileSaveResponse({
    localImage: { id: "image-1", revision: 4, status: "draft" },
    localInstances: [newerBox],
    serverImage: { id: "image-1", revision: 5, status: "unlabelled" },
    serverAnnotations: [{ id: "box-1", bbox: [10, 10, 40, 40] }],
    hasNewerEdits: true
  });
  assert.equal(result.image.revision, 5);
  assert.equal(result.image.status, "draft");
  assert.deepEqual(result.instances, [newerBox]);
  assert.equal(result.dirty, true);
});

test("a current save response adopts normalized server state", () => {
  const serverAnnotations = [{ id: "box-1", bbox: [10, 10, 40, 40], revision: 2 }];
  const result = reconcileSaveResponse({
    localImage: { id: "image-1", revision: 1 },
    localInstances: [],
    serverImage: { id: "image-1", revision: 2 },
    serverAnnotations,
    hasNewerEdits: false
  });
  assert.equal(result.image.revision, 2);
  assert.equal(result.instances, serverAnnotations);
  assert.equal(result.dirty, false);
});
