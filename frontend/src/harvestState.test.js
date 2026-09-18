import assert from "node:assert/strict";
import test from "node:test";

import { activeHarvestTracks, containedFrameRect, harvestModeAfterLoad, openHarvestTracks, snapHarvestTime, visibleHarvestMasks } from "./harvestState.js";

test("harvest times snap to 0.25 seconds without passing the video end", () => {
  assert.equal(snapHarvestTime(124), 0);
  assert.equal(snapHarvestTime(126), 250);
  assert.equal(snapHarvestTime(999, 820), 750);
  assert.equal(snapHarvestTime(1000, 1000), 750);
});

test("harvest frames are contained without cropping their bottom edge", () => {
  const wideViewer = containedFrameRect(1000, 500, 1920, 1080);
  assert.equal(wideViewer.height, 500);
  assert.ok(wideViewer.width < 1000);
  assert.equal(wideViewer.y, 0);

  const tallViewer = containedFrameRect(800, 800, 1920, 1080);
  assert.equal(tallViewer.width, 800);
  assert.ok(tallViewer.height < 800);
  assert.equal(tallViewer.x, 0);
});

test("only active tracks from the displayed video continue advancing", () => {
  const session = {
    current_video_id: "a",
    tracks: [
      { id: "1", video_id: "a", status: "active" },
      { id: "2", video_id: "a", status: "ended" },
      { id: "3", video_id: "b", status: "active" },
      { id: "4", video_id: "a", status: "needs_mask" },
    ],
  };
  assert.deepEqual(activeHarvestTracks(session).map((item) => item.id), ["1"]);
  assert.deepEqual(openHarvestTracks(session).map((item) => item.id), ["1", "4"]);
  assert.equal(harvestModeAfterLoad(session), "track");
  assert.equal(harvestModeAfterLoad({ ...session, tracks: [] }), "scan");
});

test("saved masks are visible only beside their matching video frame", () => {
  const session = {
    tracks: [
      { id: "current", video_id: "a", current_time_ms: 500, mask_url: "/mask/current" },
      { id: "next", video_id: "a", current_time_ms: 750, mask_url: "/mask/next" },
      { id: "other-video", video_id: "b", current_time_ms: 500, mask_url: "/mask/other" },
      { id: "deleted-mask", video_id: "a", current_time_ms: 500 },
    ],
  };

  assert.deepEqual(visibleHarvestMasks(session, "a", 500).map((item) => item.id), ["current"]);
  assert.deepEqual(visibleHarvestMasks(session, "a", 630).map((item) => item.id), ["next"]);
  assert.deepEqual(visibleHarvestMasks(session, "b", 500).map((item) => item.id), ["other-video"]);
});
