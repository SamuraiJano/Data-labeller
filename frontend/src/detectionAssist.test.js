import assert from "node:assert/strict";
import test from "node:test";

import { detectionPromptMetadata, parseDetectionLabels } from "./detectionAssist.js";

test("detection prompts are trimmed and deduplicated case-insensitively", () => {
  assert.deepEqual(
    parseDetectionLabels(" orange, Naartjie; ORANGE\n citrus   fruit "),
    ["orange", "Naartjie", "citrus fruit"]
  );
});

test("accepted proposals retain auditable model metadata", () => {
  assert.deepEqual(
    detectionPromptMetadata({ label: "orange", score: "0.87", model_id: "grounding-dino-tiny" }, ["orange"]),
    {
      labels: ["orange"],
      predicted_label: "orange",
      score: 0.87,
      model_id: "grounding-dino-tiny",
      accepted_as_draft: true
    }
  );
});
