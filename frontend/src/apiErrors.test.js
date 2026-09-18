import test from "node:test";
import assert from "node:assert/strict";
import { formatApiError } from "./apiErrors.js";

test("FastAPI validation details are formatted as readable field errors", () => {
  const message = formatApiError({
    detail: [{
      type: "less_than_equal",
      loc: ["body", "objects_per_image"],
      msg: "Input should be less than or equal to 300"
    }]
  });
  assert.equal(message, "objects per image: Input should be less than or equal to 300");
});

test("API string errors and fallbacks remain readable", () => {
  assert.equal(formatApiError({ detail: "Folder does not exist" }), "Folder does not exist");
  assert.equal(formatApiError({}, "Unprocessable Entity"), "Unprocessable Entity");
});
