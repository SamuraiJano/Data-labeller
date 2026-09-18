import test from "node:test";
import assert from "node:assert/strict";
import { retainExistingSelections, updateImageSelection } from "./bulkSelection.js";

test("shift selection includes the full visible range", () => {
  const selected = updateImageSelection(new Set(["b"]), "e", true, true, "b", ["a", "b", "c", "d", "e"]);
  assert.deepEqual([...selected], ["b", "c", "d", "e"]);
});

test("shift deselection removes the full visible range", () => {
  const selected = updateImageSelection(new Set(["a", "b", "c", "d"]), "c", false, true, "a", ["a", "b", "c", "d"]);
  assert.deepEqual([...selected], ["d"]);
});

test("selection is pruned when the project image list changes", () => {
  const selected = retainExistingSelections(new Set(["kept", "removed"]), [{ id: "kept" }]);
  assert.deepEqual([...selected], ["kept"]);
});
