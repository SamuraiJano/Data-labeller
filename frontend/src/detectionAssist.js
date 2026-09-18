export const DEFAULT_DETECTION_PROMPT = "orange, naartjie, citrus fruit";

export function parseDetectionLabels(value) {
  const seen = new Set();
  return String(value || "")
    .split(/[,;\n]+/)
    .map((label) => label.trim().replace(/\s+/g, " "))
    .filter((label) => {
      if (!label) return false;
      const key = label.toLocaleLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .slice(0, 12);
}

export function detectionPromptMetadata(proposal, labels) {
  return {
    labels,
    predicted_label: proposal.label,
    score: Number(proposal.score),
    model_id: proposal.model_id,
    accepted_as_draft: true
  };
}
