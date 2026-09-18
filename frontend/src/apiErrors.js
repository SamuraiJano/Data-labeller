const readableLocation = (location) => (location || [])
  .filter((part) => part !== "body")
  .map((part) => String(part).replaceAll("_", " "))
  .join(" › ");

export function formatApiError(payload, fallback = "Request failed") {
  const detail = payload?.detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      const location = readableLocation(item?.loc);
      const message = item?.msg || "Invalid value";
      return location ? `${location}: ${message}` : message;
    }).filter(Boolean);
    if (messages.length) return messages.join("\n");
  }
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object") {
    if (typeof detail.message === "string") return detail.message;
    return JSON.stringify(detail);
  }
  return fallback;
}
