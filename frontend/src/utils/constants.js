export const SIGNIFICANCE_THRESHOLD = 0.5;

export function getConfidenceColor(probability) {
  if (probability >= 0.7) return { bar: "bg-red-500", text: "text-red-400" };
  if (probability >= 0.5) return { bar: "bg-amber-500", text: "text-amber-400" };
  if (probability >= 0.3) return { bar: "bg-yellow-500", text: "text-yellow-400" };
  return { bar: "bg-emerald-500", text: "text-emerald-400" };
}
