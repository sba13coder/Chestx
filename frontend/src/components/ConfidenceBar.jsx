import { motion } from "framer-motion";

function getColor(probability) {
  if (probability >= 0.7) return { bar: "#EF4444", bg: "rgba(239, 68, 68, 0.1)", text: "#F87171", dot: "#EF4444" };
  if (probability >= 0.5) return { bar: "#F59E0B", bg: "rgba(245, 158, 11, 0.1)", text: "#FBBF24", dot: "#F59E0B" };
  if (probability >= 0.3) return { bar: "#EAB308", bg: "rgba(234, 179, 8, 0.08)", text: "#FDE047", dot: "#EAB308" };
  return { bar: "#22C55E", bg: "rgba(34, 197, 94, 0.08)", text: "#4ADE80", dot: "#22C55E" };
}

export default function ConfidenceBar({ disease, probability, isSignificant, index = 0 }) {
  const color = getColor(probability);
  const pct = (probability * 100).toFixed(1);

  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: index * 0.03 }}
      className="flex items-center gap-3 py-[7px] group cursor-default"
      style={{ opacity: isSignificant ? 1 : 0.5 }}
    >
      {/* Disease name */}
      <div className="flex items-center gap-2 w-[170px] shrink-0">
        {isSignificant && (
          <span
            className="w-[6px] h-[6px] rounded-full shrink-0"
            style={{ background: color.dot, boxShadow: `0 0 6px ${color.dot}` }}
          />
        )}
        <span
          className="text-xs truncate font-medium"
          style={{ color: isSignificant ? "var(--text-primary)" : "var(--text-secondary)" }}
          title={disease}
        >
          {disease}
        </span>
      </div>

      {/* Bar */}
      <div className="flex-1 h-[6px] rounded-full overflow-hidden" style={{ background: "rgba(148, 163, 184, 0.08)" }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(probability * 100, 1)}%` }}
          transition={{ duration: 0.6, delay: index * 0.03, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{
            background: `linear-gradient(90deg, ${color.bar}, ${color.bar}cc)`,
            boxShadow: isSignificant ? `0 0 8px ${color.bar}40` : "none",
          }}
        />
      </div>

      {/* Percentage */}
      <span
        className="text-xs w-[48px] text-right shrink-0"
        style={{ color: color.text, fontFamily: "'JetBrains Mono', monospace", fontWeight: 500 }}
      >
        {pct}%
      </span>
    </motion.div>
  );
}
