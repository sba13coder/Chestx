import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle, BarChart3 } from "lucide-react";
import ConfidenceBar from "./ConfidenceBar";

export default function PredictionResults({ predictions }) {
  if (!predictions) return null;

  const significant = predictions.filter((p) => p.is_significant);
  const other = predictions.filter((p) => !p.is_significant);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className="glass-card overflow-hidden"
    >
      {/* Header */}
      <div className="px-5 py-3.5 flex items-center justify-between" style={{ borderBottom: "1px solid var(--border-default)" }}>
        <div className="flex items-center gap-2.5">
          <BarChart3 className="w-4 h-4" style={{ color: "var(--color-primary)" }} />
          <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Disease Predictions
          </span>
        </div>
        {significant.length > 0 ? (
          <div
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
            style={{
              background: "rgba(239, 68, 68, 0.1)",
              color: "#F87171",
              border: "1px solid rgba(239, 68, 68, 0.15)",
            }}
          >
            <AlertTriangle className="w-3 h-3" />
            {significant.length} finding{significant.length > 1 ? "s" : ""} detected
          </div>
        ) : (
          <div
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
            style={{
              background: "rgba(34, 197, 94, 0.1)",
              color: "#4ADE80",
              border: "1px solid rgba(34, 197, 94, 0.15)",
            }}
          >
            <CheckCircle className="w-3 h-3" />
            No significant findings
          </div>
        )}
      </div>

      {/* Predictions List */}
      <div className="px-5 py-3">
        {/* Significant findings */}
        {significant.length > 0 && (
          <div className="mb-2">
            <p className="text-[10px] font-semibold uppercase tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>
              Significant Findings
            </p>
            {significant.map((p, i) => (
              <ConfidenceBar key={p.disease} {...p} index={i} />
            ))}
          </div>
        )}

        {/* Separator */}
        {significant.length > 0 && other.length > 0 && (
          <div className="my-2.5" style={{ borderTop: "1px solid var(--border-default)" }} />
        )}

        {/* Other findings */}
        {other.length > 0 && (
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-wider mb-1" style={{ color: "var(--text-muted)" }}>
              Below Threshold
            </p>
            {other.map((p, i) => (
              <ConfidenceBar key={p.disease} {...p} index={i + significant.length} />
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}
