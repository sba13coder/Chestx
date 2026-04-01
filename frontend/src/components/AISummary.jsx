import { motion } from "framer-motion";
import {
  BrainCircuit,
  AlertCircle,
  Stethoscope,
  ArrowRight,
  Eye,
  UserCheck,
  AlertTriangle,
  ClipboardList,
} from "lucide-react";

const sectionConfig = {
  overview: { icon: Stethoscope, title: "Overview", color: "#22D3EE" },
  findings: { icon: ClipboardList, title: "Key Findings", color: "#F87171" },
  attention: { icon: Eye, title: "Grad-CAM Interpretation", color: "#A78BFA" },
  correlations: { icon: UserCheck, title: "Patient Correlation", color: "#FBBF24" },
  next_steps: { icon: ArrowRight, title: "Recommended Next Steps", color: "#22C55E" },
  watchlist: { icon: AlertTriangle, title: "Watchlist", color: "#FB923C" },
};

function SectionBlock({ sectionKey, content, index }) {
  const config = sectionConfig[sectionKey];
  if (!config || !content) return null;

  const Icon = config.icon;
  const isArray = Array.isArray(content);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.08 }}
      className="py-3.5"
      style={{ borderBottom: "1px solid var(--border-default)" }}
    >
      {/* Section Header */}
      <div className="flex items-center gap-2 mb-2.5">
        <div
          className="w-5 h-5 rounded-md flex items-center justify-center shrink-0"
          style={{ background: `${config.color}18` }}
        >
          <Icon className="w-3 h-3" style={{ color: config.color }} />
        </div>
        <span className="text-xs font-bold uppercase tracking-wider" style={{ color: config.color }}>
          {config.title}
        </span>
      </div>

      {/* Section Content */}
      {isArray ? (
        <ul className="space-y-2 ml-7">
          {content.map((item, i) => {
            const isUrgent = typeof item === "string" && item.startsWith("URGENT:");
            return (
              <li key={i} className="flex gap-2 text-sm leading-relaxed">
                <span
                  className="mt-2 w-1.5 h-1.5 rounded-full shrink-0"
                  style={{
                    background: isUrgent ? "#EF4444" : "var(--text-muted)",
                    boxShadow: isUrgent ? "0 0 6px #EF4444" : "none",
                  }}
                />
                <span
                  style={{
                    color: isUrgent ? "#F87171" : "var(--text-secondary)",
                    fontWeight: isUrgent ? 600 : 400,
                    fontFamily: "'Noto Sans', sans-serif",
                  }}
                >
                  {item}
                </span>
              </li>
            );
          })}
        </ul>
      ) : (
        <p
          className="text-sm leading-relaxed ml-7"
          style={{ color: "var(--text-secondary)", fontFamily: "'Noto Sans', sans-serif" }}
        >
          {content}
        </p>
      )}
    </motion.div>
  );
}

export default function AISummary({ summary }) {
  if (!summary) return null;

  // Support both old string format and new sections format
  const isStructured = typeof summary === "object" && !Array.isArray(summary);

  const sectionOrder = ["overview", "findings", "attention", "correlations", "next_steps", "watchlist"];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="glass-card overflow-hidden"
    >
      {/* Header */}
      <div
        className="px-5 py-3.5 flex items-center gap-2.5"
        style={{ borderBottom: "1px solid var(--border-default)" }}
      >
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center"
          style={{
            background: "rgba(8, 145, 178, 0.12)",
            border: "1px solid rgba(34, 211, 238, 0.15)",
          }}
        >
          <BrainCircuit className="w-4 h-4" style={{ color: "#22D3EE" }} />
        </div>
        <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          AI Clinical Summary
        </span>
      </div>

      {/* Content */}
      <div className="px-5">
        {isStructured ? (
          sectionOrder
            .filter((key) => summary[key])
            .map((key, i) => (
              <SectionBlock key={key} sectionKey={key} content={summary[key]} index={i} />
            ))
        ) : (
          /* Fallback for old string format */
          <div className="py-4">
            <p
              className="text-sm leading-relaxed"
              style={{ color: "var(--text-secondary)", fontFamily: "'Noto Sans', sans-serif" }}
            >
              {summary}
            </p>
          </div>
        )}
      </div>

      {/* Disclaimer */}
      <div className="px-5 py-4">
        <div
          className="flex gap-2.5 p-3 rounded-xl"
          style={{
            background: "rgba(245, 158, 11, 0.06)",
            border: "1px solid rgba(245, 158, 11, 0.1)",
          }}
        >
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" style={{ color: "#F59E0B" }} />
          <p className="text-xs leading-relaxed" style={{ color: "var(--text-muted)" }}>
            This tool is for research and educational purposes only. AI predictions should not
            replace professional clinical judgment. Always consult a qualified radiologist for
            definitive diagnosis.
          </p>
        </div>
      </div>
    </motion.div>
  );
}
