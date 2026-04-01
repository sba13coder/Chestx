import { Activity, Shield, Cpu } from "lucide-react";
import { motion } from "framer-motion";

export default function Header() {
  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="sticky top-0 z-50"
      style={{
        background: "rgba(10, 14, 26, 0.85)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderBottom: "1px solid rgba(148, 163, 184, 0.08)",
      }}
    >
      <div className="max-w-[1400px] mx-auto px-6 py-3.5 flex items-center justify-between">
        {/* Logo & Title */}
        <div className="flex items-center gap-3.5">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{
              background: "linear-gradient(135deg, #0891B2, #22D3EE)",
              boxShadow: "0 0 20px rgba(8, 145, 178, 0.3)",
            }}
          >
            <Activity className="w-5 h-5 text-white" strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="text-[17px] font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
              CXR-AI Assistant
            </h1>
            <p className="text-[11px] font-medium tracking-wide uppercase" style={{ color: "var(--text-muted)" }}>
              Chest X-Ray Analysis & Explainability
            </p>
          </div>
        </div>

        {/* Model Badges */}
        <div className="flex items-center gap-2.5">
          <div
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium cursor-default"
            style={{
              background: "rgba(8, 145, 178, 0.12)",
              color: "#22D3EE",
              border: "1px solid rgba(34, 211, 238, 0.2)",
            }}
          >
            <Cpu className="w-3 h-3" />
            DenseNet121
          </div>
          <div
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium cursor-default"
            style={{
              background: "rgba(34, 197, 94, 0.12)",
              color: "#22C55E",
              border: "1px solid rgba(34, 197, 94, 0.2)",
            }}
          >
            <Shield className="w-3 h-3" />
            AUC 0.751
          </div>
        </div>
      </div>
    </motion.header>
  );
}
