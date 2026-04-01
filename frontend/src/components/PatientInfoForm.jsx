import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, User, Scan, Loader2 } from "lucide-react";

const INITIAL_STATE = {
  age: "",
  gender: "",
  smoking_status: "",
  surgical_history: "",
  existing_conditions: "",
};

export default function PatientInfoForm({ onSubmit, loading, hasImage }) {
  const [info, setInfo] = useState(INITIAL_STATE);
  const [expanded, setExpanded] = useState(true);

  function update(field, value) {
    setInfo((prev) => ({ ...prev, [field]: value }));
  }

  function handleSubmit(e) {
    e.preventDefault();
    onSubmit(info);
  }

  const inputStyle = {
    background: "rgba(148, 163, 184, 0.06)",
    border: "1px solid var(--border-default)",
    borderRadius: "var(--radius-sm)",
    color: "var(--text-primary)",
    fontSize: "13px",
    padding: "9px 12px",
    width: "100%",
    outline: "none",
    transition: "border-color 0.2s ease, box-shadow 0.2s ease",
  };

  const labelStyle = {
    display: "block",
    fontSize: "11px",
    fontWeight: 600,
    color: "var(--text-muted)",
    marginBottom: "5px",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  };

  return (
    <form onSubmit={handleSubmit} className="glass-card overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-3.5 flex items-center justify-between cursor-pointer group"
        style={{ background: "transparent", border: "none", color: "var(--text-primary)" }}
      >
        <div className="flex items-center gap-2.5">
          <User className="w-4 h-4" style={{ color: "var(--color-primary)" }} />
          <span className="text-sm font-semibold">Patient Information</span>
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>(Optional)</span>
        </div>
        <motion.div animate={{ rotate: expanded ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown className="w-4 h-4" style={{ color: "var(--text-muted)" }} />
        </motion.div>
      </button>

      {/* Form Fields */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            style={{ overflow: "hidden" }}
          >
            <div className="px-5 pb-4 space-y-3.5" style={{ borderTop: "1px solid var(--border-default)", paddingTop: "16px" }}>
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label style={labelStyle}>Age</label>
                  <input
                    type="number"
                    placeholder="e.g. 65"
                    value={info.age}
                    onChange={(e) => update("age", e.target.value)}
                    style={inputStyle}
                    onFocus={(e) => { e.target.style.borderColor = "var(--border-focus)"; e.target.style.boxShadow = "0 0 0 3px rgba(8, 145, 178, 0.1)"; }}
                    onBlur={(e) => { e.target.style.borderColor = "var(--border-default)"; e.target.style.boxShadow = "none"; }}
                  />
                </div>
                <div>
                  <label style={labelStyle}>Gender</label>
                  <select
                    value={info.gender}
                    onChange={(e) => update("gender", e.target.value)}
                    style={{ ...inputStyle, appearance: "none", cursor: "pointer" }}
                    onFocus={(e) => { e.target.style.borderColor = "var(--border-focus)"; }}
                    onBlur={(e) => { e.target.style.borderColor = "var(--border-default)"; }}
                  >
                    <option value="">Select</option>
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <div>
                  <label style={labelStyle}>Smoking</label>
                  <select
                    value={info.smoking_status}
                    onChange={(e) => update("smoking_status", e.target.value)}
                    style={{ ...inputStyle, appearance: "none", cursor: "pointer" }}
                    onFocus={(e) => { e.target.style.borderColor = "var(--border-focus)"; }}
                    onBlur={(e) => { e.target.style.borderColor = "var(--border-default)"; }}
                  >
                    <option value="">Select</option>
                    <option value="never">Never</option>
                    <option value="former">Former</option>
                    <option value="current">Current</option>
                  </select>
                </div>
              </div>

              <div>
                <label style={labelStyle}>Surgical History</label>
                <input
                  type="text"
                  placeholder="e.g. CABG, lung resection..."
                  value={info.surgical_history}
                  onChange={(e) => update("surgical_history", e.target.value)}
                  style={inputStyle}
                  onFocus={(e) => { e.target.style.borderColor = "var(--border-focus)"; e.target.style.boxShadow = "0 0 0 3px rgba(8, 145, 178, 0.1)"; }}
                  onBlur={(e) => { e.target.style.borderColor = "var(--border-default)"; e.target.style.boxShadow = "none"; }}
                />
              </div>

              <div>
                <label style={labelStyle}>Existing Conditions</label>
                <input
                  type="text"
                  placeholder="e.g. Hypertension, Diabetes, COPD..."
                  value={info.existing_conditions}
                  onChange={(e) => update("existing_conditions", e.target.value)}
                  style={inputStyle}
                  onFocus={(e) => { e.target.style.borderColor = "var(--border-focus)"; e.target.style.boxShadow = "0 0 0 3px rgba(8, 145, 178, 0.1)"; }}
                  onBlur={(e) => { e.target.style.borderColor = "var(--border-default)"; e.target.style.boxShadow = "none"; }}
                />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Submit Button */}
      <div className="px-5 pb-5 pt-2">
        <button
          type="submit"
          disabled={!hasImage || loading}
          className="w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 cursor-pointer transition-all"
          style={{
            background: !hasImage || loading
              ? "rgba(148, 163, 184, 0.1)"
              : "linear-gradient(135deg, #0891B2, #0E7490)",
            color: !hasImage || loading ? "var(--text-muted)" : "white",
            border: "none",
            opacity: !hasImage || loading ? 0.5 : 1,
            boxShadow: hasImage && !loading ? "0 4px 14px rgba(8, 145, 178, 0.3)" : "none",
            cursor: !hasImage || loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Analyzing X-Ray...
            </>
          ) : (
            <>
              <Scan className="w-4 h-4" />
              Analyze X-Ray
            </>
          )}
        </button>
      </div>
    </form>
  );
}
