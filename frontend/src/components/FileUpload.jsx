import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, ImagePlus, AlertCircle } from "lucide-react";

export default function FileUpload({ onFileSelect }) {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  function handleDrop(e) {
    e.preventDefault();
    setDragActive(false);
    setError(null);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      onFileSelect(file);
    } else {
      setError("Please upload a valid image file (PNG or JPEG)");
    }
  }

  function handleChange(e) {
    const file = e.target.files[0];
    if (file) {
      setError(null);
      onFileSelect(file);
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
    >
      <div
        className="glass-card cursor-pointer group relative overflow-hidden"
        style={{ padding: "48px 32px" }}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        {/* Animated background gradient on drag */}
        <AnimatePresence>
          {dragActive && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 pointer-events-none"
              style={{
                background: "radial-gradient(circle at center, rgba(8, 145, 178, 0.15), transparent 70%)",
              }}
            />
          )}
        </AnimatePresence>

        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          className="hidden"
          onChange={handleChange}
        />

        <div className="relative z-10 flex flex-col items-center text-center">
          <motion.div
            animate={dragActive ? { scale: 1.1, y: -4 } : { scale: 1, y: 0 }}
            transition={{ type: "spring", stiffness: 300 }}
            className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
            style={{
              background: dragActive
                ? "rgba(8, 145, 178, 0.2)"
                : "rgba(148, 163, 184, 0.08)",
              border: `1px solid ${dragActive ? "rgba(34, 211, 238, 0.3)" : "rgba(148, 163, 184, 0.1)"}`,
            }}
          >
            {dragActive ? (
              <Upload className="w-7 h-7" style={{ color: "#22D3EE" }} />
            ) : (
              <ImagePlus className="w-7 h-7" style={{ color: "var(--text-muted)" }} />
            )}
          </motion.div>

          <p className="text-lg font-semibold mb-1" style={{ color: "var(--text-primary)" }}>
            Upload Chest X-Ray
          </p>
          <p className="text-sm mb-4" style={{ color: "var(--text-muted)" }}>
            Drag and drop or click to browse
          </p>

          <div className="flex items-center gap-4">
            <span
              className="text-xs px-2.5 py-1 rounded-md font-medium"
              style={{ background: "rgba(148, 163, 184, 0.08)", color: "var(--text-secondary)" }}
            >
              PNG
            </span>
            <span
              className="text-xs px-2.5 py-1 rounded-md font-medium"
              style={{ background: "rgba(148, 163, 184, 0.08)", color: "var(--text-secondary)" }}
            >
              JPEG
            </span>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
              Max 10MB
            </span>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="flex items-center gap-2 mt-3 px-4 py-2.5 rounded-xl text-sm"
            style={{
              background: "rgba(239, 68, 68, 0.1)",
              border: "1px solid rgba(239, 68, 68, 0.2)",
              color: "#F87171",
            }}
          >
            <AlertCircle className="w-4 h-4 shrink-0" />
            {error}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
