import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Eye, Layers, ZoomIn, ZoomOut } from "lucide-react";

export default function XRayViewer({ imagePreview, gradcamData }) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [opacity, setOpacity] = useState(100);
  const [zoom, setZoom] = useState(1);

  const hasGradcam = gradcamData?.heatmap_overlay;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      className="glass-card overflow-hidden"
    >
      {/* Image Container */}
      <div
        className="relative overflow-hidden cursor-crosshair"
        style={{ background: "#000", minHeight: 300 }}
      >
        {/* Original Image */}
        <motion.img
          src={imagePreview}
          alt="Chest X-Ray"
          className="w-full h-auto max-h-[520px] object-contain"
          style={{
            transform: `scale(${zoom})`,
            transition: "transform 0.3s ease",
          }}
        />

        {/* Grad-CAM Overlay */}
        <AnimatePresence>
          {showHeatmap && hasGradcam && (
            <motion.img
              key="heatmap"
              initial={{ opacity: 0 }}
              animate={{ opacity: opacity / 100 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
              src={`data:image/png;base64,${gradcamData.heatmap_overlay}`}
              alt="Grad-CAM Heatmap"
              className="absolute inset-0 w-full h-full object-contain"
              style={{
                transform: `scale(${zoom})`,
                transition: "transform 0.3s ease",
              }}
            />
          )}
        </AnimatePresence>

        {/* Grad-CAM Label */}
        <AnimatePresence>
          {showHeatmap && gradcamData?.default_class && (
            <motion.div
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -12 }}
              className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold"
              style={{
                background: "rgba(0, 0, 0, 0.75)",
                backdropFilter: "blur(8px)",
                color: "#22D3EE",
                border: "1px solid rgba(34, 211, 238, 0.2)",
              }}
            >
              <Layers className="w-3.5 h-3.5" />
              Grad-CAM: {gradcamData.default_class}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Zoom Controls */}
        <div
          className="absolute bottom-4 right-4 flex items-center gap-1 rounded-lg p-1"
          style={{
            background: "rgba(0, 0, 0, 0.7)",
            backdropFilter: "blur(8px)",
          }}
        >
          <button
            onClick={(e) => { e.stopPropagation(); setZoom((z) => Math.max(1, z - 0.25)); }}
            className="p-1.5 rounded-md cursor-pointer transition-colors"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => (e.target.style.background = "rgba(255,255,255,0.1)")}
            onMouseLeave={(e) => (e.target.style.background = "transparent")}
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs font-mono px-2" style={{ color: "var(--text-secondary)" }}>
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={(e) => { e.stopPropagation(); setZoom((z) => Math.min(3, z + 0.25)); }}
            className="p-1.5 rounded-md cursor-pointer transition-colors"
            style={{ color: "var(--text-secondary)" }}
            onMouseEnter={(e) => (e.target.style.background = "rgba(255,255,255,0.1)")}
            onMouseLeave={(e) => (e.target.style.background = "transparent")}
          >
            <ZoomIn className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Controls Bar */}
      {hasGradcam && (
        <div className="px-4 py-3 flex items-center gap-4" style={{ borderTop: "1px solid var(--border-default)" }}>
          {/* Toggle Buttons */}
          <div className="flex rounded-lg overflow-hidden" style={{ border: "1px solid var(--border-default)" }}>
            <button
              onClick={() => setShowHeatmap(false)}
              className="flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium cursor-pointer transition-all"
              style={{
                background: !showHeatmap ? "var(--color-primary)" : "transparent",
                color: !showHeatmap ? "white" : "var(--text-secondary)",
              }}
            >
              <Eye className="w-3.5 h-3.5" />
              Original
            </button>
            <button
              onClick={() => setShowHeatmap(true)}
              className="flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium cursor-pointer transition-all"
              style={{
                background: showHeatmap ? "var(--color-primary)" : "transparent",
                color: showHeatmap ? "white" : "var(--text-secondary)",
                borderLeft: "1px solid var(--border-default)",
              }}
            >
              <Layers className="w-3.5 h-3.5" />
              Grad-CAM
            </button>
          </div>

          {/* Opacity Slider */}
          {showHeatmap && (
            <motion.div
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3 flex-1"
            >
              <span className="text-xs" style={{ color: "var(--text-muted)" }}>Opacity</span>
              <input
                type="range"
                min="10"
                max="100"
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="flex-1 h-1 rounded-full appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, var(--color-primary) ${opacity}%, rgba(148,163,184,0.2) ${opacity}%)`,
                }}
              />
              <span className="text-xs font-mono w-8 text-right" style={{ color: "var(--text-secondary)" }}>
                {opacity}%
              </span>
            </motion.div>
          )}
        </div>
      )}
    </motion.div>
  );
}
