import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { RotateCcw } from "lucide-react";
import Header from "./components/Header";
import FileUpload from "./components/FileUpload";
import XRayViewer from "./components/XRayViewer";
import PatientInfoForm from "./components/PatientInfoForm";
import PredictionResults from "./components/PredictionResults";
import AISummary from "./components/AISummary";
import { usePrediction } from "./hooks/usePrediction";

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const { data, loading, error, predict, reset } = usePrediction();

  const handleFileSelect = useCallback((file) => {
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    reset();
  }, [reset]);

  const handleAnalyze = useCallback(
    (patientInfo) => {
      if (imageFile) predict(imageFile, patientInfo);
    },
    [imageFile, predict]
  );

  const handleReset = useCallback(() => {
    setImageFile(null);
    setImagePreview(null);
    reset();
  }, [reset]);

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-primary)" }}>
      <Header />

      <main className="max-w-[1400px] mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* ── Left Panel: Image ────────────────────────── */}
          <div className="lg:col-span-7 space-y-4">
            <AnimatePresence mode="wait">
              {!imagePreview ? (
                <FileUpload key="upload" onFileSelect={handleFileSelect} />
              ) : (
                <motion.div
                  key="viewer"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-3"
                >
                  <XRayViewer
                    imagePreview={imagePreview}
                    gradcamData={data?.gradcam}
                  />
                  <button
                    onClick={handleReset}
                    className="w-full py-2.5 flex items-center justify-center gap-2 text-sm font-medium rounded-xl cursor-pointer transition-all"
                    style={{
                      background: "transparent",
                      border: "1px solid var(--border-default)",
                      color: "var(--text-secondary)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "var(--border-hover)";
                      e.currentTarget.style.background = "rgba(148, 163, 184, 0.05)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "var(--border-default)";
                      e.currentTarget.style.background = "transparent";
                    }}
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    Upload Different Image
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* ── Right Panel: Controls & Results ──────────── */}
          <div className="lg:col-span-5 space-y-4">
            <PatientInfoForm
              onSubmit={handleAnalyze}
              loading={loading}
              hasImage={!!imageFile}
            />

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="flex items-center gap-3 p-4 rounded-xl text-sm"
                  style={{
                    background: "rgba(239, 68, 68, 0.08)",
                    border: "1px solid rgba(239, 68, 68, 0.15)",
                    color: "#F87171",
                  }}
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Results */}
            <AnimatePresence>
              {data && (
                <>
                  <PredictionResults predictions={data.predictions} />
                  <AISummary summary={data.ai_summary} />
                </>
              )}
            </AnimatePresence>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="max-w-[1400px] mx-auto px-6 py-6" style={{ borderTop: "1px solid var(--border-default)" }}>
        <p className="text-center text-xs" style={{ color: "var(--text-muted)" }}>
          CXR-AI Assistant &mdash; For research and educational purposes only.
          Not a medical device. Does not provide clinical diagnoses.
        </p>
      </footer>
    </div>
  );
}

export default App;
