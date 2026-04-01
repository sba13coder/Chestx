"""
Artifact-driven inference runtime for chest X-ray disease detection.

Falls back to the legacy DenseNet checkpoint when no deployment manifest exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nih_pipeline.runtime import DeploymentRuntime

# ── Globals ──────────────────────────────────────────────────────────
_runtime: DeploymentRuntime | None = None
_label_cols: list[str] | None = None
_device = "cpu"

LEGACY_CHECKPOINT_PATH = ROOT_DIR / "models" / "densenet_multilabel_best.pt"
LEGACY_METRICS_PATH = ROOT_DIR / "reports_step1" / "test_metrics.json"
MANIFEST_CANDIDATES = [
    ROOT_DIR / "artifacts" / "deployment_manifest.json",
    ROOT_DIR / "models" / "deployment_manifest.json",
    ROOT_DIR / "deployment_manifest.json",
]


def _resolve_manifest(explicit_path: str | None = None) -> Path | None:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        return path if path.exists() else None

    env_path = os.getenv("NIH_DEPLOYMENT_MANIFEST")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        return path if path.exists() else None

    for candidate in MANIFEST_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def init_model(manifest_path: str | None = None, ckpt_path: str | None = None):
    global _runtime, _label_cols, _device

    _device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    resolved_manifest = _resolve_manifest(manifest_path)
    if resolved_manifest is not None:
        _runtime = DeploymentRuntime.from_manifest(resolved_manifest, device=_device)
    else:
        legacy_ckpt = Path(ckpt_path).expanduser().resolve() if ckpt_path else LEGACY_CHECKPOINT_PATH
        if not legacy_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {legacy_ckpt}")
        _runtime = DeploymentRuntime.legacy_densenet(
            legacy_ckpt,
            metrics_path=LEGACY_METRICS_PATH if LEGACY_METRICS_PATH.exists() else None,
            device=_device,
        )

    _label_cols = _runtime.label_cols
    print(f"Model loaded on {_device}: {_runtime.model_info.get('name')}")
    print(f"Labels: {_label_cols}")
    return True


def predict(pil_image: Image.Image, patient_info=None):
    if _runtime is None:
        raise RuntimeError("Model not initialized. Call init_model() first.")

    result = _runtime.predict(pil_image)
    result["ai_summary"] = generate_summary(result["predictions"], patient_info)
    return result


def predict_gradcam(pil_image: Image.Image, class_idx: int):
    if _runtime is None:
        raise RuntimeError("Model not initialized.")
    return _runtime.predict_gradcam(pil_image, class_idx)


def generate_summary(predictions, patient_info=None):
    """
    Generate a structured clinical assistance summary.
    Returns a dict with sections for richer frontend rendering.
    """
    significant = [p for p in predictions if p["is_significant"] and p["disease"] != "No Finding"]
    below = [p for p in predictions if not p["is_significant"] and p["disease"] != "No Finding"]
    patient_info = patient_info or {}

    age = patient_info.get("age", "")
    gender = patient_info.get("gender", "")
    smoking = patient_info.get("smoking_status", "")
    surgical = patient_info.get("surgical_history", "")
    conditions = patient_info.get("existing_conditions", "")

    sections = {}

    # ── 1. Overview ──────────────────────────────────────
    if not significant:
        sections["overview"] = (
            "No significant pathological findings were detected above the confidence threshold. "
            "The model did not identify any disease with high probability. However, low-confidence "
            "findings may still warrant clinical attention depending on the patient's presentation."
        )
    else:
        top3 = significant[:3]
        finding_parts = [f"{p['disease']} ({p['probability']*100:.1f}%)" for p in top3]
        remaining = len(significant) - 3
        overview = f"The analysis identified {len(significant)} finding(s) above the significance threshold. "
        overview += f"Primary findings: {', '.join(finding_parts)}"
        if remaining > 0:
            others = [p['disease'] for p in significant[3:]]
            overview += f", and {remaining} additional: {', '.join(others)}"
        overview += "."
        sections["overview"] = overview

    # ── 2. Key Findings Detail ───────────────────────────
    if significant:
        details = []
        # Disease-specific clinical context
        clinical_context = {
            "Cardiomegaly": "Enlarged cardiac silhouette may indicate heart failure, valvular disease, or pericardial effusion. Correlate with echocardiography.",
            "Effusion": "Pleural effusion detected. Consider underlying causes: heart failure, infection, malignancy, or inflammatory conditions.",
            "Edema": "Pulmonary edema pattern suggests fluid overload. Evaluate cardiac function and fluid balance.",
            "Pneumonia": "Consolidation pattern consistent with pneumonia. Consider clinical symptoms (fever, cough) and laboratory markers (WBC, CRP).",
            "Atelectasis": "Atelectatic changes identified. May be post-operative, obstructive, or compressive in nature.",
            "Pneumothorax": "Pneumothorax detected. Assess size and clinical stability. May require urgent intervention if tension pneumothorax is suspected.",
            "Emphysema": "Hyperinflation and emphysematous changes noted. Consistent with chronic obstructive pulmonary disease (COPD).",
            "Nodule": "Pulmonary nodule(s) identified. Size, morphology, and growth rate assessment recommended. Consider Fleischner criteria for follow-up.",
            "Mass": "Pulmonary mass identified. Further characterization with CT chest is strongly recommended. Malignancy must be excluded.",
            "Fibrosis": "Fibrotic changes identified. Consider interstitial lung disease, post-inflammatory scarring, or drug-induced fibrosis.",
            "Consolidation": "Consolidation may represent infection, hemorrhage, or organizing pneumonia. Correlate with clinical context.",
            "Hernia": "Hiatal hernia or diaphragmatic hernia pattern detected. Assess for associated gastroesophageal reflux symptoms.",
            "Infiltration": "Pulmonary infiltrates identified. Differential includes infection, inflammation, and aspiration.",
            "Pleural_Thickening": "Pleural thickening noted. Consider prior infection, asbestos exposure, or inflammatory etiology.",
            "Tortuous Aorta": "Tortuous or elongated aorta consistent with age-related vascular changes or hypertension.",
            "Calcification of the Aorta": "Aortic calcification indicates atherosclerotic disease. Cardiovascular risk assessment recommended.",
            "Subcutaneous Emphysema": "Subcutaneous air detected. May indicate pneumothorax, esophageal injury, or post-procedural complication.",
            "Pneumoperitoneum": "Free air under diaphragm is a critical finding. If not post-surgical, consider hollow viscus perforation requiring urgent evaluation.",
            "Pneumomediastinum": "Air in mediastinum detected. Evaluate for esophageal rupture, airway injury, or alveolar rupture.",
        }

        for p in significant[:5]:  # Top 5 detailed
            ctx = clinical_context.get(p["disease"], "")
            if ctx:
                details.append(f"{p['disease']} ({p['probability']*100:.1f}%): {ctx}")
        sections["findings"] = details

    # ── 3. Grad-CAM Interpretation ───────────────────────
    if significant:
        top = significant[0]
        sections["attention"] = (
            f"The Grad-CAM heatmap highlights regions the model considered most relevant for "
            f"predicting {top['disease']}. Warmer colors (red/yellow) indicate areas of higher "
            f"model attention. This visualization helps verify whether the model's focus aligns "
            f"with clinically expected anatomical regions for the detected pathology."
        )

    # ── 4. Patient Correlation ───────────────────────────
    correlations = []

    if smoking and smoking.lower() in ("current", "former"):
        smoking_related = [p for p in significant if p["disease"] in
                          ("Emphysema", "Infiltration", "Pneumonia", "Consolidation",
                           "Nodule", "Mass", "Fibrosis")]
        if smoking_related:
            names = ", ".join(p["disease"] for p in smoking_related)
            correlations.append(
                f"Smoking history ({smoking} smoker) is a known risk factor for: {names}. "
                f"Smoking cessation counseling and pulmonary function testing may be indicated."
            )
        else:
            correlations.append(
                f"Patient has a smoking history ({smoking}). While current findings are not "
                f"directly smoking-related, continued screening and risk assessment is advised."
            )

    if age and age.isdigit():
        age_int = int(age)
        if age_int >= 60:
            age_related = [p for p in significant if p["disease"] in
                          ("Calcification of the Aorta", "Tortuous Aorta", "Cardiomegaly",
                           "Effusion", "Edema", "Fibrosis")]
            if age_related:
                names = ", ".join(p["disease"] for p in age_related)
                correlations.append(
                    f"Patient age ({age}) increases the likelihood of: {names}. "
                    f"Age-related cardiovascular and pulmonary changes should be considered."
                )
        if age_int < 40 and significant:
            correlations.append(
                f"Patient is relatively young ({age}). Findings may warrant more aggressive "
                f"workup to rule out significant underlying pathology."
            )

    if gender:
        gender_str = "Male" if gender == "M" else "Female" if gender == "F" else gender
        cv_findings = [p for p in significant if p["disease"] in
                      ("Cardiomegaly", "Calcification of the Aorta", "Tortuous Aorta")]
        if gender == "M" and cv_findings:
            correlations.append(
                f"{gender_str} patients have higher baseline cardiovascular risk. "
                f"Consider comprehensive cardiac evaluation."
            )

    if surgical:
        correlations.append(
            f"Surgical history ({surgical}): post-operative changes may account for some findings. "
            f"Compare with pre-operative imaging if available."
        )

    if conditions:
        correlations.append(
            f"Reported conditions ({conditions}): correlate imaging findings with known diagnoses. "
            f"Assess for disease progression or new complications."
        )

    if correlations:
        sections["correlations"] = correlations

    # ── 5. Recommended Next Steps ────────────────────────
    next_steps = []

    if any(p["disease"] == "Pneumoperitoneum" and p["is_significant"] for p in predictions):
        next_steps.append("URGENT: Evaluate for hollow viscus perforation. Surgical consultation recommended.")

    if any(p["disease"] == "Pneumothorax" and p["is_significant"] for p in predictions):
        next_steps.append("URGENT: Assess pneumothorax size and clinical stability. Consider chest tube placement.")

    if any(p["disease"] in ("Mass", "Nodule") and p["is_significant"] for p in predictions):
        next_steps.append("CT chest with contrast recommended for further characterization of pulmonary mass/nodule.")

    if any(p["disease"] in ("Cardiomegaly", "Edema", "Effusion") and p["is_significant"] for p in predictions):
        next_steps.append("Echocardiography recommended to evaluate cardiac function and structure.")
        next_steps.append("Consider BNP/NT-proBNP levels to assess heart failure status.")

    if any(p["disease"] == "Pneumonia" and p["is_significant"] for p in predictions):
        next_steps.append("Obtain CBC, CRP, and blood cultures if clinically indicated. Consider sputum culture.")

    if any(p["disease"] in ("Emphysema", "Fibrosis") and p["is_significant"] for p in predictions):
        next_steps.append("Pulmonary function testing (spirometry) recommended.")

    if any(p["disease"] in ("Calcification of the Aorta", "Tortuous Aorta") and p["is_significant"] for p in predictions):
        next_steps.append("Cardiovascular risk assessment and lipid panel recommended.")

    if significant and not next_steps:
        next_steps.append("Correlate findings with clinical presentation and laboratory results.")
        next_steps.append("Consider follow-up imaging if clinically indicated.")

    if not significant:
        next_steps.append("If clinical suspicion remains, consider additional imaging modalities (CT, ultrasound).")
        next_steps.append("Routine follow-up as per clinical guidelines.")

    next_steps.append("Compare with prior imaging studies if available to assess temporal changes.")
    sections["next_steps"] = next_steps

    # ── 6. Watchlist (borderline findings) ───────────────
    borderline = [p for p in below if p["probability"] >= 0.35 and p["disease"] != "No Finding"]
    if borderline:
        watchlist = [f"{p['disease']} ({p['probability']*100:.1f}%)" for p in borderline[:5]]
        sections["watchlist"] = (
            f"Borderline findings below threshold but worth monitoring: {', '.join(watchlist)}. "
            f"These may become clinically relevant with disease progression."
        )

    return sections
