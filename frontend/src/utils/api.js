export async function predictDisease(imageFile, patientInfo = {}) {
  const formData = new FormData();
  formData.append("image", imageFile);

  Object.entries(patientInfo).forEach(([key, value]) => {
    if (value) formData.append(key, value);
  });

  const res = await fetch("/api/predict", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Server error" }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function getGradCAM(imageFile, classIdx) {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("class_idx", classIdx);

  const res = await fetch("/api/gradcam", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Server error" }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  return res.json();
}
