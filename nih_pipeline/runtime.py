"""Runtime loader for deployment manifests and inference ensembles."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .modeling import build_model, default_normalization, display_name, get_cam_target_layer, normalize_weights

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))


@dataclass
class LoadedModelSpec:
    architecture: str
    checkpoint_path: str
    weight: float
    input_size: int
    grayscale: bool
    mean: list[float]
    std: list[float]
    label_cols: list[str]
    task: str
    tta: bool
    name: str


class GenericGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.enabled = False

        def fwd_hook(_module, _inputs, output):
            if not self.enabled:
                return
            self.activations = output
            if output.requires_grad:
                output.register_hook(self._save_grad)

        self.hook = self.target_layer.register_forward_hook(fwd_hook)

    def _save_grad(self, grad):
        self.gradients = grad

    def __call__(self, x: torch.Tensor, class_idx: int):
        self.enabled = True
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[:, class_idx].sum().backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.clamp(cam, min=0)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        self.enabled = False
        return cam.detach().cpu().numpy(), logits.detach().cpu().numpy()


class DeploymentRuntime:
    def __init__(
        self,
        *,
        bundles: list[dict[str, Any]],
        label_cols: list[str],
        model_info: dict[str, Any],
        explainability_index: int,
        significance_threshold: float,
        device: str,
    ):
        self.bundles = bundles
        self.label_cols = label_cols
        self.model_info = model_info
        self.explainability_index = explainability_index
        self.significance_threshold = significance_threshold
        self.device = device

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, *, device: str | None = None) -> "DeploymentRuntime":
        manifest_path = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text())
        base_dir = manifest_path.parent
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        bundles = []
        label_cols = manifest["label_cols"]
        for item in manifest["models"]:
            spec = LoadedModelSpec(
                architecture=item["architecture"],
                checkpoint_path=str((base_dir / item["checkpoint"]).resolve()) if not Path(item["checkpoint"]).is_absolute() else item["checkpoint"],
                weight=float(item.get("weight", 1.0)),
                input_size=int(item.get("input_size", 224)),
                grayscale=bool(item.get("grayscale", True)),
                mean=list(item.get("mean") or default_normalization(bool(item.get("grayscale", True)))[0]),
                std=list(item.get("std") or default_normalization(bool(item.get("grayscale", True)))[1]),
                label_cols=label_cols,
                task=item.get("task", "single_label_softmax"),
                tta=bool(item.get("tta", False)),
                name=item.get("name", display_name(item["architecture"])),
            )
            bundles.append(cls._load_bundle(spec, resolved_device))

        weights = normalize_weights(bundle["spec"].weight for bundle in bundles)
        for bundle, weight in zip(bundles, weights, strict=True):
            bundle["spec"].weight = weight

        return cls(
            bundles=bundles,
            label_cols=label_cols,
            model_info=manifest.get("model_info", {}),
            explainability_index=int(manifest.get("explainability_model_index", 0)),
            significance_threshold=float(manifest.get("significance_threshold", 0.5)),
            device=resolved_device,
        )

    @classmethod
    def legacy_densenet(
        cls,
        checkpoint_path: str | Path,
        *,
        metrics_path: str | Path | None = None,
        device: str | None = None,
    ) -> "DeploymentRuntime":
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        label_cols = checkpoint["label_cols"]
        spec = LoadedModelSpec(
            architecture="densenet121",
            checkpoint_path=str(checkpoint_path),
            weight=1.0,
            input_size=224,
            grayscale=False,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            label_cols=label_cols,
            task="multilabel_sigmoid",
            tta=False,
            name="DenseNet121",
        )
        bundle = cls._load_bundle(spec, resolved_device)
        mean_auc = None
        if metrics_path and Path(metrics_path).exists():
            mean_auc = json.loads(Path(metrics_path).read_text()).get("mean_auc")
        return cls(
            bundles=[bundle],
            label_cols=label_cols,
            model_info={"name": "DenseNet121", "mean_auc": mean_auc},
            explainability_index=0,
            significance_threshold=0.5,
            device=resolved_device,
        )

    @staticmethod
    def _load_bundle(spec: LoadedModelSpec, device: str) -> dict[str, Any]:
        checkpoint = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(
            spec.architecture,
            num_classes=len(spec.label_cols),
            in_channels=1 if spec.grayscale else 3,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        model.eval()
        return {
            "spec": spec,
            "model": model,
            "cam": GenericGradCAM(model, get_cam_target_layer(model, spec.architecture)),
        }

    def _build_transform(self, spec: LoadedModelSpec, *, tta: bool = False):
        resize_size = spec.input_size if not tta else int(round(spec.input_size * 1.08))
        transforms: list[Any] = [T.Resize((resize_size, resize_size))]
        if tta:
            transforms.append(T.CenterCrop(spec.input_size))
        transforms.extend([T.ToTensor(), T.Normalize(spec.mean, spec.std)])
        return T.Compose(transforms)

    def _prepare_input(self, pil_image: Image.Image, spec: LoadedModelSpec, *, tta: bool = False) -> torch.Tensor:
        converted = pil_image.convert("L" if spec.grayscale else "RGB")
        tensor = self._build_transform(spec, tta=tta)(converted).unsqueeze(0)
        return tensor.to(self.device, dtype=torch.float32)

    def _predict_bundle_logits(self, bundle: dict[str, Any], pil_image: Image.Image) -> np.ndarray:
        spec = bundle["spec"]
        model = bundle["model"]
        with torch.no_grad():
            logits = model(self._prepare_input(pil_image, spec)).detach()
            if spec.tta:
                tta_logits = model(self._prepare_input(pil_image, spec, tta=True)).detach()
                logits = 0.5 * (logits + tta_logits)
        return logits.cpu().numpy()

    def _combine_logits(self, pil_image: Image.Image) -> np.ndarray:
        logits = None
        for bundle in self.bundles:
            bundle_logits = self._predict_bundle_logits(bundle, pil_image)
            weighted = bundle["spec"].weight * bundle_logits
            logits = weighted if logits is None else logits + weighted
        return logits

    def _probabilities(self, logits: np.ndarray) -> np.ndarray:
        task = self.bundles[0]["spec"].task
        if task == "single_label_softmax":
            tensor = torch.from_numpy(logits)
            probs = torch.softmax(tensor, dim=1).numpy()
            return probs[0]
        if task == "multilabel_sigmoid":
            return torch.sigmoid(torch.from_numpy(logits)).numpy()[0]
        raise ValueError(f"Unsupported task: {task}")

    def _gradcam(self, pil_image: Image.Image, class_idx: int) -> np.ndarray:
        bundle = self.bundles[self.explainability_index]
        spec = bundle["spec"]
        x = self._prepare_input(pil_image, spec)
        x.requires_grad_(True)
        cam_map, _ = bundle["cam"](x, class_idx)
        return cam_map[0]

    def predict(self, pil_image: Image.Image) -> dict[str, Any]:
        logits = self._combine_logits(pil_image)
        probs = self._probabilities(logits)
        predictions = [
            {
                "disease": name,
                "probability": round(float(prob), 4),
                "is_significant": float(prob) >= self.significance_threshold,
            }
            for name, prob in zip(self.label_cols, probs, strict=True)
        ]
        predictions.sort(key=lambda item: item["probability"], reverse=True)

        top_disease = predictions[0]["disease"]
        for item in predictions:
            if item["disease"] != "No Finding":
                top_disease = item["disease"]
                break

        class_idx = self.label_cols.index(top_disease)
        overlay = overlay_cam_on_image(pil_image.convert("RGB"), self._gradcam(pil_image, class_idx))
        return {
            "predictions": predictions,
            "gradcam": {
                "heatmap_overlay": np_to_base64(overlay),
                "original_image": pil_to_base64(pil_image.convert("RGB")),
                "default_class": top_disease,
            },
            "label_cols": self.label_cols,
            "model_info": self._model_info_payload(),
        }

    def predict_gradcam(self, pil_image: Image.Image, class_idx: int) -> dict[str, Any]:
        logits = self._combine_logits(pil_image)
        probs = self._probabilities(logits)
        overlay = overlay_cam_on_image(pil_image.convert("RGB"), self._gradcam(pil_image, class_idx))
        return {
            "heatmap_overlay": np_to_base64(overlay),
            "disease": self.label_cols[class_idx],
            "probability": round(float(probs[class_idx]), 4),
        }

    def _model_info_payload(self) -> dict[str, Any]:
        name = self.model_info.get("name")
        if not name:
            if len(self.bundles) == 1:
                name = self.bundles[0]["spec"].name
            else:
                name = " + ".join(bundle["spec"].name for bundle in self.bundles)
        return {
            "name": name,
            "mean_auc": self.model_info.get("mean_auc"),
        }


def pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def np_to_base64(arr: np.ndarray) -> str:
    return pil_to_base64(Image.fromarray(arr))


def overlay_cam_on_image(pil_img: Image.Image, cam_2d: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    from matplotlib import cm

    heat = Image.fromarray((cam_2d * 255).astype(np.uint8)).resize(
        pil_img.size, resample=Image.BILINEAR
    )
    heat_np = np.array(heat).astype(np.float32) / 255.0
    colored_heat = cm.jet(heat_np)[:, :, :3]
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    overlay = (1 - alpha) * img_np + alpha * colored_heat
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)
