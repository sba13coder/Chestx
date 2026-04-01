"""End-to-end training pipeline for the NIH noise-aware long-tail setup."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .modeling import build_model, default_normalization, display_name, normalize_weights

IGNORE_COLS = {"id", "subject_id"}

try:
    import pyarrow  # noqa: F401

    HAS_PARQUET = True
except Exception:
    HAS_PARQUET = False


@dataclass
class RunConfig:
    architecture: str
    input_size: int
    epochs: int = 50
    warmup_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    mixup_alpha: float = 0.1
    patience: int = 10
    ema_decay: float = 0.999
    logit_tau: float = 1.0
    use_tta: bool = True
    pretrained: bool = True
    grayscale: bool = True
    ssl_checkpoint: str | None = None


@dataclass
class PipelineConfig:
    train_csv: str
    val_csv: str
    test_csv: str
    image_dir: str
    output_dir: str
    device: str = "auto"
    num_workers: int = 4
    master_seed: int = 42
    teacher_seeds: list[int] = field(default_factory=lambda: [13, 29])
    teacher: RunConfig = field(default_factory=lambda: RunConfig("convnext_tiny", 448))
    model_a: RunConfig = field(default_factory=lambda: RunConfig("convnext_tiny", 448))
    model_b: RunConfig = field(default_factory=lambda: RunConfig("efficientnet_v2_s", 384))
    ssl_epochs: int = 100
    ssl_batch_size: int = 32
    ssl_temperature: float = 0.2
    ssl_projection_dim: int = 256
    disable_ssl: bool = False


class AddGaussianNoise:
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def autocast_context(device: str):
    if device == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return contextlib.nullcontext()


def build_grad_scaler(device: str) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=device == "cuda")


def save_json(path: str | Path, payload: Any) -> None:
    def _json_default(value: Any):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default))


def write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    if not HAS_PARQUET:
        raise RuntimeError("pyarrow is required to write parquet artifacts.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def infer_label_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in IGNORE_COLS]


def load_label_frame(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(csv_path)
    label_cols = infer_label_columns(frame)
    labels = frame[label_cols].to_numpy(dtype=np.float32)
    positive_counts = labels.sum(axis=1)
    if not np.allclose(positive_counts, 1.0):
        invalid = int(np.count_nonzero(np.abs(positive_counts - 1.0) > 1e-6))
        raise ValueError(f"{csv_path} is not single-label one-hot encoded. Invalid rows: {invalid}")

    frame = frame.copy().reset_index(drop=True)
    frame["label_idx"] = labels.argmax(axis=1).astype(int)
    frame["label_name"] = frame["label_idx"].map(lambda idx: label_cols[idx])
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame, label_cols


def combine_train_pool(train_csv: str, val_csv: str) -> tuple[pd.DataFrame, list[str]]:
    train_df, label_cols = load_label_frame(train_csv)
    val_df, val_label_cols = load_label_frame(val_csv)
    if label_cols != val_label_cols:
        raise ValueError("Train/val label columns do not match.")
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined["row_id"] = np.arange(len(combined), dtype=np.int64)
    return combined, label_cols


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    eye = np.eye(num_classes, dtype=np.float32)
    return eye[indices]


def build_class_groups(train_df: pd.DataFrame, label_cols: list[str]) -> dict[str, list[str]]:
    counts = train_df["label_name"].value_counts()
    ordered = counts.index.tolist()
    head_cut = math.ceil(len(ordered) / 3)
    medium_cut = math.ceil((2 * len(ordered)) / 3)
    return {
        "head": ordered[:head_cut],
        "medium": ordered[head_cut:medium_cut],
        "tail": ordered[medium_cut:],
    }


def class_stats_payload(train_df: pd.DataFrame, test_df: pd.DataFrame, label_cols: list[str]) -> dict[str, Any]:
    train_counts = train_df["label_name"].value_counts().reindex(label_cols, fill_value=0)
    test_counts = test_df["label_name"].value_counts().reindex(label_cols, fill_value=0)
    groups = build_class_groups(train_df, label_cols)
    return {
        "label_cols": label_cols,
        "train_counts": train_counts.to_dict(),
        "test_counts": test_counts.to_dict(),
        "groups": groups,
    }


def build_cv_folds(train_df: pd.DataFrame, *, n_splits: int, seed: int) -> list[dict[str, Any]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = train_df["subject_id"].to_numpy()
    targets = train_df["label_idx"].to_numpy()
    folds = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(train_df, y=targets, groups=groups)):
        train_subjects = set(train_df.iloc[train_idx]["subject_id"].tolist())
        val_subjects = set(train_df.iloc[val_idx]["subject_id"].tolist())
        leakage = len(train_subjects & val_subjects)
        if leakage:
            raise RuntimeError(f"Subject leakage detected in fold {fold}: {leakage}")
        folds.append(
            {
                "fold": fold,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "subject_leakage": leakage,
            }
        )
    return folds


def split_manifest_payload(train_df: pd.DataFrame, folds: list[dict[str, Any]], label_cols: list[str]) -> dict[str, Any]:
    assignments = []
    for fold_info in folds:
        fold = fold_info["fold"]
        for idx in fold_info["val_idx"]:
            row = train_df.iloc[idx]
            assignments.append(
                {
                    "row_id": int(row["row_id"]),
                    "image_id": row["id"],
                    "subject_id": row["subject_id"],
                    "fold": int(fold),
                    "label_name": row["label_name"],
                }
            )

    per_fold = []
    for fold_info in folds:
        val_frame = train_df.iloc[fold_info["val_idx"]]
        per_fold.append(
            {
                "fold": int(fold_info["fold"]),
                "train_size": int(fold_info["train_size"]),
                "val_size": int(fold_info["val_size"]),
                "subject_leakage": int(fold_info["subject_leakage"]),
                "val_label_counts": val_frame["label_name"].value_counts().reindex(label_cols, fill_value=0).to_dict(),
            }
        )

    return {
        "n_splits": len(folds),
        "folds": per_fold,
        "assignments": assignments,
    }


def build_transforms(input_size: int, *, train: bool, grayscale: bool):
    mean, std = default_normalization(grayscale)
    transforms: list[Any] = []
    if train:
        transforms.extend(
            [
                T.Resize((int(round(input_size * 1.12)), int(round(input_size * 1.12)))),
                T.RandomResizedCrop(input_size, scale=(0.85, 1.0)),
                T.RandomAffine(degrees=5, translate=(0.03, 0.03)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                AddGaussianNoise(std=0.01),
                T.Normalize(mean, std),
            ]
        )
    else:
        transforms.extend([T.Resize((input_size, input_size)), T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose(transforms)


class CXRSingleLabelDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str | Path,
        *,
        transform,
        soft_targets: np.ndarray,
        sample_weights: np.ndarray,
        grayscale: bool = True,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.soft_targets = soft_targets.astype(np.float32)
        self.sample_weights = sample_weights.astype(np.float32)
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        path = self.image_dir / row["id"]
        image = Image.open(path).convert("L" if self.grayscale else "RGB")
        tensor = self.transform(image)
        return {
            "image": tensor,
            "target_idx": int(row["label_idx"]),
            "target_soft": torch.tensor(self.soft_targets[idx], dtype=torch.float32),
            "sample_weight": torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            "image_id": row["id"],
            "subject_id": row["subject_id"],
        }


class SimCLRDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: str | Path, *, input_size: int):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = T.Compose(
            [
                T.Resize((int(round(input_size * 1.2)), int(round(input_size * 1.2)))),
                T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                T.RandomAffine(degrees=5, translate=(0.03, 0.03)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                AddGaussianNoise(std=0.02),
                T.Normalize(*default_normalization(True)),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_dir / self.frame.iloc[idx]["id"]).convert("L")
        return self.transform(image), self.transform(image)


def mixup_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0 or images.shape[0] < 2:
        return images, targets
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(images.shape[0], device=images.device)
    mixed_images = lam * images + (1 - lam) * images[perm]
    mixed_targets = lam * targets + (1 - lam) * targets[perm]
    return mixed_images, mixed_targets


def soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    sample_weights: torch.Tensor,
    logit_adjustment: torch.Tensor | None,
) -> torch.Tensor:
    if logit_adjustment is not None:
        logits = logits + logit_adjustment
    log_probs = F.log_softmax(logits, dim=1)
    losses = -(targets * log_probs).sum(dim=1)
    return (losses * sample_weights).mean()


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            model_state = model.state_dict()
            for key, value in self.shadow.state_dict().items():
                if not value.is_floating_point():
                    value.copy_(model_state[key])
                else:
                    value.mul_(self.decay).add_(model_state[key], alpha=1.0 - self.decay)


class ConvNeXtEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.convnext_tiny(weights=weights)
        self.backbone.features[0][0] = _grayscale_conv(self.backbone.features[0][0])
        self.embedding_dim = self.backbone.classifier[2].in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


def _grayscale_conv(conv: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        1,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(x), dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    batch = z1.shape[0]
    mask = torch.eye(2 * batch, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    positives = torch.cat([torch.arange(batch, 2 * batch), torch.arange(0, batch)]).to(z.device)
    return F.cross_entropy(sim, positives)


def load_ssl_weights(model: nn.Module, ssl_checkpoint: str | None) -> None:
    if not ssl_checkpoint:
        return
    payload = torch.load(ssl_checkpoint, map_location="cpu", weights_only=False)
    model_state = model.state_dict()
    source_state = payload["encoder_state"]
    filtered = {
        key: value
        for key, value in source_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model_state.update(filtered)
    model.load_state_dict(model_state)


def make_loaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    image_dir: str,
    *,
    train_soft_targets: np.ndarray,
    train_sample_weights: np.ndarray,
    val_soft_targets: np.ndarray,
    val_sample_weights: np.ndarray,
    run_config: RunConfig,
    class_sample_weights: np.ndarray,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = CXRSingleLabelDataset(
        train_frame,
        image_dir,
        transform=build_transforms(run_config.input_size, train=True, grayscale=run_config.grayscale),
        soft_targets=train_soft_targets,
        sample_weights=train_sample_weights,
        grayscale=run_config.grayscale,
    )
    val_ds = CXRSingleLabelDataset(
        val_frame,
        image_dir,
        transform=build_transforms(run_config.input_size, train=False, grayscale=run_config.grayscale),
        soft_targets=val_soft_targets,
        sample_weights=val_sample_weights,
        grayscale=run_config.grayscale,
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(class_sample_weights, dtype=torch.double),
        num_samples=len(class_sample_weights),
        replacement=True,
    )
    loader_kwargs = {
        "batch_size": run_config.batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def collect_probabilities(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(batch["target_idx"].numpy())
    return np.concatenate(probabilities, axis=0), np.concatenate(targets, axis=0)


def compute_metrics(probabilities: np.ndarray, targets: np.ndarray, label_cols: list[str]) -> dict[str, Any]:
    targets_one_hot = one_hot(targets, len(label_cols))
    per_auc = {}
    per_ap = {}
    aucs = []
    aps = []
    for index, label in enumerate(label_cols):
        target_col = targets_one_hot[:, index]
        prob_col = probabilities[:, index]
        if len(np.unique(target_col)) < 2:
            per_auc[label] = None
            per_ap[label] = None
            continue
        auc = roc_auc_score(target_col, prob_col)
        ap = average_precision_score(target_col, prob_col)
        per_auc[label] = float(auc)
        per_ap[label] = float(ap)
        aucs.append(auc)
        aps.append(ap)

    predictions = probabilities.argmax(axis=1)
    return {
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "mean_ap": float(np.mean(aps)) if aps else float("nan"),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "per_auc": per_auc,
        "per_ap": per_ap,
    }


def build_logit_adjustment(train_targets: np.ndarray, tau: float, num_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
    priors = counts / max(counts.sum(), 1.0)
    priors = np.clip(priors, 1e-8, None)
    return torch.tensor(tau * np.log(priors), dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: str,
    mixup_alpha: float,
    logit_adjustment: torch.Tensor,
    ema: EMA | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target_soft"].to(device, non_blocking=True)
        sample_weights = batch["sample_weight"].to(device, non_blocking=True)

        images, targets = mixup_batch(images, targets, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits = model(images)
            loss = soft_cross_entropy(
                logits,
                targets,
                sample_weights=sample_weights,
                logit_adjustment=logit_adjustment,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    logit_adjustment: torch.Tensor,
    label_cols: list[str],
) -> tuple[float, dict[str, Any], np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    probabilities = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            soft_targets = batch["target_soft"].to(device, non_blocking=True)
            sample_weights = batch["sample_weight"].to(device, non_blocking=True)
            target_idx = batch["target_idx"].numpy()
            with autocast_context(device):
                logits = model(images)
                loss = soft_cross_entropy(
                    logits,
                    soft_targets,
                    sample_weights=sample_weights,
                    logit_adjustment=logit_adjustment,
                )
            total_loss += float(loss.item()) * images.shape[0]
            total_count += images.shape[0]
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(target_idx)
    probs = np.concatenate(probabilities, axis=0)
    target_indices = np.concatenate(targets, axis=0)
    metrics = compute_metrics(probs, target_indices, label_cols)
    return total_loss / max(total_count, 1), metrics, probs


def checkpoint_payload(
    model: nn.Module,
    *,
    run_config: RunConfig,
    label_cols: list[str],
    class_priors: np.ndarray,
) -> dict[str, Any]:
    mean, std = default_normalization(run_config.grayscale)
    return {
        "architecture": run_config.architecture,
        "input_size": run_config.input_size,
        "grayscale": run_config.grayscale,
        "task": "single_label_softmax",
        "tta": run_config.use_tta,
        "mean": mean,
        "std": std,
        "label_cols": label_cols,
        "class_priors": class_priors.tolist(),
        "model_state": model.state_dict(),
    }


def train_fold(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    image_dir: str,
    label_cols: list[str],
    run_config: RunConfig,
    global_soft_targets: np.ndarray,
    global_sample_weights: np.ndarray,
    device: str,
    num_workers: int,
    seed: int,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    set_seed(seed)
    num_classes = len(label_cols)
    train_indices = train_frame["row_id"].to_numpy()
    val_indices = val_frame["row_id"].to_numpy()
    train_targets = train_frame["label_idx"].to_numpy()
    class_counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
    class_priors = class_counts / max(class_counts.sum(), 1.0)
    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
    sampler_weights = class_weights[train_targets] * global_sample_weights[train_indices]
    train_loader, val_loader = make_loaders(
        train_frame,
        val_frame,
        image_dir,
        train_soft_targets=global_soft_targets[train_indices],
        train_sample_weights=global_sample_weights[train_indices],
        val_soft_targets=global_soft_targets[val_indices],
        val_sample_weights=np.ones(len(val_indices), dtype=np.float32),
        run_config=run_config,
        class_sample_weights=sampler_weights,
        num_workers=num_workers,
    )

    model = build_model(
        run_config.architecture,
        num_classes=num_classes,
        in_channels=1 if run_config.grayscale else 3,
        pretrained=run_config.pretrained,
    ).to(device)
    load_ssl_weights(model, run_config.ssl_checkpoint)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (
            (epoch + 1) / max(run_config.warmup_epochs, 1)
            if epoch < run_config.warmup_epochs
            else 0.5
            * (
                1
                + math.cos(
                    math.pi
                    * (epoch - run_config.warmup_epochs + 1)
                    / max(run_config.epochs - run_config.warmup_epochs, 1)
                )
            )
        ),
    )
    scaler = build_grad_scaler(device)
    ema = EMA(model, decay=run_config.ema_decay)
    logit_adjustment = build_logit_adjustment(train_targets, run_config.logit_tau, num_classes, device)

    best_auc = -1.0
    best_epoch = 0
    best_probs = None
    bad_epochs = 0
    history = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[train] arch={run_config.architecture} seed={seed} train={len(train_frame)} val={len(val_frame)} "
        f"size={run_config.input_size} device={device} ckpt={checkpoint_path.name}",
        flush=True,
    )

    for epoch in range(1, run_config.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            mixup_alpha=run_config.mixup_alpha,
            logit_adjustment=logit_adjustment,
            ema=ema,
        )
        scheduler.step()
        eval_model = ema.shadow if ema is not None else model
        val_loss, metrics, probabilities = evaluate_model(
            eval_model,
            val_loader,
            device=device,
            logit_adjustment=logit_adjustment,
            label_cols=label_cols,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_auc": metrics["mean_auc"],
                "val_mean_ap": metrics["mean_ap"],
                "val_macro_f1": metrics["macro_f1"],
            }
        )
        print(
            f"[epoch] arch={run_config.architecture} seed={seed} epoch={epoch}/{run_config.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={metrics['mean_auc']:.4f} val_ap={metrics['mean_ap']:.4f} "
            f"val_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if metrics["mean_auc"] > best_auc:
            best_auc = metrics["mean_auc"]
            best_epoch = epoch
            best_probs = probabilities
            bad_epochs = 0
            torch.save(
                checkpoint_payload(eval_model, run_config=run_config, label_cols=label_cols, class_priors=class_priors),
                checkpoint_path,
            )
            print(
                f"[best] arch={run_config.architecture} seed={seed} epoch={epoch} auc={best_auc:.4f} "
                f"checkpoint={checkpoint_path}",
                flush=True,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= run_config.patience:
                print(
                    f"[early-stop] arch={run_config.architecture} seed={seed} "
                    f"epoch={epoch} patience={run_config.patience}",
                    flush=True,
                )
                break

    return {
        "best_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "best_probs": best_probs,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }


def train_full_model(
    frame: pd.DataFrame,
    *,
    image_dir: str,
    label_cols: list[str],
    run_config: RunConfig,
    soft_targets: np.ndarray,
    sample_weights: np.ndarray,
    device: str,
    num_workers: int,
    seed: int,
    checkpoint_path: str | Path,
    epochs: int,
) -> str:
    set_seed(seed)
    num_classes = len(label_cols)
    class_counts = np.bincount(frame["label_idx"].to_numpy(), minlength=num_classes).astype(np.float32)
    class_priors = class_counts / max(class_counts.sum(), 1.0)
    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
    sampler_weights = class_weights[frame["label_idx"].to_numpy()] * sample_weights[frame["row_id"].to_numpy()]
    dataset = CXRSingleLabelDataset(
        frame,
        image_dir,
        transform=build_transforms(run_config.input_size, train=True, grayscale=run_config.grayscale),
        soft_targets=soft_targets[frame["row_id"].to_numpy()],
        sample_weights=sample_weights[frame["row_id"].to_numpy()],
        grayscale=run_config.grayscale,
    )
    loader = DataLoader(
        dataset,
        sampler=WeightedRandomSampler(
            weights=torch.as_tensor(sampler_weights, dtype=torch.double),
            num_samples=len(sampler_weights),
            replacement=True,
        ),
        batch_size=run_config.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    model = build_model(
        run_config.architecture,
        num_classes=num_classes,
        in_channels=1 if run_config.grayscale else 3,
        pretrained=run_config.pretrained,
    ).to(device)
    load_ssl_weights(model, run_config.ssl_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (
            (epoch + 1) / max(run_config.warmup_epochs, 1)
            if epoch < run_config.warmup_epochs
            else 0.5
            * (1 + math.cos(math.pi * (epoch - run_config.warmup_epochs + 1) / max(epochs - run_config.warmup_epochs, 1)))
        ),
    )
    scaler = build_grad_scaler(device)
    ema = EMA(model, decay=run_config.ema_decay)
    logit_adjustment = build_logit_adjustment(frame["label_idx"].to_numpy(), run_config.logit_tau, num_classes, device)
    print(
        f"[full-train] arch={run_config.architecture} epochs={epochs} samples={len(frame)} "
        f"size={run_config.input_size} device={device} checkpoint={checkpoint_path}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        run_epoch(
            model,
            loader,
            optimizer,
            scaler,
            device=device,
            mixup_alpha=run_config.mixup_alpha,
            logit_adjustment=logit_adjustment,
            ema=ema,
        )
        scheduler.step()
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"[full-train-epoch] arch={run_config.architecture} epoch={epoch}/{epochs}",
                flush=True,
            )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(ema.shadow, run_config=run_config, label_cols=label_cols, class_priors=class_priors),
        checkpoint_path,
    )
    return str(checkpoint_path)


def predict_with_checkpoint(
    checkpoint_path: str | Path,
    frame: pd.DataFrame,
    *,
    image_dir: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(
        checkpoint["architecture"],
        num_classes=len(checkpoint["label_cols"]),
        in_channels=1 if checkpoint.get("grayscale", True) else 3,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = CXRSingleLabelDataset(
        frame,
        image_dir,
        transform=build_transforms(checkpoint["input_size"], train=False, grayscale=checkpoint.get("grayscale", True)),
        soft_targets=one_hot(frame["label_idx"].to_numpy(), len(checkpoint["label_cols"])),
        sample_weights=np.ones(len(frame), dtype=np.float32),
        grayscale=checkpoint.get("grayscale", True),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    probs, _targets = collect_probabilities(model, loader, device)

    if checkpoint.get("tta"):
        tta_dataset = CXRSingleLabelDataset(
            frame,
            image_dir,
            transform=T.Compose(
                [
                    T.Resize((int(round(checkpoint["input_size"] * 1.08)), int(round(checkpoint["input_size"] * 1.08)))),
                    T.CenterCrop(checkpoint["input_size"]),
                    T.ToTensor(),
                    T.Normalize(checkpoint["mean"], checkpoint["std"]),
                ]
            ),
            soft_targets=one_hot(frame["label_idx"].to_numpy(), len(checkpoint["label_cols"])),
            sample_weights=np.ones(len(frame), dtype=np.float32),
            grayscale=checkpoint.get("grayscale", True),
        )
        tta_loader = DataLoader(
            tta_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        tta_probs, _ = collect_probabilities(model, tta_loader, device)
        probs = 0.5 * (probs + tta_probs)

    return probs


def run_cross_validation(
    frame: pd.DataFrame,
    *,
    folds: list[dict[str, Any]],
    image_dir: str,
    label_cols: list[str],
    run_config: RunConfig,
    soft_targets: np.ndarray,
    sample_weights: np.ndarray,
    device: str,
    num_workers: int,
    stage_name: str,
    output_dir: str | Path,
    seeds: Iterable[int],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    oof_sum = np.zeros((len(frame), len(label_cols)), dtype=np.float32)
    oof_counts = np.zeros(len(frame), dtype=np.int32)
    fold_summaries = []
    best_epochs = []

    for seed in seeds:
        for fold_info in folds:
            fold = fold_info["fold"]
            print(
                f"[cv] stage={stage_name} seed={seed} fold={fold} "
                f"train={fold_info['train_size']} val={fold_info['val_size']}",
                flush=True,
            )
            train_frame = frame.iloc[fold_info["train_idx"]].copy().reset_index(drop=True)
            val_frame = frame.iloc[fold_info["val_idx"]].copy().reset_index(drop=True)
            checkpoint_path = output_dir / "checkpoints" / stage_name / f"seed_{seed}_fold_{fold}.pt"
            result = train_fold(
                train_frame,
                val_frame,
                image_dir=image_dir,
                label_cols=label_cols,
                run_config=run_config,
                global_soft_targets=soft_targets,
                global_sample_weights=sample_weights,
                device=device,
                num_workers=num_workers,
                seed=seed + fold,
                checkpoint_path=checkpoint_path,
            )
            val_indices = val_frame["row_id"].to_numpy()
            oof_sum[val_indices] += result["best_probs"]
            oof_counts[val_indices] += 1
            best_epochs.append(result["best_epoch"])
            fold_summaries.append(
                {
                    "seed": int(seed),
                    "fold": int(fold),
                    "best_auc": result["best_auc"],
                    "best_epoch": result["best_epoch"],
                    "checkpoint_path": result["checkpoint_path"],
                    "history": result["history"],
                }
            )
            print(
                f"[cv-done] stage={stage_name} seed={seed} fold={fold} "
                f"best_auc={result['best_auc']:.4f} best_epoch={result['best_epoch']}",
                flush=True,
            )

    if np.any(oof_counts == 0):
        missing = int(np.count_nonzero(oof_counts == 0))
        raise RuntimeError(f"OOF coverage is incomplete. Missing rows: {missing}")

    oof_probs = oof_sum / oof_counts[:, None]
    metrics = compute_metrics(oof_probs, frame["label_idx"].to_numpy(), label_cols)
    print(
        f"[cv-summary] stage={stage_name} mean_auc={metrics['mean_auc']:.4f} "
        f"mean_ap={metrics['mean_ap']:.4f} macro_f1={metrics['macro_f1']:.4f}",
        flush=True,
    )
    return {
        "stage_name": stage_name,
        "oof_probs": oof_probs,
        "metrics": metrics,
        "folds": fold_summaries,
        "best_epoch_median": int(np.median(best_epochs)),
    }


def build_noise_targets(
    frame: pd.DataFrame,
    *,
    teacher_probs: np.ndarray,
    label_cols: list[str],
    class_groups: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    num_classes = len(label_cols)
    labels = frame["label_idx"].to_numpy()
    teacher_top1 = teacher_probs.argmax(axis=1)
    teacher_prob_label = teacher_probs[np.arange(len(frame)), labels]
    teacher_prob_top1 = teacher_probs.max(axis=1)
    margins = teacher_prob_top1 - teacher_prob_label
    one_hot_targets = one_hot(labels, num_classes)
    soft_targets = np.zeros_like(teacher_probs, dtype=np.float32)
    sample_weights = np.ones(len(frame), dtype=np.float32)
    label_names = frame["label_name"].tolist()
    group_lookup = {}
    for group_name, names in class_groups.items():
        for name in names:
            group_lookup[name] = group_name

    statuses = []
    for idx in range(len(frame)):
        is_clean = teacher_top1[idx] == labels[idx] or teacher_prob_label[idx] >= 0.35
        is_suspicious = (
            teacher_prob_label[idx] < 0.10
            and teacher_prob_top1[idx] > 0.75
            and margins[idx] > 0.40
        )
        if is_clean:
            status = "clean"
            soft_targets[idx] = 0.9 * one_hot_targets[idx] + 0.1 * teacher_probs[idx]
            sample_weights[idx] = 1.0
        elif is_suspicious:
            status = "suspicious"
            group = group_lookup.get(label_names[idx], "tail")
            if group in {"head", "medium"}:
                soft_targets[idx] = 0.2 * one_hot_targets[idx] + 0.8 * teacher_probs[idx]
                sample_weights[idx] = 0.6
            else:
                soft_targets[idx] = 0.5 * one_hot_targets[idx] + 0.5 * teacher_probs[idx]
                sample_weights[idx] = 0.85
        else:
            status = "uncertain"
            soft_targets[idx] = 0.5 * one_hot_targets[idx] + 0.5 * teacher_probs[idx]
            sample_weights[idx] = 1.0
        statuses.append(status)

    noise_map = pd.DataFrame(
        {
            "image_id": frame["id"].tolist(),
            "subject_id": frame["subject_id"].tolist(),
            "label": label_names,
            "teacher_top1": [label_cols[idx] for idx in teacher_top1],
            "teacher_prob_label": teacher_prob_label.astype(np.float32),
            "teacher_prob_top1": teacher_prob_top1.astype(np.float32),
            "status": statuses,
            "sample_weight": sample_weights.astype(np.float32),
        }
    )
    return soft_targets, sample_weights, noise_map


def oof_predictions_frame(frame: pd.DataFrame, label_cols: list[str], probabilities: np.ndarray) -> pd.DataFrame:
    payload = pd.DataFrame(
        {
            "image_id": frame["id"].tolist(),
            "subject_id": frame["subject_id"].tolist(),
            "label": frame["label_name"].tolist(),
            "teacher_top1": [label_cols[idx] for idx in probabilities.argmax(axis=1)],
        }
    )
    for idx, label in enumerate(label_cols):
        payload[f"prob_{label}"] = probabilities[:, idx].astype(np.float32)
    return payload


def priority_label_table(
    baseline_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    *,
    labels: list[str],
) -> list[dict[str, Any]]:
    table = []
    for label in labels:
        before = baseline_metrics["per_auc"].get(label)
        after = final_metrics["per_auc"].get(label)
        table.append(
            {
                "label": label,
                "baseline_auc": before,
                "final_auc": after,
                "delta": None if before is None or after is None else float(after - before),
            }
        )
    return table


def bottom5_improvement_count(baseline_metrics: dict[str, Any], final_metrics: dict[str, Any]) -> int:
    ranked = [
        (label, auc)
        for label, auc in baseline_metrics["per_auc"].items()
        if auc is not None
    ]
    bottom5 = [label for label, _auc in sorted(ranked, key=lambda item: item[1])[:5]]
    improved = 0
    for label in bottom5:
        before = baseline_metrics["per_auc"][label]
        after = final_metrics["per_auc"].get(label)
        if after is not None and after > before:
            improved += 1
    return improved


def write_deployment_manifest(
    output_dir: str | Path,
    *,
    label_cols: list[str],
    model_a_checkpoint: str,
    model_b_checkpoint: str,
    model_a_weight: float,
    model_b_weight: float,
    mean_auc: float,
) -> None:
    manifest = {
        "label_cols": label_cols,
        "significance_threshold": 0.5,
        "explainability_model_index": 0,
        "model_info": {
            "name": "ConvNeXt-Tiny + EfficientNetV2-S",
            "mean_auc": mean_auc,
        },
        "models": [
            {
                "name": display_name("convnext_tiny"),
                "architecture": "convnext_tiny",
                "checkpoint": str(Path(model_a_checkpoint).resolve()),
                "weight": model_a_weight,
                "input_size": 448,
                "grayscale": True,
                "task": "single_label_softmax",
                "tta": True,
                "mean": [0.5],
                "std": [0.25],
            },
            {
                "name": display_name("efficientnet_v2_s"),
                "architecture": "efficientnet_v2_s",
                "checkpoint": str(Path(model_b_checkpoint).resolve()),
                "weight": model_b_weight,
                "input_size": 384,
                "grayscale": True,
                "task": "single_label_softmax",
                "tta": True,
                "mean": [0.5],
                "std": [0.25],
            },
        ],
    }
    save_json(Path(output_dir) / "deployment_manifest.json", manifest)


def run_simclr_pretraining(
    frame: pd.DataFrame,
    *,
    image_dir: str,
    output_dir: str | Path,
    input_size: int,
    batch_size: int,
    epochs: int,
    temperature: float,
    projection_dim: int,
    device: str,
    num_workers: int,
    seed: int,
) -> str:
    set_seed(seed)
    dataset = SimCLRDataset(frame, image_dir, input_size=input_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    encoder = ConvNeXtEncoder(pretrained=True).to(device)
    projector = ProjectionHead(encoder.embedding_dim, projection_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )
    print(
        f"[ssl] start arch=convnext_tiny epochs={epochs} batch_size={batch_size} "
        f"size={input_size} device={device}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        encoder.train()
        projector.train()
        epoch_losses = []
        for view1, view2 in loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                z1 = projector(encoder(view1))
                z2 = projector(encoder(view2))
                loss = nt_xent_loss(z1, z2, temperature)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        print(
            f"[ssl-epoch] epoch={epoch}/{epochs} loss={np.mean(epoch_losses):.4f}",
            flush=True,
        )

    output_path = Path(output_dir) / "checkpoints" / "ssl" / "convnext_tiny_simclr.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": encoder.backbone.state_dict()}, output_path)
    return str(output_path)


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "train_config.json", asdict(config))
    print(
        f"[pipeline] device={device} image_dir={config.image_dir} output_dir={output_dir}",
        flush=True,
    )

    train_pool, label_cols = combine_train_pool(config.train_csv, config.val_csv)
    test_df, test_label_cols = load_label_frame(config.test_csv)
    if label_cols != test_label_cols:
        raise ValueError("Train/test label columns do not match.")
    print(
        f"[data] train_pool={len(train_pool)} test={len(test_df)} labels={len(label_cols)}",
        flush=True,
    )

    folds = build_cv_folds(train_pool, n_splits=5, seed=config.master_seed)
    save_json(output_dir / "split_manifest.json", split_manifest_payload(train_pool, folds, label_cols))
    save_json(output_dir / "class_stats.json", class_stats_payload(train_pool, test_df, label_cols))
    print("[artifacts] split_manifest.json and class_stats.json written", flush=True)

    base_soft_targets = one_hot(train_pool["label_idx"].to_numpy(), len(label_cols))
    base_sample_weights = np.ones(len(train_pool), dtype=np.float32)

    teacher_result = run_cross_validation(
        train_pool,
        folds=folds,
        image_dir=config.image_dir,
        label_cols=label_cols,
        run_config=config.teacher,
        soft_targets=base_soft_targets,
        sample_weights=base_sample_weights,
        device=device,
        num_workers=config.num_workers,
        stage_name="teacher_convnext",
        output_dir=output_dir,
        seeds=config.teacher_seeds,
    )
    teacher_oof = teacher_result["oof_probs"]
    write_parquet(output_dir / "oof_predictions.parquet", oof_predictions_frame(train_pool, label_cols, teacher_oof))
    print("[artifacts] oof_predictions.parquet written", flush=True)

    class_groups = build_class_groups(train_pool, label_cols)
    noise_targets, noise_weights, noise_map = build_noise_targets(
        train_pool,
        teacher_probs=teacher_oof,
        label_cols=label_cols,
        class_groups=class_groups,
    )
    write_parquet(output_dir / "noise_map.parquet", noise_map)
    print("[artifacts] noise_map.parquet written", flush=True)

    ssl_checkpoint = None
    if teacher_result["metrics"]["mean_auc"] < 0.83 and not config.disable_ssl:
        ssl_checkpoint = run_simclr_pretraining(
            train_pool,
            image_dir=config.image_dir,
            output_dir=output_dir,
            input_size=config.model_a.input_size,
            batch_size=config.ssl_batch_size,
            epochs=config.ssl_epochs,
            temperature=config.ssl_temperature,
            projection_dim=config.ssl_projection_dim,
            device=device,
            num_workers=config.num_workers,
            seed=config.master_seed,
        )
        config.model_a.ssl_checkpoint = ssl_checkpoint

    model_a_result = run_cross_validation(
        train_pool,
        folds=folds,
        image_dir=config.image_dir,
        label_cols=label_cols,
        run_config=config.model_a,
        soft_targets=noise_targets,
        sample_weights=noise_weights,
        device=device,
        num_workers=config.num_workers,
        stage_name="model_a_convnext",
        output_dir=output_dir,
        seeds=[config.master_seed],
    )
    model_b_result = run_cross_validation(
        train_pool,
        folds=folds,
        image_dir=config.image_dir,
        label_cols=label_cols,
        run_config=config.model_b,
        soft_targets=noise_targets,
        sample_weights=noise_weights,
        device=device,
        num_workers=config.num_workers,
        stage_name="model_b_efficientnet",
        output_dir=output_dir,
        seeds=[config.master_seed],
    )

    ensemble_weights = normalize_weights(
        [model_a_result["metrics"]["mean_auc"], model_b_result["metrics"]["mean_auc"]]
    )
    ensemble_oof = (
        ensemble_weights[0] * model_a_result["oof_probs"]
        + ensemble_weights[1] * model_b_result["oof_probs"]
    )
    ensemble_metrics = compute_metrics(ensemble_oof, train_pool["label_idx"].to_numpy(), label_cols)

    metrics_cv = {
        "teacher": teacher_result["metrics"],
        "model_a": model_a_result["metrics"],
        "model_b": model_b_result["metrics"],
        "ensemble": ensemble_metrics,
        "teacher_best_epoch_median": teacher_result["best_epoch_median"],
        "model_a_best_epoch_median": model_a_result["best_epoch_median"],
        "model_b_best_epoch_median": model_b_result["best_epoch_median"],
    }
    save_json(output_dir / "metrics_cv.json", metrics_cv)
    print("[artifacts] metrics_cv.json written", flush=True)

    baseline_checkpoint = train_full_model(
        train_pool,
        image_dir=config.image_dir,
        label_cols=label_cols,
        run_config=config.teacher,
        soft_targets=base_soft_targets,
        sample_weights=base_sample_weights,
        device=device,
        num_workers=config.num_workers,
        seed=config.master_seed,
        checkpoint_path=output_dir / "checkpoints" / "full" / "baseline_convnext.pt",
        epochs=teacher_result["best_epoch_median"],
    )
    model_a_checkpoint = train_full_model(
        train_pool,
        image_dir=config.image_dir,
        label_cols=label_cols,
        run_config=config.model_a,
        soft_targets=noise_targets,
        sample_weights=noise_weights,
        device=device,
        num_workers=config.num_workers,
        seed=config.master_seed,
        checkpoint_path=output_dir / "checkpoints" / "full" / "model_a_convnext.pt",
        epochs=model_a_result["best_epoch_median"],
    )
    model_b_checkpoint = train_full_model(
        train_pool,
        image_dir=config.image_dir,
        label_cols=label_cols,
        run_config=config.model_b,
        soft_targets=noise_targets,
        sample_weights=noise_weights,
        device=device,
        num_workers=config.num_workers,
        seed=config.master_seed,
        checkpoint_path=output_dir / "checkpoints" / "full" / "model_b_efficientnet.pt",
        epochs=model_b_result["best_epoch_median"],
    )

    test_targets = test_df["label_idx"].to_numpy()
    baseline_probs = predict_with_checkpoint(
        baseline_checkpoint,
        test_df,
        image_dir=config.image_dir,
        batch_size=config.teacher.batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    baseline_metrics = compute_metrics(baseline_probs, test_targets, label_cols)

    best_single_name = "model_a" if model_a_result["metrics"]["mean_auc"] >= model_b_result["metrics"]["mean_auc"] else "model_b"
    best_single_checkpoint = model_a_checkpoint if best_single_name == "model_a" else model_b_checkpoint
    best_single_batch_size = config.model_a.batch_size if best_single_name == "model_a" else config.model_b.batch_size
    best_single_probs = predict_with_checkpoint(
        best_single_checkpoint,
        test_df,
        image_dir=config.image_dir,
        batch_size=best_single_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    best_single_metrics = compute_metrics(best_single_probs, test_targets, label_cols)

    model_b_probs = (
        predict_with_checkpoint(
            model_b_checkpoint,
            test_df,
            image_dir=config.image_dir,
            batch_size=config.model_b.batch_size,
            num_workers=config.num_workers,
            device=device,
        )
        if best_single_name == "model_a"
        else best_single_probs
    )
    model_a_probs = best_single_probs if best_single_name == "model_a" else predict_with_checkpoint(
        model_a_checkpoint,
        test_df,
        image_dir=config.image_dir,
        batch_size=config.model_a.batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    ensemble_test_probs = ensemble_weights[0] * model_a_probs + ensemble_weights[1] * model_b_probs
    ensemble_test_metrics = compute_metrics(ensemble_test_probs, test_targets, label_cols)

    metrics_test = {
        "baseline": baseline_metrics,
        "best_single_name": best_single_name,
        "best_single": best_single_metrics,
        "ensemble": ensemble_test_metrics,
        "analysis": {
            "priority_labels": priority_label_table(
                baseline_metrics,
                ensemble_test_metrics,
                labels=[
                    "Pneumomediastinum",
                    "Pleural_Thickening",
                    "Infiltration",
                    "Nodule",
                    "Mass",
                    "No Finding",
                ],
            ),
            "bottom5_improved_count": bottom5_improvement_count(baseline_metrics, ensemble_test_metrics),
        },
    }
    save_json(output_dir / "metrics_test.json", metrics_test)
    print("[artifacts] metrics_test.json written", flush=True)

    write_deployment_manifest(
        output_dir,
        label_cols=label_cols,
        model_a_checkpoint=model_a_checkpoint,
        model_b_checkpoint=model_b_checkpoint,
        model_a_weight=ensemble_weights[0],
        model_b_weight=ensemble_weights[1],
        mean_auc=ensemble_test_metrics["mean_auc"],
    )
    print("[artifacts] deployment_manifest.json written", flush=True)

    return {
        "output_dir": str(output_dir),
        "metrics_cv": metrics_cv,
        "metrics_test": metrics_test,
        "ssl_checkpoint": ssl_checkpoint,
    }


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Train the NIH noise-aware long-tail pipeline.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--disable-ssl", action="store_true")
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--main-batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    teacher = RunConfig("convnext_tiny", 448, batch_size=args.teacher_batch_size)
    model_a = RunConfig("convnext_tiny", 448, batch_size=args.main_batch_size)
    model_b = RunConfig("efficientnet_v2_s", 384, batch_size=args.main_batch_size)
    return PipelineConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.num_workers,
        master_seed=args.master_seed,
        teacher=teacher,
        model_a=model_a,
        model_b=model_b,
        disable_ssl=args.disable_ssl,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    config = parse_args(argv)
    results = run_pipeline(config)
    print(json.dumps(results["metrics_test"], indent=2))
    return results


if __name__ == "__main__":
    main()
