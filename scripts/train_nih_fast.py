"""Fast NIH trainer targeting sub-hour turnaround on a single machine."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nih_pipeline.modeling import display_name
from nih_pipeline.pipeline import (
    CXRSingleLabelDataset,
    EMA,
    RunConfig,
    autocast_context,
    build_grad_scaler,
    build_logit_adjustment,
    build_model,
    build_transforms,
    checkpoint_payload,
    compute_metrics,
    load_label_frame,
    resolve_device,
    run_epoch,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast NIH single-model trainer.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--architecture", default="efficientnet_v2_s", choices=["efficientnet_v2_s", "convnext_tiny", "densenet121"])
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-cap", type=int, default=900)
    parser.add_argument("--medium-cap", type=int, default=400)
    parser.add_argument("--tail-keep-threshold", type=int, default=200)
    parser.add_argument("--no-finding-cap", type=int, default=2500)
    parser.add_argument("--mixup-alpha", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-history")
    parser.add_argument("--resume-lr-scale", type=float, default=0.5)
    return parser.parse_args()


def smart_subset(
    frame: pd.DataFrame,
    *,
    seed: int,
    head_cap: int,
    medium_cap: int,
    tail_keep_threshold: int,
    no_finding_cap: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rng = np.random.default_rng(seed)
    counts = frame["label_name"].value_counts()
    selected = []
    caps = {}
    for label, count in counts.items():
        class_rows = frame[frame["label_name"] == label]
        if count <= tail_keep_threshold:
            keep = count
        elif count <= 1000:
            keep = min(count, medium_cap)
        elif label == "No Finding":
            keep = min(count, no_finding_cap)
        else:
            keep = min(count, head_cap)
        caps[label] = int(keep)
        if keep >= count:
            selected.append(class_rows)
        else:
            sampled_idx = rng.choice(class_rows.index.to_numpy(), size=keep, replace=False)
            selected.append(class_rows.loc[sampled_idx])
    subset = pd.concat(selected, ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return subset, caps


def smoothed_targets(label_indices: np.ndarray, num_classes: int, smoothing: float) -> np.ndarray:
    base = np.full((len(label_indices), num_classes), smoothing / num_classes, dtype=np.float32)
    base[np.arange(len(label_indices)), label_indices] = 1.0 - smoothing + (smoothing / num_classes)
    return base


def make_loader(
    frame: pd.DataFrame,
    *,
    image_dir: str,
    input_size: int,
    grayscale: bool,
    batch_size: int,
    num_workers: int,
    train: bool,
    soft_targets: np.ndarray,
    sample_weights: np.ndarray,
) -> DataLoader:
    dataset = CXRSingleLabelDataset(
        frame,
        image_dir,
        transform=build_transforms(input_size, train=train, grayscale=grayscale),
        soft_targets=soft_targets,
        sample_weights=sample_weights,
        grayscale=grayscale,
    )
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if train:
        class_counts = np.bincount(frame["label_idx"].to_numpy(), minlength=soft_targets.shape[1]).astype(np.float32)
        class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
        sampler_weights = class_weights[frame["label_idx"].to_numpy()] * sample_weights
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sampler_weights, dtype=torch.double),
            num_samples=len(sampler_weights),
            replacement=True,
        )
        return DataLoader(dataset, sampler=sampler, **kwargs)
    return DataLoader(dataset, shuffle=False, **kwargs)


def evaluate_fast(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str,
    logit_adjustment: torch.Tensor,
    label_cols: list[str],
) -> tuple[float, dict[str, object], np.ndarray]:
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
            with autocast_context(device):
                logits = model(images)
                logits = logits + logit_adjustment
                log_probs = torch.log_softmax(logits, dim=1)
                losses = -(soft_targets * log_probs).sum(dim=1)
                loss = (losses * sample_weights).mean()
            total_loss += float(loss.item()) * images.shape[0]
            total_count += images.shape[0]
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(batch["target_idx"].numpy())
    probs = np.concatenate(probabilities, axis=0)
    target_indices = np.concatenate(targets, axis=0)
    metrics = compute_metrics(probs, target_indices, label_cols)
    return total_loss / max(total_count, 1), metrics, probs


def predict_dataset(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    *,
    image_dir: str,
    input_size: int,
    grayscale: bool,
    batch_size: int,
    num_workers: int,
    device: str,
    logit_adjustment: torch.Tensor,
    tta: bool,
) -> np.ndarray:
    dummy_targets = smoothed_targets(frame["label_idx"].to_numpy(), len(frame["label_idx"].unique()) if False else len(logit_adjustment), 0.0)
    dataset = CXRSingleLabelDataset(
        frame,
        image_dir,
        transform=build_transforms(input_size, train=False, grayscale=grayscale),
        soft_targets=dummy_targets,
        sample_weights=np.ones(len(frame), dtype=np.float32),
        grayscale=grayscale,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images) + logit_adjustment
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    if not tta:
        return probs

    tta_dataset = CXRSingleLabelDataset(
        frame,
        image_dir,
        transform=torchvision_tta_transform(input_size),
        soft_targets=dummy_targets,
        sample_weights=np.ones(len(frame), dtype=np.float32),
        grayscale=grayscale,
    )
    tta_loader = DataLoader(
        tta_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    tta_probs = []
    with torch.no_grad():
        for batch in tta_loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images) + logit_adjustment
            tta_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return 0.5 * (probs + np.concatenate(tta_probs, axis=0))


def torchvision_tta_transform(input_size: int):
    import torchvision.transforms as T
    from nih_pipeline.modeling import default_normalization

    mean, std = default_normalization(True)
    return T.Compose(
        [
            T.Resize((int(round(input_size * 1.08)), int(round(input_size * 1.08)))),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, label_cols = load_label_frame(args.train_csv)
    val_df, val_label_cols = load_label_frame(args.val_csv)
    test_df, test_label_cols = load_label_frame(args.test_csv)
    if label_cols != val_label_cols or label_cols != test_label_cols:
        raise ValueError("Label columns mismatch across CSVs.")

    train_subset, subset_caps = smart_subset(
        train_df,
        seed=args.seed,
        head_cap=args.head_cap,
        medium_cap=args.medium_cap,
        tail_keep_threshold=args.tail_keep_threshold,
        no_finding_cap=args.no_finding_cap,
    )
    subset_summary = {
        "full_train_rows": int(len(train_df)),
        "subset_rows": int(len(train_subset)),
        "caps": subset_caps,
        "subset_counts": train_subset["label_name"].value_counts().reindex(label_cols, fill_value=0).to_dict(),
    }
    save_json(output_dir / "subset_manifest.json", subset_summary)

    run_config = RunConfig(
        architecture=args.architecture,
        input_size=args.input_size,
        epochs=args.epochs,
        warmup_epochs=1,
        batch_size=args.batch_size,
        learning_rate=3e-4,
        weight_decay=1e-4,
        mixup_alpha=args.mixup_alpha,
        patience=3,
        ema_decay=0.999,
        logit_tau=1.0,
        use_tta=args.tta,
        pretrained=True,
        grayscale=True,
    )

    num_classes = len(label_cols)
    train_targets = smoothed_targets(train_subset["label_idx"].to_numpy(), num_classes, args.label_smoothing)
    val_targets = smoothed_targets(val_df["label_idx"].to_numpy(), num_classes, 0.0)
    train_weights = np.ones(len(train_subset), dtype=np.float32)
    val_weights = np.ones(len(val_df), dtype=np.float32)

    train_loader = make_loader(
        train_subset,
        image_dir=args.image_dir,
        input_size=args.input_size,
        grayscale=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
        soft_targets=train_targets,
        sample_weights=train_weights,
    )
    val_loader = make_loader(
        val_df,
        image_dir=args.image_dir,
        input_size=args.input_size,
        grayscale=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
        soft_targets=val_targets,
        sample_weights=val_weights,
    )

    model = build_model(args.architecture, num_classes=num_classes, in_channels=1, pretrained=True).to(device)
    start_epoch = 0
    best_auc = -1.0
    best_epoch = 0
    history = []
    scaler = build_grad_scaler(device)
    ema = EMA(model, decay=run_config.ema_decay)
    logit_adjustment = build_logit_adjustment(train_subset["label_idx"].to_numpy(), run_config.logit_tau, num_classes, device)
    checkpoint_path = output_dir / "best_fast_model.pt"

    learning_rate = run_config.learning_rate
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        ema.shadow.load_state_dict(checkpoint["model_state"])
        checkpoint_path = Path(args.resume_checkpoint)
        learning_rate *= args.resume_lr_scale
        if args.resume_history:
            history_payload = json.loads(Path(args.resume_history).read_text())
            history = history_payload.get("history", [])
            best_epoch = int(history_payload.get("best_epoch", len(history)))
            best_auc = float(history_payload.get("best_val_auc", -1.0))
            start_epoch = len(history)
        print(
            f"[fast-resume] checkpoint={args.resume_checkpoint} start_epoch={start_epoch} "
            f"best_epoch={best_epoch} best_val_auc={best_auc:.4f} lr_scale={args.resume_lr_scale}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=run_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    print(
        f"[fast] device={device} arch={args.architecture} size={args.input_size} "
        f"subset={len(train_subset)} val={len(val_df)} test={len(test_df)} batch={args.batch_size}",
        flush=True,
    )
    print(f"[fast] display_name={display_name(args.architecture)} output_dir={output_dir}", flush=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
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
        eval_model = ema.shadow
        val_loss, val_metrics, _ = evaluate_fast(
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
                "val_mean_auc": val_metrics["mean_auc"],
                "val_mean_ap": val_metrics["mean_ap"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"[fast-epoch] epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_metrics['mean_auc']:.4f} "
            f"val_ap={val_metrics['mean_ap']:.4f} val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["mean_auc"] > best_auc:
            best_auc = val_metrics["mean_auc"]
            best_epoch = epoch
            torch.save(
                checkpoint_payload(
                    eval_model,
                    run_config=run_config,
                    label_cols=label_cols,
                    class_priors=np.bincount(train_subset["label_idx"].to_numpy(), minlength=num_classes).astype(np.float32),
                ),
                checkpoint_path,
            )
            print(f"[fast-best] epoch={epoch} val_auc={best_auc:.4f} checkpoint={checkpoint_path}", flush=True)

    save_json(output_dir / "metrics_val_fast.json", {"best_epoch": best_epoch, "best_val_auc": best_auc, "history": history})

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    best_model = build_model(checkpoint["architecture"], num_classes=num_classes, in_channels=1, pretrained=False).to(device)
    best_model.load_state_dict(checkpoint["model_state"])
    test_logit_adjustment = build_logit_adjustment(train_subset["label_idx"].to_numpy(), run_config.logit_tau, num_classes, device)
    test_probs = predict_dataset(
        best_model,
        test_df,
        image_dir=args.image_dir,
        input_size=args.input_size,
        grayscale=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        logit_adjustment=test_logit_adjustment,
        tta=args.tta,
    )
    test_metrics = compute_metrics(test_probs, test_df["label_idx"].to_numpy(), label_cols)
    save_json(output_dir / "metrics_test_fast.json", test_metrics)
    print(json.dumps(test_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
