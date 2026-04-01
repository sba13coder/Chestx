from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from nih_pipeline.modeling import build_model
from nih_pipeline.pipeline import (
    build_class_groups,
    build_cv_folds,
    build_noise_targets,
    compute_metrics,
    combine_train_pool,
    load_label_frame,
    split_manifest_payload,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


class NIHSmokeTests(unittest.TestCase):
    def test_model_builders_forward(self):
        x = torch.randn(2, 1, 64, 64)
        for architecture in ("convnext_tiny", "efficientnet_v2_s", "densenet121"):
            model = build_model(architecture, num_classes=3, in_channels=1, pretrained=False)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (2, 3))

    def test_split_and_noise_helpers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rows = [
                {"id": "img_1.png", "subject_id": "s1", "A": 1, "B": 0, "C": 0},
                {"id": "img_2.png", "subject_id": "s2", "A": 0, "B": 1, "C": 0},
                {"id": "img_3.png", "subject_id": "s3", "A": 0, "B": 0, "C": 1},
                {"id": "img_4.png", "subject_id": "s4", "A": 1, "B": 0, "C": 0},
                {"id": "img_5.png", "subject_id": "s5", "A": 0, "B": 1, "C": 0},
                {"id": "img_6.png", "subject_id": "s6", "A": 0, "B": 0, "C": 1},
            ]
            train_csv = tmp / "train.csv"
            val_csv = tmp / "val.csv"
            _write_csv(train_csv, rows[:3])
            _write_csv(val_csv, rows[3:])

            for row in rows:
                Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(tmp / row["id"])

            combined, label_cols = combine_train_pool(str(train_csv), str(val_csv))
            loaded, loaded_cols = load_label_frame(str(train_csv))
            self.assertEqual(label_cols, loaded_cols)
            self.assertEqual(len(combined), 6)
            self.assertEqual(len(loaded), 3)

            folds = build_cv_folds(combined, n_splits=2, seed=42)
            manifest = split_manifest_payload(combined, folds, label_cols)
            self.assertEqual(manifest["n_splits"], 2)
            self.assertEqual(len(manifest["assignments"]), 6)

            probs = np.array(
                [
                    [0.90, 0.05, 0.05],
                    [0.05, 0.80, 0.15],
                    [0.05, 0.10, 0.85],
                    [0.25, 0.60, 0.15],
                    [0.05, 0.20, 0.75],
                    [0.70, 0.10, 0.20],
                ],
                dtype=np.float32,
            )
            groups = build_class_groups(combined, label_cols)
            soft_targets, sample_weights, noise_map = build_noise_targets(
                combined,
                teacher_probs=probs,
                label_cols=label_cols,
                class_groups=groups,
            )
            self.assertEqual(soft_targets.shape, probs.shape)
            self.assertEqual(sample_weights.shape[0], len(combined))
            self.assertEqual(set(noise_map.columns), {
                "image_id",
                "subject_id",
                "label",
                "teacher_top1",
                "teacher_prob_label",
                "teacher_prob_top1",
                "status",
                "sample_weight",
            })

            metrics = compute_metrics(probs, combined["label_idx"].to_numpy(), label_cols)
            self.assertIn("mean_auc", metrics)
            self.assertIn("macro_f1", metrics)
            self.assertEqual(set(metrics["per_auc"]), set(label_cols))


if __name__ == "__main__":
    unittest.main()
