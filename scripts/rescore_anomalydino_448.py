#!/usr/bin/env python3
"""Re-evaluate saved AnomalyDINO patch maps at a 448-pixel common grid."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_pipeline.metrics import image_metrics, pixel_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--eval-size", type=int, default=448)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def expand_map(path: Path, size: int) -> np.ndarray:
    patch_map = np.load(path).astype(np.float32)
    anomaly_map = cv2.resize(
        patch_map,
        (size, size),
        interpolation=cv2.INTER_LINEAR,
    )
    return gaussian_filter(anomaly_map, sigma=4)


def main() -> None:
    args = parse_args()
    output = args.output or args.run_dir / f"metrics_448_seed={args.seed}.json"
    maps_root = args.run_dir / "anomaly_maps" / f"seed={args.seed}"
    with (args.run_dir / f"measurements_seed={args.seed}.csv").open(
        newline="",
        encoding="utf-8",
    ) as stream:
        measurement_rows = list(csv.DictReader(stream))

    per_category = {}
    for category_dir in sorted(maps_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        scores: list[float] = []
        labels: list[int] = []
        maps: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        for split, label in (("good", 0), ("bad", 1)):
            folder = category_dir / "test" / split
            for path in sorted(folder.glob("*.npy")):
                sample = f"{split}/{path.stem}.JPG"
                measurement = next(
                    (
                        row
                        for row in measurement_rows
                        if row["Object"] == category and row["Sample"] == sample
                    ),
                    None,
                )
                if measurement is None:
                    raise KeyError(f"No score found for {category}/{sample}")
                scores.append(float(measurement["Anomaly_Score"]))
                labels.append(label)
                maps.append(expand_map(path, args.eval_size))
                if label:
                    mask_path = (
                        args.data_root
                        / category
                        / "ground_truth"
                        / "bad"
                        / f"{path.stem}.png"
                    )
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise FileNotFoundError(mask_path)
                    mask = cv2.resize(
                        mask,
                        (args.eval_size, args.eval_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    masks.append((mask > 0).astype(np.uint8))
                else:
                    masks.append(
                        np.zeros((args.eval_size, args.eval_size), dtype=np.uint8)
                    )

        image_result = image_metrics(scores, labels)
        pixel_result = pixel_metrics(maps, masks)
        per_category[category] = {
            "classification_AUROC": image_result["image_auroc"],
            "classification_AP": image_result["image_ap"],
            "classification_F1": image_result["image_f1"],
            "seg_AUROC": pixel_result["pixel_auroc"],
        }

    payload = {
        "eval_size": args.eval_size,
        "mean_classification_au_roc": float(
            np.mean([row["classification_AUROC"] for row in per_category.values()])
        ),
        "mean_classification_ap": float(
            np.mean([row["classification_AP"] for row in per_category.values()])
        ),
        "mean_classification_f1": float(
            np.mean([row["classification_F1"] for row in per_category.values()])
        ),
        "mean_segmentation_au_roc": float(
            np.mean([row["seg_AUROC"] for row in per_category.values()])
        ),
        **per_category,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
