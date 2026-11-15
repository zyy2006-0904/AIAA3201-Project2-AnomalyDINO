#!/usr/bin/env python3
"""Export representative low-scoring defects and high-scoring normal images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anomalies", type=int, default=3)
    parser.add_argument("--normals", type=int, default=2)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    measurements_path = args.run_dir / f"measurements_seed={args.seed}.csv"
    with measurements_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["Anomaly_Score"] = float(row["Anomaly_Score"])

    anomaly_rows = sorted(
        (row for row in rows if row["Sample"].startswith("bad/")),
        key=lambda row: row["Anomaly_Score"],
    )[: args.anomalies]
    normal_rows = sorted(
        (row for row in rows if row["Sample"].startswith("good/")),
        key=lambda row: row["Anomaly_Score"],
        reverse=True,
    )[: args.normals]
    selected = [
        {**row, "failure_type": "low-scoring defect"}
        for row in anomaly_rows
    ] + [
        {**row, "failure_type": "high-scoring normal"}
        for row in normal_rows
    ]

    exported = []
    map_root = args.run_dir / "anomaly_maps" / f"seed={args.seed}"
    for index, row in enumerate(selected, start=1):
        category = row["Object"]
        split, filename = row["Sample"].split("/", 1)
        stem = Path(filename).stem
        category_root = args.data_root / category
        image_path = category_root / "test" / split / filename
        mask_path = category_root / "ground_truth" / "bad" / f"{stem}.png"
        map_path = map_root / category / "test" / split / f"{stem}.npy"
        image = read_rgb(image_path)
        patch_map = np.load(map_path)
        anomaly_map = cv2.resize(
            patch_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        if split == "bad":
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(image)
        axes[0].set_title("Input")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground-truth mask")
        image_map = axes[2].imshow(anomaly_map, cmap="turbo")
        axes[2].set_title("Patch-neighbor anomaly map")
        axes[3].imshow(image)
        axes[3].imshow(
            anomaly_map,
            cmap="turbo",
            alpha=0.45,
            vmin=float(anomaly_map.min()),
            vmax=float(anomaly_map.max()),
        )
        axes[3].set_title(
            f"{row['failure_type']}\nscore={row['Anomaly_Score']:.4f}"
        )
        for axis in axes:
            axis.axis("off")
        fig.colorbar(
            image_map,
            ax=axes[2],
            fraction=0.046,
            pad=0.02,
            orientation="horizontal",
        )
        fig.suptitle(f"{category}: {args.run_dir.name}")
        fig.tight_layout()
        destination = args.output / f"{index:02d}_{category}_{split}_{stem}.png"
        fig.savefig(destination, dpi=220)
        plt.close(fig)
        exported.append(
            {
                "file": str(destination),
                "category": category,
                "sample": row["Sample"],
                "score": row["Anomaly_Score"],
                "failure_type": row["failure_type"],
            }
        )

    (args.output / "failure_cases.json").write_text(
        json.dumps(exported, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(exported)} failure cases to {args.output}")


if __name__ == "__main__":
    main()
