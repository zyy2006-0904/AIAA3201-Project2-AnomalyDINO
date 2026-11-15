#!/usr/bin/env python3
"""Build the official VisA 1cls layout without duplicating the image archive."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe_fryum",
)
CALIBRATION_COUNT = 20
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/data/disk1/shared/aiaa3201/project2_anomaly/VisA"),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path(
            "/data/disk1/shared/aiaa3201/project2_anomaly/VisA/split_csv/1cls.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "aiaa3201_project2/data/VisA_pytorch/1cls",
    )
    return parser.parse_args()


def symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(
            f"Refusing to replace non-symlink file: {destination}"
        )
    destination.symlink_to(source.resolve())


def image_name(path: Path) -> str:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path}")
    return path.name


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {
        category: {
            "train_normal": [],
            "test_normal": [],
            "test_anomaly": [],
            "masks": [],
        }
        for category in CATEGORIES
    }

    with args.split_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    for row in rows:
        category = row["object"]
        if category not in CATEGORIES:
            continue

        split = row["split"]
        label = row["label"]
        source_image = args.source / row["image"]
        if not source_image.is_file():
            raise FileNotFoundError(source_image)

        if label == "normal":
            output_label = "good"
            counts[category][f"{split}_normal"].append(source_image.name)
        else:
            output_label = "bad"
            counts[category]["test_anomaly"].append(source_image.name)

        destination_image = (
            args.output
            / category
            / split
            / output_label
            / image_name(source_image)
        )
        symlink(source_image, destination_image)

        if label == "anomaly":
            if not row["mask"]:
                raise ValueError(f"Missing mask path for {source_image}")
            source_mask = args.source / row["mask"]
            if not source_mask.is_file():
                raise FileNotFoundError(source_mask)
            destination_mask = (
                args.output
                / category
                / "ground_truth"
                / "bad"
                / source_mask.name
            )
            symlink(source_mask, destination_mask)
            counts[category]["masks"].append(source_mask.name)

    manifest = {
        "source": str(args.source),
        "split_file": str(args.split_file),
        "output": str(args.output),
        "categories": list(CATEGORIES),
        "calibration_count": CALIBRATION_COUNT,
        "calibration_policy": (
            "The first 20 lexicographically sorted training filenames per "
            "category are reserved for threshold calibration and excluded "
            "from all few-shot reference selections."
        ),
        "counts": {},
    }

    for category in CATEGORIES:
        for key in ("train_normal", "test_normal", "test_anomaly", "masks"):
            counts[category][key] = sorted(counts[category][key])
        train_count = len(counts[category]["train_normal"])
        if train_count <= CALIBRATION_COUNT:
            raise ValueError(f"Not enough training images for {category}")
        manifest["counts"][category] = {
            key: len(value) for key, value in counts[category].items()
        }
        manifest["counts"][category]["calibration_images"] = counts[category][
            "train_normal"
        ][:CALIBRATION_COUNT]

    broken = [
        path
        for path in args.output.rglob("*")
        if path.is_symlink() and not path.exists()
    ]
    if broken:
        raise RuntimeError(f"Broken symlinks detected, first: {broken[0]}")

    manifest_path = args.output.parent.parent / "prepared_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Prepared dataset: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
