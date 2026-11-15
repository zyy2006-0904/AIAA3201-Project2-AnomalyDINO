#!/usr/bin/env python3
"""Required normal-only template-difference baseline for AIAA 3201 Project 2."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe_fryum",
)
CALIBRATION_COUNT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--save-maps-shot", type=int, default=4)
    parser.add_argument("--save-maps-seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in extensions
    )


def load_image(path: Path, resolution: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(
        image,
        (resolution, resolution),
        interpolation=cv2.INTER_AREA,
    )


def load_mask(path: Path | None, resolution: int) -> np.ndarray:
    if path is None:
        return np.zeros((resolution, resolution), dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(
        mask,
        (resolution, resolution),
        interpolation=cv2.INTER_NEAREST,
    )
    return (mask > 0).astype(np.uint8)


def mean_top_fraction(values: np.ndarray, fraction: float = 0.01) -> float:
    flat = values.reshape(-1)
    count = max(1, int(flat.size * fraction))
    if count == flat.size:
        return float(flat.mean())
    return float(np.partition(flat, flat.size - count)[-count:].mean())


def score_against_templates(
    image: np.ndarray,
    references: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    differences = np.mean(
        np.abs(references - image[None, ...]),
        axis=3,
        dtype=np.float32,
    )
    closest = int(np.argmin(differences.mean(axis=(1, 2))))
    difference_map = cv2.GaussianBlur(
        differences[closest],
        (0, 0),
        sigmaX=4,
        sigmaY=4,
    )
    return mean_top_fraction(difference_map), difference_map, closest


def mean_metrics(per_category: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = (
        "image_auroc",
        "image_ap",
        "image_f1",
        "pixel_auroc",
        "pixel_ap",
        "mean_inference_seconds",
    )
    return {
        f"mean_{key}": float(np.mean([metrics[key] for metrics in per_category.values()]))
        for key in keys
    }


def evaluate_category(
    data_root: Path,
    category: str,
    references: np.ndarray,
    reference_names: list[str],
    calibration_paths: list[Path],
    resolution: int,
    max_images: int | None,
) -> tuple[dict[str, float], list[dict[str, object]], np.ndarray, np.ndarray, float]:
    calibration_scores = []
    for path in calibration_paths:
        score, _, _ = score_against_templates(
            load_image(path, resolution),
            references,
        )
        calibration_scores.append(score)
    threshold = float(np.percentile(calibration_scores, 95))

    image_scores: list[float] = []
    image_labels: list[int] = []
    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    measurements: list[dict[str, object]] = []
    inference_times: list[float] = []

    for split_dir, label in (("good", 0), ("bad", 1)):
        folder = data_root / category / "test" / split_dir
        paths = image_files(folder)
        if max_images is not None:
            paths = paths[:max_images]
        for path in paths:
            started = time.perf_counter()
            image = load_image(path, resolution)
            score, anomaly_map, closest = score_against_templates(
                image,
                references,
            )
            inference_times.append(time.perf_counter() - started)
            image_scores.append(score)
            image_labels.append(label)
            pixel_scores.append(anomaly_map)

            mask_path = None
            if label:
                mask_path = (
                    data_root
                    / category
                    / "ground_truth"
                    / "bad"
                    / f"{path.stem}.png"
                )
            pixel_labels.append(load_mask(mask_path, resolution))
            measurements.append(
                {
                    "category": category,
                    "image": str(path.relative_to(data_root)),
                    "label": "normal" if label == 0 else "anomaly",
                    "image_score": score,
                    "threshold": threshold,
                    "decision": "NG" if score > threshold else "OK",
                    "closest_reference": reference_names[closest],
                    "inference_seconds": inference_times[-1],
                }
            )

    image_score_array = np.asarray(image_scores, dtype=np.float64)
    image_label_array = np.asarray(image_labels, dtype=np.uint8)
    pixel_score_array = np.stack(pixel_scores)
    pixel_label_array = np.stack(pixel_labels)
    precision, recall, _ = precision_recall_curve(
        image_label_array,
        image_score_array,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    metrics = {
        "image_auroc": float(roc_auc_score(image_label_array, image_score_array)),
        "image_ap": float(
            average_precision_score(image_label_array, image_score_array)
        ),
        "image_f1": float(np.max(f1)),
        "pixel_auroc": float(
            roc_auc_score(pixel_label_array.ravel(), pixel_score_array.ravel())
        ),
        "pixel_ap": float(
            average_precision_score(
                pixel_label_array.ravel(),
                pixel_score_array.ravel(),
            )
        ),
        "calibration_threshold": threshold,
        "mean_inference_seconds": float(np.mean(inference_times)),
        "num_test_images": len(image_scores),
    }
    return (
        metrics,
        measurements,
        pixel_score_array,
        pixel_label_array,
        threshold,
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for shot in args.shots:
        for seed in args.seeds:
            run_dir = args.output / f"{shot}-shot" / f"seed={seed}"
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                print(f"Skipping completed run: {run_dir}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            per_category: dict[str, dict[str, float]] = {}
            all_measurements: list[dict[str, object]] = []

            for category in args.categories:
                train_folder = args.data_root / category / "train" / "good"
                training_images = image_files(train_folder)
                calibration_paths = training_images[:CALIBRATION_COUNT]
                eligible_paths = training_images[CALIBRATION_COUNT:]
                selected_paths = eligible_paths[
                    seed * shot : (seed + 1) * shot
                ]
                if len(selected_paths) != shot:
                    raise ValueError(
                        f"Need {shot} references for {category}, "
                        f"found {len(selected_paths)}"
                    )
                references = np.stack(
                    [load_image(path, args.resolution) for path in selected_paths]
                )
                reference_names = [path.name for path in selected_paths]
                reference_metadata = {
                    "category": category,
                    "shot": shot,
                    "seed": seed,
                    "calibration_images": [
                        path.name for path in calibration_paths
                    ],
                    "reference_images": reference_names,
                }
                references_dir = run_dir / "references"
                references_dir.mkdir(exist_ok=True)
                (references_dir / f"{category}.json").write_text(
                    json.dumps(reference_metadata, indent=2) + "\n",
                    encoding="utf-8",
                )

                print(
                    f"Template baseline: category={category}, "
                    f"shot={shot}, seed={seed}"
                )
                (
                    category_metrics,
                    measurements,
                    pixel_maps,
                    pixel_masks,
                    _,
                ) = evaluate_category(
                    data_root=args.data_root,
                    category=category,
                    references=references,
                    reference_names=reference_names,
                    calibration_paths=calibration_paths,
                    resolution=args.resolution,
                    max_images=args.max_images,
                )
                per_category[category] = category_metrics
                all_measurements.extend(measurements)

                if (
                    shot == args.save_maps_shot
                    and seed == args.save_maps_seed
                    and args.max_images is None
                ):
                    maps_dir = args.output / "maps" / category
                    maps_dir.mkdir(parents=True, exist_ok=True)
                    for measurement, anomaly_map, mask in zip(
                        measurements,
                        pixel_maps,
                        pixel_masks,
                        strict=True,
                    ):
                        relative = Path(str(measurement["image"]))
                        stem = "__".join(relative.with_suffix("").parts)
                        normalized_map = cv2.normalize(
                            anomaly_map,
                            None,
                            0,
                            255,
                            cv2.NORM_MINMAX,
                        ).astype(np.uint8)
                        cv2.imwrite(
                            str(maps_dir / f"{stem}_heatmap.png"),
                            normalized_map,
                        )
                        cv2.imwrite(
                            str(maps_dir / f"{stem}_mask.png"),
                            mask * 255,
                        )

            metrics = {
                "method": "template_difference",
                "shot": shot,
                "seed": seed,
                "resolution": args.resolution,
                **mean_metrics(per_category),
                "categories": per_category,
            }
            metrics_path.write_text(
                json.dumps(metrics, indent=2) + "\n",
                encoding="utf-8",
            )
            measurements_path = run_dir / "measurements.csv"
            with measurements_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=list(all_measurements[0]),
                )
                writer.writeheader()
                writer.writerows(all_measurements)
            print(f"Wrote {metrics_path}")

    print("Template baseline finished.")


if __name__ == "__main__":
    main()
