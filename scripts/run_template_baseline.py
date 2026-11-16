#!/usr/bin/env python3
"""Run the integrated Part 1 template-difference baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_pipeline.baseline import DEFAULT_CHANNELS, evaluate_class, list_images


CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe_fryum",
)
CALIBRATION_COUNT = 20
METRIC_KEYS = (
    "image_auroc",
    "image_ap",
    "image_f1",
    "pixel_auroc",
    "mean_inference_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=DEFAULT_CHANNELS,
        default=list(DEFAULT_CHANNELS),
    )
    parser.add_argument(
        "--template-mode",
        choices=("global", "per_pixel"),
        default="global",
        help=(
            "global selects one closest template as required by the handout; "
            "per_pixel takes the minimum difference independently at each pixel."
        ),
    )
    parser.add_argument(
        "--score-mode",
        choices=("percentile", "top1"),
        default="percentile",
    )
    parser.add_argument("--score-percentile", type=float, default=99.5)
    parser.add_argument("--border-crop", type=int, default=0)
    parser.add_argument("--ecc", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def aggregate_metrics(per_category: dict[str, dict]) -> dict[str, float]:
    return {
        f"mean_{key}": float(
            np.mean([metrics[key] for metrics in per_category.values()])
        )
        for key in METRIC_KEYS
    }


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
            per_category: dict[str, dict] = {}
            all_records: list[dict] = []

            for category in args.categories:
                train_folder = args.data_root / category / "train" / "good"
                training = list_images(train_folder)
                if len(training) < CALIBRATION_COUNT + shot * (seed + 1):
                    raise ValueError(f"Not enough training images for {category}")
                calibration = training[:CALIBRATION_COUNT]
                pool = training[CALIBRATION_COUNT:]
                references = pool[seed * shot : (seed + 1) * shot]

                (run_dir / "references").mkdir(exist_ok=True)
                (run_dir / "references" / f"{category}.json").write_text(
                    json.dumps(
                        {
                            "category": category,
                            "shot": shot,
                            "seed": seed,
                            "calibration_images": calibration,
                            "reference_images": references,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                print(f"Template: {category} {shot}-shot seed={seed}", flush=True)
                category_result = evaluate_class(
                    str(args.data_root),
                    category,
                    references,
                    calibration,
                    size=args.resolution,
                    channels=tuple(args.channels),
                    template_mode=args.template_mode,
                    score_mode=args.score_mode,
                    score_percentile=args.score_percentile,
                    border_crop=args.border_crop,
                    use_ecc=args.ecc,
                    max_images=args.max_images,
                )
                records = category_result.pop("records")
                all_records.extend({"category": category, **row} for row in records)
                per_category[category] = category_result

            decisions = np.asarray(
                [row["decision"] == "NG" for row in all_records],
                dtype=bool,
            )
            labels = np.asarray(
                [row["file"].startswith("bad/") for row in all_records],
                dtype=bool,
            )
            metrics = {
                "method": "integrated_template_difference",
                "shot": shot,
                "seed": seed,
                "resolution": args.resolution,
                "config": {
                    "channels": args.channels,
                    "template_mode": args.template_mode,
                    "score_mode": args.score_mode,
                    "score_percentile": args.score_percentile,
                    "border_crop": args.border_crop,
                    "ecc": args.ecc,
                },
                **aggregate_metrics(per_category),
                "test_false_positive_rate": float(
                    decisions[~labels].mean() if np.any(~labels) else 0.0
                ),
                "test_false_negative_rate": float(
                    (~decisions[labels]).mean() if np.any(labels) else 0.0
                ),
                "categories": per_category,
            }
            metrics_path.write_text(
                json.dumps(metrics, indent=2) + "\n",
                encoding="utf-8",
            )
            with (run_dir / "measurements.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
                writer.writeheader()
                writer.writerows(all_records)
            print(f"Wrote {metrics_path}", flush=True)

    print("Integrated template baseline finished.")


if __name__ == "__main__":
    main()
