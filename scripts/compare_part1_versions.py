#!/usr/bin/env python3
"""Summarise the performance change from the original to integrated Part 1."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--new-summary", type=Path, required=True)
    parser.add_argument("--new-template-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def old_results(archive: Path) -> dict[int, dict[str, float]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    rate_groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.name.startswith("template_results/"):
                continue
            if member.name.endswith("/metrics.json"):
                shot = int(member.name.split("/")[1].split("-")[0])
                groups[shot].append(json.load(handle.extractfile(member)))
            elif member.name.endswith("/measurements.csv"):
                shot = int(member.name.split("/")[1].split("-")[0])
                stream = io.TextIOWrapper(handle.extractfile(member), encoding="utf-8")
                rows = list(csv.DictReader(stream))
                normal = [row for row in rows if row["label"] == "normal"]
                anomaly = [row for row in rows if row["label"] == "anomaly"]
                rate_groups[shot].append(
                    (
                        sum(row["decision"] == "NG" for row in normal) / len(normal),
                        sum(row["decision"] == "OK" for row in anomaly) / len(anomaly),
                    )
                )
    results = {}
    for shot, runs in groups.items():
        results[shot] = {
            "image_auroc": float(np.mean([run["mean_image_auroc"] for run in runs])),
            "pixel_auroc": float(np.mean([run["mean_pixel_auroc"] for run in runs])),
            "false_positive_rate": float(
                np.mean([rates[0] for rates in rate_groups[shot]])
            ),
            "false_negative_rate": float(
                np.mean([rates[1] for rates in rate_groups[shot]])
            ),
        }
    return results


def new_results(
    path: Path,
    template_root: Path,
) -> dict[int, dict[str, float]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    values = defaultdict(dict)
    metric_names = {
        "mean_classification_au_roc": "image_auroc",
        "mean_segmentation_au_roc": "pixel_auroc",
    }
    for row in rows:
        if row["method"] != "template":
            continue
        if row["metric"] in metric_names:
            values[int(row["shot"])][metric_names[row["metric"]]] = float(row["mean"])
    for shot in (1, 2, 4, 8):
        runs = [
            json.load(path.open(encoding="utf-8"))
            for path in (template_root / f"{shot}-shot").glob("seed=*/metrics.json")
        ]
        values[shot]["false_positive_rate"] = float(
            np.mean([run["test_false_positive_rate"] for run in runs])
        )
        values[shot]["false_negative_rate"] = float(
            np.mean([run["test_false_negative_rate"] for run in runs])
        )
    return values


def main() -> None:
    args = parse_args()
    old = old_results(args.archive)
    new = new_results(args.new_summary, args.new_template_root)
    rows = []
    for shot in (1, 2, 4, 8):
        row = {"shot": shot}
        for metric in (
            "image_auroc",
            "pixel_auroc",
            "false_positive_rate",
            "false_negative_rate",
        ):
            row[f"old_{metric}"] = old[shot][metric]
            row[f"new_{metric}"] = new[shot][metric]
            row[f"delta_{metric}"] = new[shot][metric] - old[shot][metric]
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
