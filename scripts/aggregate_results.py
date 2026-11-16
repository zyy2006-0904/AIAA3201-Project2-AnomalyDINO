#!/usr/bin/env python3
"""Aggregate official and template-baseline metrics across seeds and shots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe_fryum",
)
SHOTS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
METRICS = (
    "mean_classification_au_roc",
    "mean_classification_ap",
    "mean_classification_f1",
    "mean_segmentation_au_roc",
    "mean_au_pro",
)
METHOD_LABELS = {
    "template": "Template difference",
    "anomalydino": "AnomalyDINO",
    "multilayer": "Multi-layer AnomalyDINO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def official_runs(root: Path) -> dict[str, dict[int, dict[int, Path]]]:
    runs: dict[str, dict[int, dict[int, Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for metrics_path in root.glob(
        "results_VisA/dinov2_vits14_448*/*/metrics_seed=*.json"
    ):
        run_dir = metrics_path.parent.name
        if "smoke" in run_dir:
            continue
        shot = int(run_dir.split("-shot", 1)[0])
        seed = int(metrics_path.stem.split("=", 1)[1])
        method = "multilayer" if "multilayer" in run_dir else "anomalydino"
        runs[method][shot][seed] = metrics_path
    for metrics_path in root.glob(
        "results_VisA/dinov2_vits14_448*/*/metrics_448_seed=*.json"
    ):
        run_dir = metrics_path.parent.name
        if "smoke" in run_dir:
            continue
        shot = int(run_dir.split("-shot", 1)[0])
        seed = int(metrics_path.stem.rsplit("=", 1)[1])
        method = "multilayer" if "multilayer" in run_dir else "anomalydino"
        runs[method][shot][seed] = metrics_path
    return runs


def template_runs(root: Path) -> dict[str, dict[int, dict[int, Path]]]:
    runs: dict[str, dict[int, dict[int, Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for metrics_path in root.glob("*-shot/seed=*/metrics.json"):
        shot = int(metrics_path.parent.parent.name.split("-shot", 1)[0])
        seed = int(metrics_path.parent.name.split("=", 1)[1])
        runs["template"][shot][seed] = metrics_path
    return runs


def load_metrics(path: Path, method: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method == "template":
        return {
            "mean_classification_au_roc": payload["mean_image_auroc"],
            "mean_classification_ap": payload["mean_image_ap"],
            "mean_classification_f1": payload["mean_image_f1"],
            "mean_segmentation_au_roc": payload["mean_pixel_auroc"],
            "mean_au_pro": None,
            "categories": payload["categories"],
            "category_metric_names": {
                "mean_classification_au_roc": "image_auroc",
                "mean_classification_ap": "image_ap",
                "mean_classification_f1": "image_f1",
                "mean_segmentation_au_roc": "pixel_auroc",
                "mean_au_pro": None,
            },
        }
    return {
        **{name: payload.get(name) for name in METRICS},
        "categories": payload,
        "category_metric_names": {
            "mean_classification_au_roc": "classification_AUROC",
            "mean_classification_ap": "classification_AP",
            "mean_classification_f1": "classification_F1",
            "mean_segmentation_au_roc": "seg_AUROC",
            "mean_au_pro": "seg_AUPRO",
        },
    }


def std(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {
        **template_runs(args.template_root),
        **official_runs(args.official_root),
    }

    summary_rows = []
    category_rows = []
    summary: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}

    for method, shots in runs.items():
        summary[method] = {}
        for shot in SHOTS:
            seed_paths = shots.get(shot, {})
            loaded = {
                seed: load_metrics(seed_paths[seed], method)
                for seed in SEEDS
                if seed in seed_paths
            }
            if not loaded:
                continue
            summary[method][str(shot)] = {}
            for metric in METRICS:
                values = [payload[metric] for payload in loaded.values()]
                clean = [value for value in values if value is not None]
                summary[method][str(shot)][metric] = {
                    "mean": float(np.mean(clean)) if clean else None,
                    "std": std(values),
                    "count": len(clean),
                }
                summary_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "shot": shot,
                        "metric": metric,
                        "mean": summary[method][str(shot)][metric]["mean"],
                        "std": summary[method][str(shot)][metric]["std"],
                        "seed_count": len(clean),
                    }
                )

            for category in CATEGORIES:
                for metric in METRICS:
                    source_name = loaded[0]["category_metric_names"][metric]
                    if source_name is None:
                        continue
                    values = [
                        payload["categories"][category][source_name]
                        for payload in loaded.values()
                    ]
                    category_rows.append(
                        {
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "shot": shot,
                            "category": category,
                            "metric": metric,
                            "mean": float(np.mean(values)),
                            "std": std(values),
                            "seed_count": len(values),
                        }
                    )

    with (args.output / "summary_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output / "summary_by_category.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(category_rows[0]))
        writer.writeheader()
        writer.writerows(category_rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    plt.figure(figsize=(7, 4.5))
    for method in ("template", "anomalydino", "multilayer"):
        if method not in summary:
            continue
        x = np.asarray(SHOTS)
        means = np.asarray(
            [
                summary[method][str(shot)]["mean_classification_au_roc"]["mean"]
                for shot in SHOTS
            ]
        ) * 100
        errors = np.asarray(
            [
                summary[method][str(shot)]["mean_classification_au_roc"]["std"]
                for shot in SHOTS
            ]
        ) * 100
        plt.errorbar(x, means, yerr=errors, marker="o", capsize=3, label=METHOD_LABELS[method])
    plt.xlabel("Normal reference images (shots)")
    plt.ylabel("Image AUROC (%)")
    plt.xticks(SHOTS)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output / "image_auroc_vs_shots.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for method in ("template", "anomalydino", "multilayer"):
        if method not in summary:
            continue
        x = np.asarray(SHOTS)
        means = np.asarray(
            [
                summary[method][str(shot)]["mean_segmentation_au_roc"]["mean"]
                for shot in SHOTS
            ]
        ) * 100
        errors = np.asarray(
            [
                summary[method][str(shot)]["mean_segmentation_au_roc"]["std"]
                for shot in SHOTS
            ]
        ) * 100
        plt.errorbar(x, means, yerr=errors, marker="o", capsize=3, label=METHOD_LABELS[method])
    plt.xlabel("Normal reference images (shots)")
    plt.ylabel("Patch-level pixel AUROC (%)")
    plt.xticks(SHOTS)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output / "pixel_auroc_vs_shots.png", dpi=200)
    plt.close()
    print(f"Wrote aggregate results to {args.output}")


if __name__ == "__main__":
    main()
