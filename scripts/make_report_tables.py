#!/usr/bin/env python3
"""Create compact LaTeX tables from aggregated experiment results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHOD_NAMES = {
    "template": "Template",
    "anomalydino": "AnomalyDINO",
    "multilayer": "Multi-layer",
}
SHOTS = (1, 2, 4, 8)
CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe_fryum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--category-summary", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def lookup(
    rows: list[dict[str, str]],
    method: str,
    shot: int,
    metric: str,
) -> tuple[float, float]:
    for row in rows:
        if (
            row["method"] == method
            and int(row["shot"]) == shot
            and row["metric"] == metric
        ):
            return float(row["mean"]), float(row["std"])
    raise KeyError((method, shot, metric))


def metric_cell(value: tuple[float, float]) -> str:
    mean, std = value
    return f"{100 * mean:.1f} $\\pm$ {100 * std:.1f}"


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = read_csv(args.summary)
    category_summary = read_csv(args.category_summary)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Unweighted category mean and standard deviation over seeds.",
        r"Image AUROC is shown in the upper block and 448-pixel localization",
        r"AUROC in the lower block.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & 1-shot & 2-shot & 4-shot & 8-shot \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Image-level AUROC (\%)}}\\",
    ]
    for method in METHOD_NAMES:
        cells = [
            metric_cell(
                lookup(
                    summary,
                    method,
                    shot,
                    "mean_classification_au_roc",
                )
            )
            for shot in SHOTS
        ]
        lines.append(
            METHOD_NAMES[method] + " & " + " & ".join(cells) + r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{5}{l}{\emph{448-pixel localization AUROC (\%)}}\\",
        ]
    )
    for method in METHOD_NAMES:
        cells = [
            metric_cell(
                lookup(
                    summary,
                    method,
                    shot,
                    "mean_segmentation_au_roc",
                )
            )
            for shot in SHOTS
        ]
        lines.append(
            METHOD_NAMES[method] + " & " + " & ".join(cells) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (args.output / "main_metrics.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Per-category image AUROC (\%) at 4-shot, mean over three seeds.}",
        r"\label{tab:category}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        "Method & " + " & ".join(categories.replace("_", r"\_") for categories in CATEGORIES) + r" \\",
        r"\midrule",
    ]
    for method in METHOD_NAMES:
        cells = []
        for category in CATEGORIES:
            matches = [
                row
                for row in category_summary
                if row["method"] == method
                and int(row["shot"]) == 4
                and row["category"] == category
                and row["metric"] == "mean_classification_au_roc"
            ]
            if not matches:
                cells.append("--")
            else:
                cells.append(f"{100 * float(matches[0]['mean']):.1f}")
        lines.append(
            METHOD_NAMES[method] + " & " + " & ".join(cells) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (args.output / "category_metrics.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    efficiency_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Encoder efficiency at $448\times448$ on an RTX A6000.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Variant & ms/image & peak GPU (MiB) \\",
        r"\midrule",
    ]
    if args.benchmark_dir is not None:
        for filename, label in (
            ("baseline.json", "AnomalyDINO"),
            ("multilayer.json", "Multi-layer"),
        ):
            path = args.benchmark_dir / filename
            if not path.exists():
                efficiency_lines.append(f"{label} & -- & -- \\\\")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            efficiency_lines.append(
                f"{label} & "
                f"{1000 * data['mean_forward_seconds']:.1f} & "
                f"{data['peak_reserved_gpu_mb']:.0f} \\\\"
            )
    efficiency_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (args.output / "efficiency.tex").write_text(
        "\n".join(efficiency_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote LaTeX tables to {args.output}")


if __name__ == "__main__":
    main()
