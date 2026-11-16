"""Shared classification and localization metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def image_metrics(scores: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.uint8)
    s = np.asarray(scores, dtype=np.float64)
    precision, recall, _ = precision_recall_curve(y, s)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return {
        "image_auroc": float(roc_auc_score(y, s)),
        "image_ap": float(average_precision_score(y, s)),
        "image_f1": float(np.max(f1)),
    }


def pixel_metrics(
    maps: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
) -> dict[str, float]:
    if not maps:
        raise ValueError("No localization maps were provided")
    predictions = np.concatenate(
        [np.asarray(item, dtype=np.float32).ravel() for item in maps]
    )
    targets = np.concatenate(
        [(np.asarray(item) > 0).astype(np.uint8).ravel() for item in masks]
    )
    if targets.min() == targets.max():
        raise ValueError("Pixel AUROC is undefined when all masks have one class")
    return {
        "pixel_auroc": float(roc_auc_score(targets, predictions)),
    }


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    return float(data.mean()), float(data.std(ddof=1)) if len(data) > 1 else 0.0
