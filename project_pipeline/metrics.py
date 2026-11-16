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
    arrays = [np.asarray(item, dtype=np.float32) for item in maps]
    targets = [np.asarray(item) > 0 for item in masks]
    positive_count = sum(int(item.sum()) for item in targets)
    total_count = sum(item.size for item in targets)
    if positive_count == 0 or positive_count == total_count:
        raise ValueError("Pixel AUROC is undefined when all masks have one class")
    minimum = min(float(item.min()) for item in arrays)
    maximum = max(float(item.max()) for item in arrays)
    if not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
        return {"pixel_auroc": 0.5}

    bins = 65536
    positives = np.zeros(bins, dtype=np.int64)
    negatives = np.zeros(bins, dtype=np.int64)
    scale = (bins - 1) / (maximum - minimum)
    for prediction, target in zip(arrays, targets, strict=True):
        quantized = np.rint((prediction - minimum) * scale).astype(np.int32)
        positives += np.bincount(
            quantized[target].ravel(),
            minlength=bins,
        )
        negatives += np.bincount(
            quantized[~target].ravel(),
            minlength=bins,
        )

    negative_count = total_count - positive_count
    lower_negatives = np.concatenate(([0], np.cumsum(negatives[:-1])))
    rank_sum = np.sum(positives * (lower_negatives + 0.5 * negatives))
    auroc = rank_sum / (positive_count * negative_count)
    return {
        "pixel_auroc": float(auroc),
    }


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    return float(data.mean()), float(data.std(ddof=1)) if len(data) > 1 else 0.0
