from __future__ import annotations

import os

import cv2
import numpy as np

from project_pipeline.baseline import TemplateBaseline, select_references
from project_pipeline.metrics import image_metrics, mean_std, pixel_metrics
from sklearn.metrics import roc_auc_score


def test_global_template_selection_and_metrics(tmp_path):
    rng = np.random.default_rng(0)
    folder = tmp_path
    paths = []
    for index in range(2):
        image = np.full((48, 48, 3), 100 + index, dtype=np.uint8)
        path = folder / f"ref_{index}.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        paths.append(str(path))

    test = np.full((48, 48, 3), 101, dtype=np.uint8)
    test[20:25, 20:25] = [250, 10, 10]
    test_path = folder / "test.png"
    cv2.imwrite(str(test_path), cv2.cvtColor(test, cv2.COLOR_RGB2BGR))

    model = TemplateBaseline(size=48, border_crop=0).fit(paths)
    score, anomaly_map, selected = model.predict(str(test_path))
    assert selected == 1
    assert score > 0
    assert anomaly_map.shape == (48, 48)


def test_metrics_and_sample_std():
    image_result = image_metrics([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert image_result["image_auroc"] == 1.0

    prediction = np.zeros((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint8)
    prediction[1:3, 1:3] = 1
    mask[1:3, 1:3] = 1
    assert pixel_metrics([prediction], [mask])["pixel_auroc"] == 1.0

    rng = np.random.default_rng(3)
    predictions = [rng.random((24, 24), dtype=np.float32) for _ in range(5)]
    masks = [rng.random((24, 24)) > 0.8 for _ in range(5)]
    exact = roc_auc_score(
        np.concatenate([mask.ravel() for mask in masks]),
        np.concatenate([prediction.ravel() for prediction in predictions]),
    )
    quantized = pixel_metrics(predictions, masks)["pixel_auroc"]
    assert abs(exact - quantized) < 0.001

    mean, std = mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert np.isclose(std, 1.0)


def test_reference_selection_excludes_calibration():
    names = [f"{index:03d}.png" for index in range(40)]
    calibration, references = select_references(names, shot=4, seed=2)
    assert calibration == names[:20]
    assert references == ["028.png", "029.png", "030.png", "031.png"]
    assert not set(calibration) & set(references)
