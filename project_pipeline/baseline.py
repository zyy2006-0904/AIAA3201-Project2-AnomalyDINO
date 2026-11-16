"""Integrated multi-channel template baseline.

The primary mode follows the course requirement literally: one globally closest
normal template is selected for each test image. A per-pixel multi-template mode
is retained as a controlled ablation inspired by the teammate's implementation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .metrics import image_metrics, pixel_metrics


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_CHANNELS = ("rgb", "gray", "gradient", "ssim")
CALIBRATION_COUNT = 20


def list_images(folder: str | os.PathLike[str]) -> list[str]:
    return sorted(
        name
        for name in os.listdir(folder)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )


def select_references(
    training_names: list[str],
    shot: int,
    seed: int,
    calibration_count: int = CALIBRATION_COUNT,
) -> tuple[list[str], list[str]]:
    ordered = sorted(training_names)
    calibration = ordered[:calibration_count]
    pool = ordered[calibration_count:]
    references = pool[seed * shot : (seed + 1) * shot]
    if len(references) != shot:
        raise ValueError(f"Cannot select {shot} references for seed {seed}")
    return calibration, references


def load_rgb(path: str | os.PathLike[str], size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def load_mask(path: str | os.PathLike[str] | None, size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


@dataclass
class PreparedImage:
    rgb: np.ndarray
    gray: np.ndarray
    gradient: np.ndarray
    ssim_mean: np.ndarray
    ssim_square_mean: np.ndarray


def prepare_image(image: np.ndarray) -> PreparedImage:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rgb = image.astype(np.float32) / 255.0
    ssim_mean = cv2.GaussianBlur(
        rgb,
        (0, 0),
        sigmaX=1.5,
        sigmaY=1.5,
        borderType=cv2.BORDER_REFLECT,
    ).mean(axis=2)
    ssim_square_mean = cv2.GaussianBlur(
        rgb * rgb,
        (0, 0),
        sigmaX=1.5,
        sigmaY=1.5,
        borderType=cv2.BORDER_REFLECT,
    ).mean(axis=2)
    return PreparedImage(
        rgb=rgb,
        gray=gray / 255.0,
        gradient=cv2.magnitude(gx, gy),
        ssim_mean=ssim_mean,
        ssim_square_mean=ssim_square_mean,
    )


@dataclass
class TemplateBaseline:
    size: int = 448
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    template_mode: Literal["global", "per_pixel"] = "global"
    score_mode: Literal["percentile", "top1"] = "percentile"
    score_percentile: float = 99.5
    border_crop: int = 0
    smooth_sigma: float = 4.0
    morph_kernel: int = 3
    use_ecc: bool = False
    ecc_max_iter: int = 50
    references: list[PreparedImage] = field(default_factory=list)
    reference_paths: list[str] = field(default_factory=list)
    threshold: float | None = None
    calibration_scores: list[float] = field(default_factory=list)

    def fit(self, paths: list[str]) -> "TemplateBaseline":
        self.reference_paths = list(paths)
        self.references = [prepare_image(load_rgb(path, self.size)) for path in paths]
        if not self.references:
            raise ValueError("At least one normal reference image is required")
        return self

    def calibrate(self, paths: list[str]) -> float:
        self.calibration_scores = [self.predict(path)[0] for path in paths]
        self.threshold = float(np.percentile(self.calibration_scores, 95))
        return self.threshold

    def crop(self, array: np.ndarray) -> np.ndarray:
        margin = self.border_crop
        if margin <= 0:
            return array
        return array[margin:-margin, margin:-margin]

    def _difference(self, image: PreparedImage, reference: PreparedImage) -> np.ndarray:
        if self.use_ecc:
            rgb = _align_rgb(reference.rgb, image.rgb, self.ecc_max_iter)
            reference = prepare_image((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

        maps: list[np.ndarray] = []
        if "rgb" in self.channels:
            maps.append(np.abs(image.rgb - reference.rgb).mean(axis=2))
        if "gray" in self.channels:
            maps.append(np.abs(image.gray - reference.gray))
        if "gradient" in self.channels:
            scale = max(
                float(image.gradient.max()),
                float(reference.gradient.max()),
                1.0,
            )
            maps.append(np.abs(image.gradient - reference.gradient) / scale)
        if "ssim" in self.channels:
            maps.append(_ssim_difference(image, reference))
        if not maps:
            raise ValueError("At least one difference channel is required")
        return np.max(np.stack(maps), axis=0).astype(np.float32)

    def predict(self, path: str) -> tuple[float, np.ndarray, int | None]:
        image = prepare_image(load_rgb(path, self.size))
        maps = np.stack([self._difference(image, ref) for ref in self.references])
        if self.template_mode == "global":
            selected = int(np.argmin(maps.mean(axis=(1, 2))))
            anomaly_map = maps[selected]
        else:
            selected = None
            anomaly_map = maps.min(axis=0)

        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.smooth_sigma)
        if self.morph_kernel > 1:
            kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
            anomaly_map = cv2.morphologyEx(anomaly_map, cv2.MORPH_CLOSE, kernel)
            anomaly_map = cv2.morphologyEx(anomaly_map, cv2.MORPH_OPEN, kernel)

        values = self.crop(anomaly_map).reshape(-1)
        if self.score_mode == "percentile":
            score = float(np.percentile(values, self.score_percentile))
        else:
            count = max(1, int(len(values) * 0.01))
            score = float(np.partition(values, len(values) - count)[-count:].mean())
        return score, anomaly_map, selected


def _ssim_difference(image: PreparedImage, reference: PreparedImage) -> np.ndarray:
    """Fast Gaussian SSIM map with per-channel averaging.

    The implementation uses OpenCV's C++ Gaussian filter rather than the
    Python-level multichannel SSIM loop. It follows the standard SSIM constants
    for data in [0, 1] and keeps the full spatial map used by the baseline.
    """
    c1 = 0.01**2
    c2 = 0.03**2
    cross_mean = cv2.GaussianBlur(
        image.rgb * reference.rgb,
        (0, 0),
        sigmaX=1.5,
        sigmaY=1.5,
        borderType=cv2.BORDER_REFLECT,
    ).mean(axis=2)
    mu_image = image.ssim_mean
    mu_reference = reference.ssim_mean
    image_variance = np.maximum(
        image.ssim_square_mean - mu_image * mu_image,
        0.0,
    )
    reference_variance = np.maximum(
        reference.ssim_square_mean - mu_reference * mu_reference,
        0.0,
    )
    covariance = cross_mean - mu_image * mu_reference
    numerator = (2 * mu_image * mu_reference + c1) * (2 * covariance + c2)
    denominator = (
        (mu_image * mu_image + mu_reference * mu_reference + c1)
        * (image_variance + reference_variance + c2)
    )
    similarity = np.clip(numerator / denominator, -1.0, 1.0)
    return (1.0 - similarity).astype(np.float32)


def _align_rgb(reference: np.ndarray, image: np.ndarray, max_iter: int) -> np.ndarray:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iter,
        1e-5,
    )
    try:
        _, warp = cv2.findTransformECC(
            image_gray,
            ref_gray,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error:
        return reference
    return cv2.warpAffine(
        reference,
        warp,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def evaluate_class(
    data_root: str,
    category: str,
    references: list[str],
    calibration: list[str],
    *,
    size: int,
    channels: tuple[str, ...],
    template_mode: Literal["global", "per_pixel"],
    score_mode: Literal["percentile", "top1"],
    score_percentile: float,
    border_crop: int,
    use_ecc: bool,
    max_images: int | None,
) -> dict:
    train_dir = os.path.join(data_root, category, "train", "good")
    model = TemplateBaseline(
        size=size,
        channels=channels,
        template_mode=template_mode,
        score_mode=score_mode,
        score_percentile=score_percentile,
        border_crop=border_crop,
        use_ecc=use_ecc,
    ).fit([os.path.join(train_dir, name) for name in references])
    model.calibrate([os.path.join(train_dir, name) for name in calibration])

    scores: list[float] = []
    labels: list[int] = []
    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    times: list[float] = []
    records: list[dict] = []

    for split, label in (("good", 0), ("bad", 1)):
        folder = os.path.join(data_root, category, "test", split)
        names = list_images(folder)
        if max_images is not None:
            names = names[:max_images]
        for name in names:
            started = time.perf_counter()
            score, anomaly_map, selected = model.predict(os.path.join(folder, name))
            times.append(time.perf_counter() - started)
            mask_path = (
                os.path.join(
                    data_root,
                    category,
                    "ground_truth",
                    "bad",
                    f"{os.path.splitext(name)[0]}.png",
                )
                if label
                else None
            )
            scores.append(score)
            labels.append(label)
            maps.append(model.crop(anomaly_map))
            masks.append(model.crop(load_mask(mask_path, size)))
            records.append(
                {
                    "file": f"{split}/{name}",
                    "score": score,
                    "decision": "NG" if score > model.threshold else "OK",
                    "closest_reference": (
                        references[selected] if selected is not None else None
                    ),
                }
            )

    return {
        "references": references,
        "calibration": calibration,
        "threshold": model.threshold,
        **image_metrics(scores, labels),
        **pixel_metrics(maps, masks),
        "mean_inference_seconds": float(np.mean(times)),
        "num_test_images": len(scores),
        "records": records,
    }
