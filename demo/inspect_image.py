#!/usr/bin/env python3
"""Run the required single-image OK/NG and heatmap demonstration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch

from src.backbones import get_model
from src.post_eval import mean_top1p
from src.utils import augment_image, dists2map, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--category", default="candle")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "aiaa3201_project2/data/VisA_pytorch/1cls",
    )
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rgb(path: Path | str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def score_image(model, index, image, masking: bool) -> tuple[float, np.ndarray]:
    image_tensor, grid_size = model.prepare_image(image)
    features = model.extract_features(image_tensor)
    mask = model.compute_background_mask(
        features,
        grid_size,
        threshold=10,
        masking_type=masking,
    )
    selected = features[mask].astype(np.float32)
    faiss.normalize_L2(selected)
    distances, _ = index.search(selected, k=1)
    distances = distances[:, 0] / 2
    patch_map = np.zeros(mask.size, dtype=np.float32)
    patch_map[mask] = distances
    flat_map = patch_map.reshape(grid_size)
    return mean_top1p(patch_map), flat_map


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_folder = args.data_root / args.category / "train" / "good"
    training = list_images(str(train_folder))
    calibration_names = training[:20]
    references = training[20:]
    selected_references = references[
        args.seed * args.shot : (args.seed + 1) * args.shot
    ]

    model = get_model(
        "dinov2_vits14",
        "cuda",
        args.resolution,
        layers=args.layers,
    )
    memory = []
    with torch.inference_mode():
        for filename in selected_references:
            image = load_rgb(train_folder / filename)
            for augmented in augment_image(image):
                image_tensor, grid_size = model.prepare_image(augmented)
                features = model.extract_features(image_tensor)
                mask = model.compute_background_mask(
                    features,
                    grid_size,
                    threshold=10,
                    masking_type=False,
                )
                memory.append(features[mask].astype(np.float32))
    memory_array = np.concatenate(memory, axis=0)
    faiss.normalize_L2(memory_array)
    index = faiss.IndexFlatL2(memory_array.shape[1])
    index.add(memory_array)

    calibration_scores = []
    with torch.inference_mode():
        for filename in calibration_names:
            score, _ = score_image(
                model,
                index,
                load_rgb(train_folder / filename),
                masking=True,
            )
            calibration_scores.append(score)
    threshold = float(np.percentile(calibration_scores, 95))

    with torch.inference_mode():
        score, patch_map = score_image(
            model,
            index,
            load_rgb(args.image),
            masking=True,
        )
    decision = "NG" if score > threshold else "OK"
    image = load_rgb(args.image)
    anomaly_map = dists2map(patch_map, image.shape)
    normalized_map = anomaly_map - anomaly_map.min()
    normalized_map /= max(float(normalized_map.max()), 1e-8)
    heatmap = cv2.applyColorMap(
        np.uint8(normalized_map * 255),
        cv2.COLORMAP_TURBO,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)

    cv2.imwrite(str(args.output / "input.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(args.output / "heatmap.png"),
        cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(args.output / "overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    result = {
        "image": str(args.image),
        "category": args.category,
        "shot": args.shot,
        "seed": args.seed,
        "layers": args.layers,
        "anomaly_score": float(score),
        "calibration_threshold": float(threshold),
        "decision": decision,
        "reference_images": selected_references,
        "calibration_images": calibration_names,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
