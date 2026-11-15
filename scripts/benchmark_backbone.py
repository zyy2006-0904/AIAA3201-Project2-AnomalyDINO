#!/usr/bin/env python3
"""Measure DINOv2 forward-pass latency and peak GPU memory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    from src.backbones import get_model

    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = get_model(
        "dinov2_vits14",
        "cuda",
        smaller_edge_size=448,
        layers=args.layers,
    )
    tensor, _ = model.prepare_image(str(args.image))
    for _ in range(10):
        model.extract_features(tensor)
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(args.iterations):
        model.extract_features(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "model": "dinov2_vits14",
        "resolution": 448,
        "layers": args.layers,
        "iterations": args.iterations,
        "mean_forward_seconds": elapsed / args.iterations,
        "images_per_second": args.iterations / elapsed,
        "peak_allocated_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_gpu_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "feature_extraction_included": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
