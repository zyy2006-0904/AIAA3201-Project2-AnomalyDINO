#!/usr/bin/env python3
"""Create a short recorded demonstration from one demo inference result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def resize_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            int(round(image.shape[1] * scale)),
            int(round(image.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def main() -> None:
    args = parse_args()
    result = json.loads(
        (args.demo_dir / "result.json").read_text(encoding="utf-8")
    )
    image = cv2.imread(str(args.demo_dir / "input.png"), cv2.IMREAD_COLOR)
    overlay = cv2.imread(str(args.demo_dir / "overlay.png"), cv2.IMREAD_COLOR)
    if image is None or overlay is None:
        raise FileNotFoundError("input.png and overlay.png are required")

    width, height = 1600, 720
    left = resize_panel(image, width // 2, height - 170)
    right = resize_panel(overlay, width // 2, height - 170)
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    canvas[: height - 170, : width // 2] = left
    canvas[: height - 170, width // 2 :] = right
    for x, text in (
        (0, "Input product image"),
        (width // 2, "Anomaly heatmap overlay"),
    ):
        cv2.putText(
            canvas,
            text,
            (x + 35, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )

    fps = args.fps
    frame_count = max(1, int(args.seconds * fps))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not initialize the MP4 writer")

    score = result["anomaly_score"]
    threshold = result["calibration_threshold"]
    for index in range(frame_count):
        frame = canvas.copy()
        progress = min(1.0, (index + 1) / max(1, int(fps * 2.2)))
        display_score = score * progress
        cv2.putText(
            frame,
            (
                f"Category: {result['category']}   "
                f"{result['shot']}-shot   seed {result['seed']}"
            ),
            (35, height - 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (55, 55, 55),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Anomaly score: {display_score:.4f} / threshold {threshold:.4f}",
            (35, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        if index >= int(fps * 2.2):
            is_ng = result["decision"] == "NG"
            color = (40, 40, 220) if is_ng else (30, 150, 30)
            cv2.putText(
                frame,
                result["decision"],
                (width - 270, height - 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.65,
                color,
                4,
                cv2.LINE_AA,
            )
        writer.write(frame)
    writer.release()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
