#!/usr/bin/env python3
"""Combine exported failure panels into one report figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(
        (args.input / "failure_cases.json").read_text(encoding="utf-8")
    )
    panels = [Image.open(record["file"]).convert("RGB") for record in records]
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    gap = 18
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * width + (columns + 1) * gap, rows * height + (rows + 1) * gap),
        "white",
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        x = gap + column * (width + gap)
        y = gap + row * (height + gap)
        canvas.paste(panel, (x, y))
    canvas.save(args.output, quality=95)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
