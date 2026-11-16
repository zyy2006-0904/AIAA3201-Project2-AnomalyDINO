#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/aiaa3201_project2}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/VisA_pytorch/1cls}"
OUT_ROOT="${OUT_ROOT:-/tmp/aiaa3201_part1_ablation}"

mkdir -p "$OUT_ROOT"

"$PYTHON" "$PROJECT_ROOT/scripts/run_template_baseline.py" \
  --data-root "$DATA_ROOT" \
  --output "$OUT_ROOT/global_full" \
  --shots 1 --seeds 0 1 2

"$PYTHON" "$PROJECT_ROOT/scripts/run_template_baseline.py" \
  --data-root "$DATA_ROOT" \
  --output "$OUT_ROOT/perpixel_crop32" \
  --shots 1 --seeds 0 1 2 \
  --template-mode per_pixel --border-crop 32

"$PYTHON" "$PROJECT_ROOT/scripts/run_template_baseline.py" \
  --data-root "$DATA_ROOT" \
  --output "$OUT_ROOT/rgb_global_full" \
  --shots 1 --seeds 0 1 2 \
  --channels rgb
