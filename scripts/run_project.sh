#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/aiaa3201_project2}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
CODE_ROOT="${CODE_ROOT:-$PROJECT_ROOT/code/AnomalyDINO}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/VisA_pytorch/1cls}"
RUN_ROOT="${RUN_ROOT:-/tmp/aiaa3201_runs}"

mkdir -p "$RUN_ROOT"
cd "$RUN_ROOT"

export PYTHONHOME="${PYTHONHOME:-$PROJECT_ROOT/.venv}"
export PYTHONPATH="$CODE_ROOT"
export TORCH_HOME="${TORCH_HOME:-/tmp/aiaa_torch}"
export DINOV2_HUB_DIR="${DINOV2_HUB_DIR:-$TORCH_HOME/hub/facebookresearch_dinov2_main}"

"$PYTHON" "$PROJECT_ROOT/scripts/run_template_baseline.py" \
  --data-root "$DATA_ROOT" \
  --output "$RUN_ROOT/template_results" \
  --shots 1 2 4 8 \
  --seeds 0 1 2

for variant in baseline improvement; do
  extra=()
  if [[ "$variant" == "improvement" ]]; then
    extra=(--layer 9 11 --tag multilayer9-11)
  else
    extra=(--tag baseline)
  fi
  "$PYTHON" "$CODE_ROOT/run_anomalydino.py" \
    --dataset VisA \
    --model_name dinov2_vits14 \
    --data_root "$DATA_ROOT" \
    --shots 1 2 4 8 \
    --num_seeds 3 \
    --preprocess agnostic \
    --resolution 448 \
    --eval_segm \
    --faiss_on_cpu \
    "${extra[@]}"
done

for method_dir in dinov2_vits14_448 dinov2_vits14_448_layer9+11; do
  for run_dir in "$RUN_ROOT/results_VisA/$method_dir"/*; do
    [[ -d "$run_dir" ]] || continue
    [[ "$run_dir" == *smoke* ]] && continue
    for seed in 0 1 2; do
      "$PYTHON" "$PROJECT_ROOT/scripts/rescore_anomalydino_448.py" \
        --run-dir "$run_dir" \
        --seed "$seed" \
        --data-root "$DATA_ROOT"
    done
  done
done

"$PYTHON" "$PROJECT_ROOT/scripts/aggregate_results.py" \
  --official-root "$RUN_ROOT" \
  --template-root "$RUN_ROOT/template_results" \
  --output "$PROJECT_ROOT/results"
