#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

CKPT="${1:?checkpoint path required}"
OUT_DIR="${2:-outputs/fourth_pass/mvc_test_eval}"
SPLIT="${3:-test}"
CONFIG="${CONFIG:-external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml}"
MANIFEST="${MANIFEST:-data/real_projected_shapenet55_adapointr_predictions/manifests/${SPLIT}_triplets.csv}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BOOTSTRAP="${BOOTSTRAP:-5000}"

mkdir -p "$OUT_DIR"

ARGS=(
  tools/export_finetuned_adapointr_predictions.py
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" \
  --device cuda:0 \
  --batch-size 1 \
  --n-partial 2048 \
  --n-output 4096 \
  --n-gt 4096 \
  --eval-seed 200570 \
  --input-seed 570 \
  --save-predictions
)

if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

python "${ARGS[@]}" >"$OUT_DIR/stdout.log" 2>"$OUT_DIR/stderr.log"

python tools/analyze_real_results_stats.py \
  "$OUT_DIR/per_sample_metrics.csv" \
  --out-dir "$OUT_DIR/stats" \
  --baseline coarse \
  --candidate refined \
  --bootstrap "$BOOTSTRAP" \
  --seed 570 \
  >"$OUT_DIR/stats_stdout.log" \
  2>"$OUT_DIR/stats_stderr.log"
