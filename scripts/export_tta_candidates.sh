#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

OUT_ROOT="${1:-outputs/seed_pointr_if/tta_adapointr}"
mkdir -p "$OUT_ROOT/logs"
{
  echo "$0 $*"
  date -u
  pwd
} >"$OUT_ROOT/logs/command_log.txt"

for split in train val test; do
  python tools/export_pointr_tta_predictions.py \
    --config external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml \
    --checkpoint external/PoinTr/pretrained/AdaPoinTr_ps55.pth \
    --data-root external/PoinTr/data/ShapeNet55-34 \
    --split "$split" \
    --split-name "$split" \
    --out-root "$OUT_ROOT" \
    --batch-size 2 \
    --input-points 2048 \
    --device cuda:0 \
    --seed 570 \
    >"$OUT_ROOT/logs/${split}_stdout.log" \
    2>"$OUT_ROOT/logs/${split}_stderr.log"
done
