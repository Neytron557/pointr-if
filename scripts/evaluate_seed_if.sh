#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

CKPT="${1:?checkpoint path required}"
OUT_DIR="${2:-outputs/seed_pointr_if/seed570/test_eval}"
SPLIT="${3:-test}"
CAND_MANIFEST="${4:-outputs/seed_pointr_if/candidate_bank/${SPLIT}_candidate_manifest.csv}"
TRIPLET="data/real_projected_shapenet55_adapointr_predictions/manifests/${SPLIT}_triplets.csv"
mkdir -p "$OUT_DIR"

python -m pointr_if.evaluate \
  --manifest "$TRIPLET" \
  --candidate-manifest "$CAND_MANIFEST" \
  --checkpoint "$CKPT" \
  --out-dir "$OUT_DIR" \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda:0 \
  --save-predictions \
  --resample-mode fps \
  --n-partial 2048 \
  --n-coarse 4096 \
  --n-gt 4096 \
  --n-output 4096 \
  --eval-seed 200570 \
  >"$OUT_DIR/stdout.log" \
  2>"$OUT_DIR/stderr.log"
