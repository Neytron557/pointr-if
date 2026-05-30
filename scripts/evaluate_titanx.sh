#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CKPT="${1:-outputs/titanx_pointr_if/best_model.pt}"
OUT="${2:-outputs/titanx_eval}"
python -m pointr_if.evaluate_refiner --ckpt "$CKPT" --output-dir "$OUT" --device cuda
python -m pointr_if.visualize --ckpt "$CKPT" --output-dir "$OUT/visuals" --n 16 --device cuda
