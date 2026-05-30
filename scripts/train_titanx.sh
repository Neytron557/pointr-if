#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG="${1:-configs/titanx_fast_subset.yaml}"
OUT="${2:-outputs/titanx_pointr_if}"
python -m pointr_if.train_refiner --config "$CONFIG" --output-dir "$OUT" --device cuda
