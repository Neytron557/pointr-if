#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

CONFIG="${1:-configs/real_projected_shapenet55_seed_if.yaml}"
OUT_DIR="${2:-outputs/seed_pointr_if/seed570}"
mkdir -p "$OUT_DIR"

python -m pointr_if.train_refiner \
  --config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  >"$OUT_DIR/stdout.log" \
  2>"$OUT_DIR/stderr.log"
