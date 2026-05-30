#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

OUT_BASE="${1:-outputs/seed_pointr_if}"
OUT_DIR="${2:-$OUT_BASE/category_source_policy_seedbank}"
mkdir -p "$OUT_DIR"

python tools/evaluate_candidate_source_policy.py \
  --val-category-metrics "$OUT_BASE/val_candidate_oracle_seedbank/category_metrics.csv" \
  --test-per-sample-metrics "$OUT_BASE/test_candidate_oracle_seedbank/per_sample_metrics.csv" \
  --out-dir "$OUT_DIR" \
  --baseline-method adapointr_coarse \
  --min-val-samples 3 \
  >"$OUT_DIR/stdout.log" \
  2>"$OUT_DIR/stderr.log"
