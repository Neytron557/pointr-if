#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

SPLIT="${1:-test}"
OUT_BASE="${2:-outputs/seed_pointr_if}"
TRIPLET="data/real_projected_shapenet55_adapointr_predictions/manifests/${SPLIT}_triplets.csv"
CAND_MANIFEST="$OUT_BASE/candidate_bank_seed/${SPLIT}_candidate_manifest.csv"
CAND_ROOT="$OUT_BASE/candidate_bank_seed/${SPLIT}"
ORACLE_DIR="$OUT_BASE/${SPLIT}_candidate_oracle_seedbank"
mkdir -p "$OUT_BASE/logs" "$OUT_BASE/candidate_bank_seed" "$ORACLE_DIR"

python tools/build_candidate_manifest.py \
  --triplet-manifest "$TRIPLET" \
  --out-manifest "$CAND_MANIFEST" \
  --out-root "$CAND_ROOT" \
  --candidate-points 4096 \
  --resample-mode random \
  --include-symmetry \
  >"$OUT_BASE/logs/${SPLIT}_candidate_manifest.stdout.log" \
  2>"$OUT_BASE/logs/${SPLIT}_candidate_manifest.stderr.log"

python tools/evaluate_candidate_bank_oracle.py \
  --candidate-manifest "$CAND_MANIFEST" \
  --out-dir "$ORACLE_DIR" \
  --resample-mode fps \
  --n-gt 4096 \
  --n-output 4096 \
  --point-oracle-points 0 \
  --seed 200570 \
  --device cuda:0 \
  >"$OUT_BASE/logs/${SPLIT}_candidate_oracle.stdout.log" \
  2>"$OUT_BASE/logs/${SPLIT}_candidate_oracle.stderr.log"
