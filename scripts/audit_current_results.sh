#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

OUT_DIR="${1:-reports/third_pass_protocol_audit}"
mkdir -p "$OUT_DIR"
{
  echo "$0 $*"
  date -u
  pwd
  python --version
  nvidia-smi || true
} >"$OUT_DIR/command_log.txt"

for split in train val test; do
  python tools/validate_triplet_manifest.py \
    "data/real_projected_shapenet55_adapointr_predictions/manifests/${split}_triplets.csv" \
    --max-samples 999999 \
    --json-out "$OUT_DIR/${split}_manifest_full_recheck.json" \
    >"$OUT_DIR/${split}_manifest_full_recheck.stdout.log" \
    2>"$OUT_DIR/${split}_manifest_full_recheck.stderr.log"

  python tools/audit_point_counts.py \
    "data/real_projected_shapenet55_adapointr_predictions/manifests/${split}_triplets.csv" \
    --max-samples 999999 \
    --out "$OUT_DIR/${split}_point_counts.json" \
    >"$OUT_DIR/${split}_point_counts.stdout.log" \
    2>"$OUT_DIR/${split}_point_counts.stderr.log"
done
