#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/real_projected_shapenet55_adapointr_predictions/manifests/train_triplets.csv}"
VAL_MANIFEST="${VAL_MANIFEST:-data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv}"
GROUP_DIR="${GROUP_DIR:-data/real_projected_shapenet55_groups}"
GEN_ROOT="${GEN_ROOT:-data/generated_train_views}"
VIEWS_PER_OBJECT="${VIEWS_PER_OBJECT:-4}"

mkdir -p "$GROUP_DIR" reports "$GEN_ROOT/logs"

python tools/build_multiview_groups.py \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --out-dir "$GROUP_DIR" \
  --audit-out reports/real_projected_shapenet55_group_audit.json \
  >"$GEN_ROOT/logs/build_groups.stdout.log" \
  2>"$GEN_ROOT/logs/build_groups.stderr.log"

python tools/generate_train_projected_views.py \
  --groups "$GROUP_DIR/train_groups.json" \
  --out-root "$GEN_ROOT" \
  --out-groups "$GROUP_DIR/train_groups_augmented.json" \
  --views-per-object "$VIEWS_PER_OBJECT" \
  --n-points 2048 \
  --image-size 64 \
  >"$GEN_ROOT/logs/generate_train_views.stdout.log" \
  2>"$GEN_ROOT/logs/generate_train_views.stderr.log"
