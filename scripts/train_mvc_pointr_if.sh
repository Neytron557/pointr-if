#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

OUT_DIR="${1:-outputs/fourth_pass/mvc_decoder_seed570}"
CONFIG="${CONFIG:-external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml}"
CHECKPOINT="${CHECKPOINT:-external/PoinTr/pretrained/AdaPoinTr_ps55.pth}"
TRAIN_GROUPS="${TRAIN_GROUPS:-data/real_projected_shapenet55_groups/train_groups_augmented.json}"
VAL_MANIFEST="${VAL_MANIFEST:-data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv}"
EPOCHS="${EPOCHS:-8}"
SEED="${SEED:-570}"
VIEWS_PER_OBJECT="${VIEWS_PER_OBJECT:-2}"
UNFREEZE="${UNFREEZE:-decoder_only}"
UNFREEZE_LAST_ENCODER_BLOCKS="${UNFREEZE_LAST_ENCODER_BLOCKS:-0}"
LR_DECODER="${LR_DECODER:-3e-5}"
LR_ENCODER="${LR_ENCODER:-0.0}"
LAMBDA_SELF="${LAMBDA_SELF:-0.1}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"

mkdir -p "$OUT_DIR"

ARGS=(
  tools/train_adapointr_multiview_consistency.py
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --train-groups "$TRAIN_GROUPS"
  --val-manifest "$VAL_MANIFEST"
  --out-dir "$OUT_DIR"
  --epochs "$EPOCHS"
  --views-per-object "$VIEWS_PER_OBJECT"
  --batch-size 1
  --grad-accum-steps 8
  --num-workers 0
  --lr-decoder "$LR_DECODER"
  --lr-encoder "$LR_ENCODER"
  --unfreeze "$UNFREEZE"
  --unfreeze-last-encoder-blocks "$UNFREEZE_LAST_ENCODER_BLOCKS"
  --lambda-target 1.0
  --lambda-self "$LAMBDA_SELF"
  --lambda-missing 0.0
  --n-partial 2048
  --n-output 4096
  --n-gt 4096
  --self-points 2048
  --eval-seed 200570
  --input-seed 570
  --seed "$SEED"
  --amp
  --device cuda:0
)

if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
  ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ -n "$MAX_VAL_SAMPLES" ]]; then
  ARGS+=(--max-val-samples "$MAX_VAL_SAMPLES")
fi

python "${ARGS[@]}" >"$OUT_DIR/stdout.log" 2>"$OUT_DIR/stderr.log"
