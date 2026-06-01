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
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_THREADS="${NUM_THREADS:-4}"
UNFREEZE="${UNFREEZE:-decoder_only}"
UNFREEZE_LAST_ENCODER_BLOCKS="${UNFREEZE_LAST_ENCODER_BLOCKS:-0}"
LR_DECODER="${LR_DECODER:-3e-5}"
LR_ENCODER="${LR_ENCODER:-0.0}"
LAMBDA_SELF="${LAMBDA_SELF:-0.1}"
LAMBDA_TARGET="${LAMBDA_TARGET:-1.0}"
LAMBDA_MISSING="${LAMBDA_MISSING:-0.0}"
N_PARTIAL="${N_PARTIAL:-2048}"
N_OUTPUT="${N_OUTPUT:-4096}"
N_GT="${N_GT:-4096}"
SELF_POINTS="${SELF_POINTS:-2048}"
TARGET_SAMPLE_MODE="${TARGET_SAMPLE_MODE:-fps}"
SCHEDULER="${SCHEDULER:-none}"
DEVICE="${DEVICE:-cuda:0}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-}"
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
  --batch-size "$BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --num-workers "$NUM_WORKERS"
  --lr-decoder "$LR_DECODER"
  --lr-encoder "$LR_ENCODER"
  --unfreeze "$UNFREEZE"
  --unfreeze-last-encoder-blocks "$UNFREEZE_LAST_ENCODER_BLOCKS"
  --lambda-target "$LAMBDA_TARGET"
  --lambda-self "$LAMBDA_SELF"
  --lambda-missing "$LAMBDA_MISSING"
  --n-partial "$N_PARTIAL"
  --n-output "$N_OUTPUT"
  --n-gt "$N_GT"
  --self-points "$SELF_POINTS"
  --eval-seed 200570
  --input-seed 570
  --seed "$SEED"
  --num-threads "$NUM_THREADS"
  --target-sample-mode "$TARGET_SAMPLE_MODE"
  --scheduler "$SCHEDULER"
  --amp
  --device "$DEVICE"
)

if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
  ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ -n "$MAX_VAL_SAMPLES" ]]; then
  ARGS+=(--max-val-samples "$MAX_VAL_SAMPLES")
fi
if [[ -n "$TIME_BUDGET_HOURS" ]]; then
  ARGS+=(--time-budget-hours "$TIME_BUDGET_HOURS")
fi

python "${ARGS[@]}" >"$OUT_DIR/stdout.log" 2>"$OUT_DIR/stderr.log"
