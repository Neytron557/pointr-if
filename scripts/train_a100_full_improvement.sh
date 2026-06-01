#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

OUT_DIR="${1:-outputs/a100_full_improvement}"
PCN_ARTIFACT_DIR="${PCN_ARTIFACT_DIR:-data/pcn_a100}"
PCN_ROOT="${PCN_ROOT:-data/ShapeNetCompletion}"
PCN_CONFIG="${PCN_CONFIG:-external/PoinTr/cfgs/PCN_models/AdaPoinTr.yaml}"
PCN_CHECKPOINT="${PCN_CHECKPOINT:-external/PoinTr/pretrained/AdaPoinTr_PCN.pth}"

NUM_THREADS="${NUM_THREADS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VIEWS_PER_OBJECT="${VIEWS_PER_OBJECT:-4}"
A100_BATCH_SIZE="${A100_BATCH_SIZE:-16}"
BATCH_SIZE="${BATCH_SIZE:-$(( A100_BATCH_SIZE / VIEWS_PER_OBJECT ))}"
if [[ "$BATCH_SIZE" -lt 1 ]]; then
  BATCH_SIZE=1
fi
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

RUN_A="${RUN_A:-1}"
RUN_B="${RUN_B:-1}"
RUN_A_HOURS="${RUN_A_HOURS:-4.5}"
RUN_B_HOURS="${RUN_B_HOURS:-5.5}"
RUN_B_EPOCHS="${RUN_B_EPOCHS:-60}"
RUN_B_BATCH_SIZE="${RUN_B_BATCH_SIZE:-16}"
RUN_B_EXPORT_BATCH_SIZE="${RUN_B_EXPORT_BATCH_SIZE:-8}"
RUN_B_TRAIN_LIMIT="${RUN_B_TRAIN_LIMIT:-}"
RUN_B_VAL_LIMIT="${RUN_B_VAL_LIMIT:-}"
RUN_B_TEST_LIMIT="${RUN_B_TEST_LIMIT:-}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"

mkdir -p "$OUT_DIR"

python - <<'PY'
import json
import os
import torch

torch.set_num_threads(int(os.environ.get("NUM_THREADS", "8")))
info = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda0_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "torch_num_threads": torch.get_num_threads(),
}
print(json.dumps(info, indent=2))
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one visible CUDA device. Use CUDA_VISIBLE_DEVICES=1.")
PY

if [[ ! -f "$PCN_ARTIFACT_DIR/pcn_artifact_summary.json" ]]; then
  PCN_ROOT="$PCN_ROOT" OUT_DIR="$PCN_ARTIFACT_DIR" bash scripts/build_pcn_artifacts.sh
fi

FEATURE_ROOT="$OUT_DIR/run_b_features"

export_split() {
  local split="$1"
  local source_manifest="$PCN_ARTIFACT_DIR/${split}_sources.csv"
  local resolved_manifest="$FEATURE_ROOT/manifests/${split}_triplets.csv"
  local limit_var="$2"
  local limit="${!limit_var:-}"
  if [[ "$FORCE_EXPORT" != "1" && -f "$resolved_manifest" ]]; then
    return
  fi
  local args=(
    tools/export_adapointr_features.py
    --source-manifest "$source_manifest"
    --out-root "$FEATURE_ROOT"
    --config "$PCN_CONFIG"
    --checkpoint "$PCN_CHECKPOINT"
    --split-name "$split"
    --input-points 2048
    --output-points 4096
    --batch-size "$RUN_B_EXPORT_BATCH_SIZE"
    --device cuda:0
    --seed 570
    --feature-dtype float16
  )
  if [[ -n "$limit" ]]; then
    args+=(--limit "$limit")
  fi
  python "${args[@]}"
}

# Run A needs frozen coarse val/test manifests; Run B also needs train features.
export_split val RUN_B_VAL_LIMIT
export_split test RUN_B_TEST_LIMIT

if [[ "$RUN_A" == "1" ]]; then
  CONFIG="$PCN_CONFIG" \
  CHECKPOINT="$PCN_CHECKPOINT" \
  TRAIN_GROUPS="$PCN_ARTIFACT_DIR/train_groups.json" \
  VAL_MANIFEST="$FEATURE_ROOT/manifests/val_triplets.csv" \
  EPOCHS="${EPOCHS:-12}" \
  VIEWS_PER_OBJECT="$VIEWS_PER_OBJECT" \
  BATCH_SIZE="$BATCH_SIZE" \
  GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
  NUM_WORKERS="$NUM_WORKERS" \
  NUM_THREADS="$NUM_THREADS" \
  UNFREEZE="${UNFREEZE:-decoder_plus_last_encoder}" \
  UNFREEZE_LAST_ENCODER_BLOCKS="${UNFREEZE_LAST_ENCODER_BLOCKS:-2}" \
  LR_DECODER="${LR_DECODER:-2e-5}" \
  LR_ENCODER="${LR_ENCODER:-1e-6}" \
  LAMBDA_TARGET="${LAMBDA_TARGET:-1.0}" \
  LAMBDA_SELF="${LAMBDA_SELF:-0.08}" \
  N_PARTIAL=2048 \
  N_OUTPUT=4096 \
  N_GT=4096 \
  SELF_POINTS=2048 \
  TARGET_SAMPLE_MODE="${TARGET_SAMPLE_MODE:-random}" \
  SCHEDULER="${SCHEDULER:-cosine}" \
  DEVICE=cuda:0 \
  TIME_BUDGET_HOURS="$RUN_A_HOURS" \
  bash scripts/train_mvc_pointr_if.sh "$OUT_DIR/run_a_pcn_mvc"

  CONFIG="$PCN_CONFIG" \
  MANIFEST="$FEATURE_ROOT/manifests/test_triplets.csv" \
  DEVICE=cuda:0 \
  BATCH_SIZE=1 \
  N_PARTIAL=2048 \
  N_OUTPUT=4096 \
  N_GT=4096 \
  bash scripts/evaluate_mvc_pointr_if.sh \
    "$OUT_DIR/run_a_pcn_mvc/ckpt-best.pth" \
    "$OUT_DIR/run_a_pcn_mvc/test_eval" \
    test
fi

if [[ "$RUN_B" == "1" ]]; then
  export_split train RUN_B_TRAIN_LIMIT
  python -m pointr_if.train \
    --config configs/pcn_feature_gmv_if_a100.yaml \
    --train-manifest "$FEATURE_ROOT/manifests/train_triplets.csv" \
    --val-manifest "$FEATURE_ROOT/manifests/val_triplets.csv" \
    --out-dir "$OUT_DIR/run_b_feature_refiner" \
    --epochs "$RUN_B_EPOCHS" \
    --batch-size "$RUN_B_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --time-budget-hours "$RUN_B_HOURS" \
    --seed 570 \
    --device cuda \
    --amp

  python -m pointr_if.evaluate \
    --manifest "$FEATURE_ROOT/manifests/test_triplets.csv" \
    --checkpoint "$OUT_DIR/run_b_feature_refiner/best_model.pt" \
    --out-dir "$OUT_DIR/run_b_feature_refiner/test_eval" \
    --batch-size "$RUN_B_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device cuda \
    --eval-seed 200570 \
    --n-partial 2048 \
    --n-coarse 4096 \
    --n-gt 4096 \
    --n-output 4096 \
    --save-predictions

  python tools/analyze_real_results_stats.py \
    "$OUT_DIR/run_b_feature_refiner/test_eval/per_sample_metrics.csv" \
    --out-dir "$OUT_DIR/run_b_feature_refiner/test_eval/stats" \
    --baseline coarse \
    --candidate refined \
    --bootstrap 5000 \
    --seed 570
fi
