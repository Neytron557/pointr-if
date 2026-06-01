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
export NUM_THREADS="${NUM_THREADS:-8}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

OUT_DIR="${1:-outputs/reproduce_run_b_best}"
PCN_ARTIFACT_DIR="${PCN_ARTIFACT_DIR:-data/pcn_a100}"
PCN_ROOT="${PCN_ROOT:-data/ShapeNetCompletion}"
PCN_CONFIG="${PCN_CONFIG:-external/PoinTr/cfgs/PCN_models/AdaPoinTr.yaml}"
PCN_CHECKPOINT="${PCN_CHECKPOINT:-external/PoinTr/pretrained/AdaPoinTr_PCN.pth}"
BEST_CHECKPOINT="${BEST_CHECKPOINT:-checkpoints/run_b/best.pt}"
FEATURE_ROOT="${FEATURE_ROOT:-$OUT_DIR/run_b_features}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$PCN_ARTIFACT_DIR/pcn_artifact_summary.json" ]]; then
  PCN_ROOT="$PCN_ROOT" OUT_DIR="$PCN_ARTIFACT_DIR" bash scripts/build_pcn_artifacts.sh
fi

if [[ "$FORCE_EXPORT" == "1" || ! -f "$FEATURE_ROOT/manifests/test_triplets.csv" ]]; then
  python tools/export_adapointr_features.py \
    --source-manifest "$PCN_ARTIFACT_DIR/test_sources.csv" \
    --out-root "$FEATURE_ROOT" \
    --config "$PCN_CONFIG" \
    --checkpoint "$PCN_CHECKPOINT" \
    --split-name test \
    --input-points 2048 \
    --output-points 4096 \
    --batch-size "$EXPORT_BATCH_SIZE" \
    --device cuda:0 \
    --seed 570 \
    --feature-dtype float16
fi

python -m pointr_if.evaluate \
  --manifest "$FEATURE_ROOT/manifests/test_triplets.csv" \
  --checkpoint "$BEST_CHECKPOINT" \
  --out-dir "$OUT_DIR/test_eval" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device cuda \
  --eval-seed 200570 \
  --n-partial 2048 \
  --n-coarse 4096 \
  --n-gt 4096 \
  --n-output 4096

python tools/analyze_real_results_stats.py \
  "$OUT_DIR/test_eval/per_sample_metrics.csv" \
  --out-dir "$OUT_DIR/test_eval/stats" \
  --baseline coarse \
  --candidate refined \
  --bootstrap 5000 \
  --seed 570
