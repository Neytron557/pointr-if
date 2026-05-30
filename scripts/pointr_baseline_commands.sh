#!/usr/bin/env bash
set -euo pipefail
# Reference commands for the official yuxumin/PoinTr repository.
# Run these in a separate official PoinTr clone after installing its CUDA extensions.
# Official repo: https://github.com/yuxumin/PoinTr

POINTR_ROOT="${POINTR_ROOT:-$HOME/PoinTr}"
cd "$POINTR_ROOT"

# Evaluate pretrained PoinTr on PCN:
bash ./scripts/test.sh 0 \
  --ckpts ./pretrained/PoinTr_PCN.pth \
  --config ./cfgs/PCN_models/PoinTr.yaml \
  --exp_name pointr_pcn_baseline

# Evaluate pretrained AdaPoinTr on PCN as a stronger backbone/reference:
bash ./scripts/test.sh 0 \
  --ckpts ./pretrained/AdaPoinTr_PCN.pth \
  --config ./cfgs/PCN_models/AdaPoinTr.yaml \
  --exp_name adapointr_pcn_reference

# Inference on a folder of partial point clouds to produce coarse completions.
# Adjust --pc_root and --out_pc_root to match the subset used for PoinTr-IF.
python tools/inference.py \
  cfgs/PCN_models/AdaPoinTr.yaml pretrained/AdaPoinTr_PCN.pth \
  --pc_root data/PCN/val/partial \
  --out_pc_root ../pointr_if_project/data/coarse/adapointr_val
