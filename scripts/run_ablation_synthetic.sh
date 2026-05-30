#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p outputs/ablation_synthetic
python -m pointr_if.train_refiner --config configs/smoke_synthetic.yaml --output-dir outputs/ablation_synthetic/local_occ
python -m pointr_if.train_refiner --config configs/ablation_global_only.yaml --output-dir outputs/ablation_synthetic/global_only
python -m pointr_if.train_refiner --config configs/ablation_no_occ.yaml --output-dir outputs/ablation_synthetic/no_occ
python scripts/summarize_runs.py outputs/ablation_synthetic/*/summary.json \
  --out outputs/ablation_synthetic/summary_table.csv \
  --markdown-out outputs/ablation_synthetic/summary_table.md
