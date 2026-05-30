#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p outputs/smoke
python -m pointr_if.train_refiner --config configs/smoke_synthetic.yaml --output-dir outputs/smoke/local_occ
python -m pointr_if.evaluate_refiner --ckpt outputs/smoke/local_occ/best_model.pt --output-dir outputs/smoke/local_occ/eval --device cpu
python -m pointr_if.visualize --ckpt outputs/smoke/local_occ/best_model.pt --output-dir outputs/smoke/local_occ/visuals --n 4 --device cpu
