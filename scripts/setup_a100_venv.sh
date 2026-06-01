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
export MAX_JOBS="${MAX_JOBS:-8}"

if [[ -f /opt/rh/gcc-toolset-13/enable ]]; then
  # PyTorch 2.6+ extensions require GCC 9+. This server's default GCC is 8.5.
  source /opt/rh/gcc-toolset-13/enable
fi

if [[ ! -d external/PoinTr/models ]]; then
  git submodule update --init external/PoinTr
fi

if [[ ! -f external/PoinTr/pointnet2_ops/pointnet2_utils.py ]] \
  || ! grep -q "_chamfer_distances" external/PoinTr/extensions/chamfer_dist/__init__.py \
  || grep -q "import models.TopNet" external/PoinTr/models/__init__.py; then
  git -C external/PoinTr apply "$ROOT/patches/external_pointr_runtime.patch"
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

# Prefer CUDA 12.8 wheels on this server; fall back to compatible wheels if needed.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
  || python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
  || python -m pip install torch torchvision

python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install easydict einops scipy tensorboardX timm==0.4.5 ninja

gcc_major="$(gcc -dumpfullversion | cut -d. -f1)"
if [[ "$gcc_major" -lt 9 ]]; then
  echo "GCC 9+ is required to build PyTorch CUDA extensions; active compiler is: $(gcc --version | head -n 1)" >&2
  exit 1
fi
echo "Building PoinTr CUDA extensions with: $(gcc --version | head -n 1)"
(
  cd external/PoinTr/extensions/chamfer_dist
  rm -rf build chamfer.egg-info dist
  python setup.py install
)

python - <<'PY'
import json
import os
import torch

print(json.dumps({
    "python": os.sys.executable,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "visible_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the new .venv")
PY

PYTHONPATH="$ROOT/src" python scripts/doctor.py >/dev/null
