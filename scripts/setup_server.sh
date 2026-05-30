#!/usr/bin/env bash
set -euo pipefail
# Server setup for the PoinTr-IF refiner. This does not install NVIDIA drivers.
# First confirm the GPU driver works: nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
# Pick a torch wheel matching the installed CUDA driver if necessary. The default below works for many CUDA 11.8 systems.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 || pip install torch torchvision
pip install -r requirements.txt
pip install -e .
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
