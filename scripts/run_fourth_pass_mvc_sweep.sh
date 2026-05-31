#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

bash scripts/build_fourth_pass_groups.sh

EPOCHS="${EPOCHS:-8}" \
SEED="${SEED:-570}" \
TRAIN_GROUPS=data/real_projected_shapenet55_groups/train_groups.json \
VIEWS_PER_OBJECT=1 \
UNFREEZE=decoder_only \
UNFREEZE_LAST_ENCODER_BLOCKS=0 \
LR_DECODER="${LR_DECODER:-3e-5}" \
LR_ENCODER=0.0 \
LAMBDA_SELF=0.0 \
bash scripts/train_mvc_pointr_if.sh outputs/fourth_pass/adapointr_ft_cd

EPOCHS="${EPOCHS:-8}" \
SEED="${SEED:-571}" \
TRAIN_GROUPS=data/real_projected_shapenet55_groups/train_groups_augmented.json \
VIEWS_PER_OBJECT=2 \
UNFREEZE=decoder_plus_last_encoder \
UNFREEZE_LAST_ENCODER_BLOCKS=1 \
LR_DECODER="${LR_DECODER:-2e-5}" \
LR_ENCODER="${LR_ENCODER:-1e-6}" \
LAMBDA_SELF="${LAMBDA_SELF:-0.05}" \
bash scripts/train_mvc_pointr_if.sh outputs/fourth_pass/adapointr_mvc_lam005_last_encoder
