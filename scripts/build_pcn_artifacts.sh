#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

PCN_ROOT="${PCN_ROOT:-data/ShapeNetCompletion}"
CATEGORY_JSON="${CATEGORY_JSON:-external/PoinTr/data/PCN/PCN.json}"
OUT_DIR="${OUT_DIR:-data/pcn_a100}"

python tools/build_pcn_artifacts.py \
  --pcn-root "$PCN_ROOT" \
  --category-json "$CATEGORY_JSON" \
  --out-dir "$OUT_DIR"
