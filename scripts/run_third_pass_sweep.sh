#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$PATH"

OUT_BASE="${1:-outputs/seed_pointr_if}"
bash scripts/audit_current_results.sh "reports/third_pass_protocol_audit"
for split in train val test; do
  bash scripts/evaluate_candidate_oracle.sh "$split" "$OUT_BASE"
done
bash scripts/evaluate_candidate_source_policy.sh "$OUT_BASE" "$OUT_BASE/category_source_policy_seedbank"

OUT_BASE="$OUT_BASE" python - <<'PY'
import json
import os
from pathlib import Path
out_base = Path(os.environ["OUT_BASE"])
summary = json.loads((out_base / "test_candidate_oracle_seedbank" / "oracle_summary.json").read_text())
gain = summary["overall"]["improvements_vs_adapointr_percent"]["sample_oracle"]
print(f"candidate oracle gain: {gain:.4f}%")
raise SystemExit(0 if gain >= 1.0 else 2)
PY

bash scripts/train_seed_if.sh configs/real_projected_shapenet55_seed_if.yaml "$OUT_BASE/seed570_passthrough"
bash scripts/evaluate_seed_if.sh \
  "$OUT_BASE/seed570_passthrough/best_model.pt" \
  "$OUT_BASE/seed570_passthrough/test_eval" \
  test \
  "$OUT_BASE/candidate_bank_seed/test_candidate_manifest.csv"
