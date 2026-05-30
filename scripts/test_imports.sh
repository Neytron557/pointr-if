#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python - <<'PY'
from torch.utils.data import DataLoader
from pointr_if.datasets import SyntheticCompletionDataset
from pointr_if.models import ImplicitSurfaceRefiner
from pointr_if.point_ops import sample_occupancy_queries, chamfer_distance

d = SyntheticCompletionDataset(num_samples=2, n_partial=16, n_coarse=32, n_gt=32, seed=0)
b = next(iter(DataLoader(d, batch_size=2)))
m = ImplicitSurfaceRefiner(hidden_dim=16, point_feature_dim=12, global_dim=24, fourier_bands=2, k=4)
q, labels = sample_occupancy_queries(b['gt'].float(), n_query=10)
out = m(b['partial'].float(), b['coarse'].float(), q.float())
cd = chamfer_distance(out['refined'], b['gt'].float()).detach()
print('refined', tuple(out['refined'].shape), 'occ', tuple(out['occ_logits'].shape), 'cd', float(cd))
PY
