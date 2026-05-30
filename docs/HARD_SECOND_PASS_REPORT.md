# Hard Second Pass Report

Date: 2026-05-30

## Summary

The original real run was useful but too weak: the naive PoinTr-IF refiner improved held-out test Chamfer by only 0.0957%, with a bootstrap 95% CI crossing zero and non-significant paired tests. This second pass adds deterministic paired statistics, fixes archive packaging, and implements a gated multi-view self-structure refiner (`GMV-PoinTr-IF`) that produces a stronger and statistically significant improvement over the same AdaPoinTr coarse baseline.

The best held-out test result is the seed 570 GMV run:

| method | n | Chamfer | F-score | CD improvement vs AdaPoinTr coarse | bootstrap 95% CI | pos/neg | paired t p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Partial input | 150 | 0.158159 | 0.175398 | -127.94% | n/a | n/a | n/a | n/a |
| Visible anchor FPS(partial + coarse) | 150 | 0.072260 | 0.234357 | -4.14% | n/a | n/a | n/a | n/a |
| AdaPoinTr coarse | 150 | 0.069386 | 0.274315 | 0.00% | n/a | n/a | n/a | n/a |
| Naive PoinTr-IF seed 570 | 150 | 0.069320 | 0.274787 | 0.0957% | [-0.0457%, 0.2390%] | 87/63 | 0.1937 | 0.09366 |
| Gated-local PoinTr-IF seed 570 | 150 | 0.069243 | 0.274894 | 0.2062% | [0.0841%, 0.3301%] | 97/53 | 0.001277 | 0.0002435 |
| GMV-PoinTr-IF seed 571 | 150 | 0.069219 | 0.276132 | 0.2405% | [0.0832%, 0.4234%] | 96/54 | 0.005839 | 0.003713 |
| GMV-PoinTr-IF seed 570 | 150 | 0.069101 | 0.277758 | 0.4107% | [0.2389%, 0.6037%] | 100/50 | 1.33e-05 | 1.02e-06 |
| GMV category selector seed 570 | 150 | 0.069104 | 0.277690 | 0.4074% | see refined stats | 144 refined / 6 coarse selected | n/a | n/a |
| Oracle GMV seed 570, non-deployable | 150 | 0.068973 | 0.279554 | 0.5956% | n/a | 100 refined / 50 coarse chosen | n/a | n/a |

All rows use the same 150 held-out test sample IDs. The final seed 570 and seed 571 evaluations both use `--eval-seed 200570` so the resampled partial/coarse/GT point subsets are identical across training seeds.

## Data Scaling Audit

PCN remained unavailable because it requires manual gated access in this environment. The available local real benchmark is Projected ShapeNet55/34. The official split metadata is much larger, but this workspace contains only 1,500 usable projected partial point clouds paired with extracted ground truth and AdaPoinTr coarse exports:

| split | usable samples | role |
|---|---:|---|
| train | 1200 | refiner training |
| val | 150 | checkpoint selection and selector policy fitting |
| test | 150 | held-out evaluation |

Because no additional projected partial files were locally available, this pass did not fabricate a larger split. Instead, it reran the final method under a second training seed and pinned deterministic evaluation.

## Method Change

`GMV-PoinTr-IF` is implemented in `src/pointr_if/models.py` as `GatedMultiViewSelfStructureRefiner`.

Key components:

- Point-set encoder over partial and AdaPoinTr coarse points, with source flags.
- Local geometry features: nearest partial distance, nearest coarse distance, normalized radius, and local KNN offset statistics.
- Six canonical orthographic self-structure views from partial and coarse points.
- Tiny CNN over multi-view occupancy/depth maps, queried back at each coarse point by canonical projection.
- Learned residual gate with conservative initialization and bounded residual scale.
- Gate regularization in `src/pointr_if/train_refiner.py`, including improvement-target BCE, sparsity, and residual L2 terms.

Configs:

```text
configs/real_projected_shapenet55_gated_local_if.yaml
configs/real_projected_shapenet55_gmv_if.yaml
```

## Reproduction Commands

Primary GMV seed 570 training:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.train \
  --config configs/real_projected_shapenet55_gmv_if.yaml \
  --train-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/train_triplets.csv \
  --val-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv \
  --out-dir outputs/real_projected_shapenet55_gmv_if \
  --epochs 40 --batch-size 4 --num-workers 4 --seed 570 --device cuda
```

Primary test evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.evaluate \
  --manifest data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv \
  --checkpoint outputs/real_projected_shapenet55_gmv_if/best_model.pt \
  --out-dir outputs/real_projected_shapenet55_gmv_if/test_eval \
  --batch-size 4 --num-workers 4 --device cuda --eval-seed 200570 \
  --save-predictions --save-visualizations --max-visualizations 12
```

Paired statistics:

```bash
.venv/bin/python tools/analyze_real_results_stats.py \
  outputs/real_projected_shapenet55_gmv_if/test_eval/per_sample_metrics.csv \
  --out-dir outputs/real_projected_shapenet55_gmv_if/test_eval/stats \
  --baseline coarse --candidate refined --bootstrap 5000 --seed 570
```

Second seed:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.train \
  --config configs/real_projected_shapenet55_gmv_if.yaml \
  --train-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/train_triplets.csv \
  --val-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv \
  --out-dir outputs/real_projected_shapenet55_gmv_if_seed571 \
  --epochs 40 --batch-size 4 --num-workers 4 --seed 571 --device cuda
```

## Artifact Paths

Primary result:

```text
outputs/real_projected_shapenet55_gmv_if/
outputs/real_projected_shapenet55_gmv_if/val_eval/metrics.json
outputs/real_projected_shapenet55_gmv_if/val_eval/stats/paired_stats.md
outputs/real_projected_shapenet55_gmv_if/test_eval/metrics.json
outputs/real_projected_shapenet55_gmv_if/test_eval/stats/paired_stats.md
outputs/real_projected_shapenet55_gmv_if/test_eval/ranked_qualitative.png
```

Replication result:

```text
outputs/real_projected_shapenet55_gmv_if_seed571/
outputs/real_projected_shapenet55_gmv_if_seed571/test_eval/metrics.json
outputs/real_projected_shapenet55_gmv_if_seed571/test_eval/stats/paired_stats.md
```

Packaging:

```text
tools/make_final_archive.py
deliverables/pointr_if_hard_second_pass.zip
deliverables/archive_listing.txt
```

