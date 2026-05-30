# Experiment Log

## Hard Second Pass: GMV-PoinTr-IF

Date: 2026-05-30

The first real run below is retained as the naive baseline, but its held-out test improvement was only 0.0957% and was not statistically significant. The hard second pass added paired statistics, deterministic evaluation seeding, archive packaging, a gated-local refiner, and the final gated multi-view self-structure refiner (`GMV-PoinTr-IF`).

Local scaling audit:

- Official Projected ShapeNet55/34 metadata lists larger train/test splits.
- This workspace has 1,500 usable projected partial samples paired with extracted GT and AdaPoinTr coarse exports.
- The real experiment therefore remains 1200 train / 150 val / 150 test, with an added seed 571 replicate for the final method.
- Final cross-seed evaluation uses `--eval-seed 200570`.

Final held-out test table:

| method | n | chamfer | fscore | CD improvement vs coarse | bootstrap 95% CI | paired t p | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| partial input | 150 | 0.158159 | 0.175398 | -127.94% | n/a | n/a | n/a |
| visible anchor FPS(partial + coarse) | 150 | 0.072260 | 0.234357 | -4.14% | n/a | n/a | n/a |
| AdaPoinTr coarse | 150 | 0.069386 | 0.274315 | 0.00% | n/a | n/a | n/a |
| naive PoinTr-IF seed 570 | 150 | 0.069320 | 0.274787 | 0.0957% | [-0.0457%, 0.2390%] | 0.1937 | 0.09366 |
| gated-local PoinTr-IF seed 570 | 150 | 0.069243 | 0.274894 | 0.2062% | [0.0841%, 0.3301%] | 0.001277 | 0.0002435 |
| GMV-PoinTr-IF seed 571 | 150 | 0.069219 | 0.276132 | 0.2405% | [0.0832%, 0.4234%] | 0.005839 | 0.003713 |
| GMV-PoinTr-IF seed 570 | 150 | 0.069101 | 0.277758 | 0.4107% | [0.2389%, 0.6037%] | 1.33e-05 | 1.02e-06 |

Primary commands:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.train \
  --config configs/real_projected_shapenet55_gmv_if.yaml \
  --train-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/train_triplets.csv \
  --val-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv \
  --out-dir outputs/real_projected_shapenet55_gmv_if \
  --epochs 40 --batch-size 4 --num-workers 4 --seed 570 --device cuda

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.evaluate \
  --manifest data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv \
  --checkpoint outputs/real_projected_shapenet55_gmv_if/best_model.pt \
  --out-dir outputs/real_projected_shapenet55_gmv_if/test_eval \
  --batch-size 4 --num-workers 4 --device cuda --eval-seed 200570 \
  --save-predictions --save-visualizations --max-visualizations 12

.venv/bin/python tools/analyze_real_results_stats.py \
  outputs/real_projected_shapenet55_gmv_if/test_eval/per_sample_metrics.csv \
  --out-dir outputs/real_projected_shapenet55_gmv_if/test_eval/stats \
  --baseline coarse --candidate refined --bootstrap 5000 --seed 570
```

Primary artifacts:

```text
outputs/real_projected_shapenet55_gmv_if/test_eval/metrics.json
outputs/real_projected_shapenet55_gmv_if/test_eval/stats/paired_stats.md
outputs/real_projected_shapenet55_gmv_if/test_eval/selector_category/selector_metrics.json
outputs/real_projected_shapenet55_gmv_if/test_eval/ranked_qualitative.png
outputs/real_projected_shapenet55_gmv_if_seed571/test_eval/stats/paired_stats.md
docs/HARD_SECOND_PASS_REPORT.md
```

## Real Projected ShapeNet55/34 Run

Date: 2026-05-30

This is the current final experiment run for the project. It uses real Projected ShapeNet55/34 data and a real AdaPoinTr checkpoint. Synthetic runs remain available only as debug checks.

### Environment

- Project root: `/home/ubuntu/ai_and_ml/pointr_if_project`
- Python: 3.12.3
- PyTorch: `2.7.1+cu126`
- CUDA available to PyTorch: `true`
- GPU: NVIDIA TITAN X (Pascal), 12288 MiB VRAM
- NVIDIA driver: 580.159.03

Verification commands:

```bash
pwd
nvidia-smi
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

### Dataset and Checkpoint

- Dataset: Projected ShapeNet55/34 under `external/PoinTr/data/ShapeNet55-34`.
- Complete point clouds: official ShapeNet55/34 `.npy` files.
- Partial point clouds: projected partial `.pcd` files, symlinked as `projected_partial_noise`.
- Backbone checkpoint: `external/PoinTr/pretrained/AdaPoinTr_ps55.pth`.
- Backbone config: `external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml`.
- PCN note: official PCN download required gated/manual access in this environment, so this run uses the prompt's Priority B ShapeNet55/34 path.

Split summary:

| split | samples | categories |
|---|---:|---:|
| train | 1200 | 53 |
| val | 150 | 21 |
| test | 150 | 23 |

Selected subset files:

```text
data/real_projected_shapenet55_subset/train_selected_members.json
data/real_projected_shapenet55_subset/test_selected_members.json
```

Manifest validation reports:

```text
reports/real_projected_shapenet55_train_manifest_validation.json
reports/real_projected_shapenet55_val_manifest_validation.json
reports/real_projected_shapenet55_test_manifest_validation.json
```

These reports were regenerated with `--max-samples 999999`, so all rows were path-checked and loaded for numeric sanity.

### AdaPoinTr Coarse Export

Command log:

```text
data/real_projected_shapenet55_adapointr_predictions/logs/command_log.txt
```

Train export command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/export_pointr_predictions.py \
  --pointr-root external/PoinTr \
  --config external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml \
  --checkpoint external/PoinTr/pretrained/AdaPoinTr_ps55.pth \
  --data-root external/PoinTr/data/ShapeNet55-34 \
  --split train \
  --split-name train \
  --selected-json data/real_projected_shapenet55_subset/train_selected_members.json \
  --out-root data/real_projected_shapenet55_adapointr_predictions \
  --limit 1200 \
  --batch-size 2 \
  --device cuda \
  --seed 570
```

The validation and test manifests were generated from the real test member list with deterministic offsets:

- Validation: `--offset 0 --limit 150 --split-name val`
- Test: `--offset 150 --limit 150 --split-name test`

Export summaries:

```text
data/real_projected_shapenet55_adapointr_predictions/logs/train_export_summary.json
data/real_projected_shapenet55_adapointr_predictions/logs/val_export_summary.json
data/real_projected_shapenet55_adapointr_predictions/logs/test_export_summary.json
```

### Refiner Training

Command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.train \
  --config configs/real_projected_shapenet55_adapointr_if_fast.yaml \
  --train-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/train_triplets.csv \
  --val-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv \
  --out-dir outputs/real_projected_shapenet55_adapointr_if_fast \
  --epochs 20 \
  --batch-size 4 \
  --num-workers 4 \
  --seed 570 \
  --device cuda
```

Training artifacts:

```text
outputs/real_projected_shapenet55_adapointr_if_fast/best_model.pt
outputs/real_projected_shapenet55_adapointr_if_fast/last_model.pt
outputs/real_projected_shapenet55_adapointr_if_fast/metrics.csv
outputs/real_projected_shapenet55_adapointr_if_fast/stdout.log
outputs/real_projected_shapenet55_adapointr_if_fast/summary.json
outputs/real_projected_shapenet55_adapointr_if_fast/command_log.txt
outputs/real_projected_shapenet55_adapointr_if_fast/resolved_config.yaml
```

Best validation Chamfer during training:

| metric | value |
|---|---:|
| AdaPoinTr coarse val Chamfer | 0.071001 |
| PoinTr-IF refined val Chamfer | 0.070887 |
| CD improvement | 0.16% |
| coarse F-score | 0.303374 |
| refined F-score | 0.304251 |

### Evaluation

Validation command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.evaluate \
  --manifest data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv \
  --checkpoint outputs/real_projected_shapenet55_adapointr_if_fast/best_model.pt \
  --out-dir outputs/real_projected_shapenet55_adapointr_if_fast/val_eval \
  --batch-size 4 \
  --num-workers 4 \
  --device cuda \
  --save-predictions \
  --save-visualizations \
  --max-visualizations 12
```

Test command:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.evaluate \
  --manifest data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv \
  --checkpoint outputs/real_projected_shapenet55_adapointr_if_fast/best_model.pt \
  --out-dir outputs/real_projected_shapenet55_adapointr_if_fast/test_eval \
  --batch-size 4 \
  --num-workers 4 \
  --device cuda \
  --save-predictions \
  --save-visualizations \
  --max-visualizations 12
```

Overall results:

| eval | method | n | chamfer | fscore | CD improvement vs coarse |
|---|---|---:|---:|---:|---:|
| val | partial input | 150 | 0.167753 | 0.194933 | -136.57% |
| val | visible anchor FPS(partial + coarse) | 150 | 0.073574 | 0.268183 | -3.75% |
| val | AdaPoinTr coarse | 150 | 0.070912 | 0.303461 | 0.00% |
| val | PoinTr-IF refined | 150 | 0.070826 | 0.303528 | 0.12% |
| test | partial input | 150 | 0.158159 | 0.175398 | -127.94% |
| test | visible anchor FPS(partial + coarse) | 150 | 0.072260 | 0.234357 | -4.14% |
| test | AdaPoinTr coarse | 150 | 0.069386 | 0.274315 | 0.00% |
| test | PoinTr-IF refined | 150 | 0.069320 | 0.274787 | 0.10% |

Evaluation artifacts:

```text
outputs/real_projected_shapenet55_adapointr_if_fast/val_eval/metrics.json
outputs/real_projected_shapenet55_adapointr_if_fast/val_eval/per_sample_metrics.csv
outputs/real_projected_shapenet55_adapointr_if_fast/val_eval/category_metrics.csv
outputs/real_projected_shapenet55_adapointr_if_fast/test_eval/metrics.json
outputs/real_projected_shapenet55_adapointr_if_fast/test_eval/per_sample_metrics.csv
outputs/real_projected_shapenet55_adapointr_if_fast/test_eval/category_metrics.csv
outputs/real_projected_shapenet55_adapointr_if_fast/real_results_summary_table.md
```

Qualitative outputs:

```text
outputs/real_projected_shapenet55_adapointr_if_fast/test_eval/visualizations/
outputs/real_projected_shapenet55_adapointr_if_fast/test_eval/qualitative_grid.png
```

The test evaluation saved 12 qualitative PNG visualizations and 300 `.npy` prediction files for anchor/refined outputs.

### Interpretation

The real run shows a small but measured gain for the trained refiner over AdaPoinTr coarse output on both validation and held-out test subsets. The gain is modest because AdaPoinTr is already a strong pretrained baseline and the refiner is intentionally lightweight for the project deadline. The visible-anchor FPS baseline did not improve Chamfer on this subset, so the final report should present it as a negative ablation rather than a success claim.

## Debug-Only Synthetic Runs

Earlier synthetic smoke and ablation outputs remain useful for code debugging, but they are not final benchmark evidence.

Synthetic smoke run:

```bash
bash scripts/run_smoke.sh
```

Synthetic ablation run:

```bash
bash scripts/run_ablation_synthetic.sh
```

Representative debug-only synthetic table:

| Run | Coarse CD | Refined CD | CD improvement | Coarse F-score | Refined F-score | F-score gain |
|---|---:|---:|---:|---:|---:|---:|
| global_only | 0.13639 | 0.13434 | 1.51% | 0.38220 | 0.39028 | +0.00808 |
| local_occ | 0.13639 | 0.13360 | 2.05% | 0.38220 | 0.39174 | +0.00954 |
| no_occ | 0.13639 | 0.13342 | 2.18% | 0.38220 | 0.39102 | +0.00882 |

These synthetic numbers must be labeled debug-only if mentioned.
