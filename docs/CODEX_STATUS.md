# Codex Status

**Last updated:** 2026-05-30

## Status

Hard second-pass real-data ShapeNet55/34 training and evaluation are complete. The repository now contains real AdaPoinTr coarse exports, naive/gated-local/GMV PoinTr-IF training logs, deterministic held-out evaluations, paired statistics, per-category metrics, selector baselines, and ranked qualitative visualizations.

## Environment Summary

- Project root: `/home/ubuntu/ai_and_ml/pointr_if_project`
- Python: 3.12.3 in `.venv`
- PyTorch: `2.7.1+cu126`
- CUDA available to PyTorch: `true`
- GPU: NVIDIA TITAN X (Pascal), 12288 MiB VRAM
- NVIDIA driver: 580.159.03
- Git status for this package: no `.git` repository is present in `pointr_if_project`

## Data and Checkpoints

- PCN: not available through the environment without manual gated access.
- Real benchmark used: Projected ShapeNet55/34.
- Data root: `external/PoinTr/data/ShapeNet55-34`.
- Train/val/test sample counts: 1200 / 150 / 150.
- Train/val/test category counts: 53 / 21 / 23.
- AdaPoinTr checkpoint: `external/PoinTr/pretrained/AdaPoinTr_ps55.pth`.
- AdaPoinTr config: `external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml`.
- Final refiner checkpoint: `outputs/real_projected_shapenet55_gmv_if/best_model.pt`.
- Replicate checkpoint: `outputs/real_projected_shapenet55_gmv_if_seed571/best_model.pt`.
- Deterministic evaluation seed for final GMV comparisons: `200570`.

## Commands Run

Environment gate:

```bash
pwd
ls -lah
python --version
which python
nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
df -h
free -h
```

AdaPoinTr coarse exports:

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

Validation and test exports used the same command with `--split test`, the test selected-member JSON, and deterministic offsets:

- Validation: `--split-name val --offset 0 --limit 150`
- Test: `--split-name test --offset 150 --limit 150`

Manifest validation:

```bash
.venv/bin/python tools/validate_triplet_manifest.py \
  data/real_projected_shapenet55_adapointr_predictions/manifests/train_triplets.csv \
  --max-samples 999999 \
  --json-out reports/real_projected_shapenet55_train_manifest_validation.json

.venv/bin/python tools/validate_triplet_manifest.py \
  data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv \
  --max-samples 999999 \
  --json-out reports/real_projected_shapenet55_val_manifest_validation.json

.venv/bin/python tools/validate_triplet_manifest.py \
  data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv \
  --max-samples 999999 \
  --json-out reports/real_projected_shapenet55_test_manifest_validation.json
```

Refiner training:

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

Hard second-pass GMV training:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.train \
  --config configs/real_projected_shapenet55_gmv_if.yaml \
  --train-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/train_triplets.csv \
  --val-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv \
  --out-dir outputs/real_projected_shapenet55_gmv_if \
  --epochs 40 \
  --batch-size 4 \
  --num-workers 4 \
  --seed 570 \
  --device cuda
```

Second seed:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.train \
  --config configs/real_projected_shapenet55_gmv_if.yaml \
  --train-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/train_triplets.csv \
  --val-manifest data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv \
  --out-dir outputs/real_projected_shapenet55_gmv_if_seed571 \
  --epochs 40 \
  --batch-size 4 \
  --num-workers 4 \
  --seed 571 \
  --device cuda
```

Evaluation:

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

Summary generation:

```bash
.venv/bin/python tools/compare_real_results.py \
  outputs/real_projected_shapenet55_adapointr_if_fast/val_eval \
  outputs/real_projected_shapenet55_adapointr_if_fast/test_eval \
  --out outputs/real_projected_shapenet55_adapointr_if_fast/real_results_summary_table.md

.venv/bin/python tools/make_qualitative_grid.py \
  --image-dir outputs/real_projected_shapenet55_adapointr_if_fast/test_eval/visualizations \
  --out outputs/real_projected_shapenet55_adapointr_if_fast/test_eval/qualitative_grid.png \
  --limit 6 \
  --columns 2
```

## Real Metrics

Held-out test, all on the same sample IDs:

| method | n | chamfer | fscore | CD improvement vs coarse | bootstrap 95% CI | paired t p |
|---|---:|---:|---:|---:|---:|---:|
| partial input | 150 | 0.158159 | 0.175398 | -127.94% | n/a | n/a |
| visible anchor FPS(partial + coarse) | 150 | 0.072260 | 0.234357 | -4.14% | n/a | n/a |
| AdaPoinTr coarse | 150 | 0.069386 | 0.274315 | 0.00% | n/a | n/a |
| naive PoinTr-IF seed 570 | 150 | 0.069320 | 0.274787 | 0.0957% | [-0.0457%, 0.2390%] | 0.1937 |
| gated-local PoinTr-IF seed 570 | 150 | 0.069243 | 0.274894 | 0.2062% | [0.0841%, 0.3301%] | 0.001277 |
| GMV-PoinTr-IF seed 571 | 150 | 0.069219 | 0.276132 | 0.2405% | [0.0832%, 0.4234%] | 0.005839 |
| GMV-PoinTr-IF seed 570 | 150 | 0.069101 | 0.277758 | 0.4107% | [0.2389%, 0.6037%] | 1.33e-05 |

## Artifact Paths

```text
data/real_projected_shapenet55_adapointr_predictions/manifests/
data/real_projected_shapenet55_adapointr_predictions/logs/
reports/real_projected_shapenet55_*_manifest_validation.json
outputs/real_projected_shapenet55_adapointr_if_fast/
outputs/real_projected_shapenet55_adapointr_if_fast/val_eval/
outputs/real_projected_shapenet55_adapointr_if_fast/test_eval/
outputs/real_projected_shapenet55_adapointr_if_fast/test_eval/qualitative_grid.png
outputs/real_projected_shapenet55_gated_local_if/
outputs/real_projected_shapenet55_gmv_if/
outputs/real_projected_shapenet55_gmv_if/test_eval/stats/paired_stats.md
outputs/real_projected_shapenet55_gmv_if/test_eval/ranked_qualitative.png
outputs/real_projected_shapenet55_gmv_if_seed571/
docs/HARD_SECOND_PASS_REPORT.md
```

## Remaining Caveats

- This is a real Projected ShapeNet55/34 subset result, not an official PCN result.
- The local projected data cap is 1200/150/150 usable train/val/test samples. The official split metadata is larger, but additional local projected partial files were not available.
- The final GMV gain is statistically significant but still modest. It should be presented as a lightweight refinement improvement over a strong AdaPoinTr baseline, not as a replacement for the backbone.
- The visible-anchor baseline is a negative ablation for Chamfer on this subset.
