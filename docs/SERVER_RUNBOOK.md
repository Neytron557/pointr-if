# Server Runbook

This runbook reproduces or extends the real Projected ShapeNet55/34 experiment. The latest completed run used an NVIDIA TITAN X (Pascal) with 12 GB VRAM.

## 1. Verify GPU and Environment

```bash
cd /home/ubuntu/ai_and_ml/pointr_if_project
source .venv/bin/activate
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

Do not run real training on CPU. If CUDA is unavailable, fix the driver/PyTorch environment first.

Known working environment:

- Python 3.12.3
- PyTorch `2.7.1+cu126`
- NVIDIA TITAN X (Pascal), 12288 MiB VRAM
- NVIDIA driver 580.159.03

## 2. Data and Checkpoints

Required existing paths for the completed run:

```text
external/PoinTr/data/ShapeNet55-34
external/PoinTr/data/ShapeNet55-34/pcd
external/PoinTr/data/ShapeNet55-34/shapenet_pc
external/PoinTr/pretrained/AdaPoinTr_ps55.pth
external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml
```

The partial point-cloud directory is symlinked as:

```text
external/PoinTr/data/ShapeNet55-34/projected_partial_noise -> pcd
```

PCN should still be preferred for a future official PCN run, but it was not accessible from this environment without manual gated download. The current completed result uses the accepted ShapeNet55/34 path.

## 3. Export AdaPoinTr Coarse Predictions

Train split:

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

Validation split:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/export_pointr_predictions.py \
  --pointr-root external/PoinTr \
  --config external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml \
  --checkpoint external/PoinTr/pretrained/AdaPoinTr_ps55.pth \
  --data-root external/PoinTr/data/ShapeNet55-34 \
  --split test \
  --split-name val \
  --selected-json data/real_projected_shapenet55_subset/test_selected_members.json \
  --out-root data/real_projected_shapenet55_adapointr_predictions \
  --offset 0 \
  --limit 150 \
  --batch-size 2 \
  --device cuda \
  --seed 570
```

Held-out test split:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/export_pointr_predictions.py \
  --pointr-root external/PoinTr \
  --config external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml \
  --checkpoint external/PoinTr/pretrained/AdaPoinTr_ps55.pth \
  --data-root external/PoinTr/data/ShapeNet55-34 \
  --split test \
  --split-name test \
  --selected-json data/real_projected_shapenet55_subset/test_selected_members.json \
  --out-root data/real_projected_shapenet55_adapointr_predictions \
  --offset 150 \
  --limit 150 \
  --batch-size 2 \
  --device cuda \
  --seed 570
```

Expected outputs:

```text
data/real_projected_shapenet55_adapointr_predictions/manifests/train_triplets.csv
data/real_projected_shapenet55_adapointr_predictions/manifests/val_triplets.csv
data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv
data/real_projected_shapenet55_adapointr_predictions/logs/
```

## 4. Validate Manifests

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

The manifest schema is:

```text
sample_id,category,partial_path,coarse_path,gt_path,split
```

## 5. Train PoinTr-IF

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

Expected training outputs:

```text
outputs/real_projected_shapenet55_adapointr_if_fast/best_model.pt
outputs/real_projected_shapenet55_adapointr_if_fast/last_model.pt
outputs/real_projected_shapenet55_adapointr_if_fast/metrics.csv
outputs/real_projected_shapenet55_adapointr_if_fast/summary.json
outputs/real_projected_shapenet55_adapointr_if_fast/stdout.log
outputs/real_projected_shapenet55_adapointr_if_fast/command_log.txt
```

For memory pressure on 12 GB VRAM, reduce `--batch-size` to 2 before changing model settings.

## 6. Evaluate and Visualize

Validation:

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

Held-out test:

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

Create summary outputs:

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

## 7. Report Numbers

Use the saved table:

```text
outputs/real_projected_shapenet55_adapointr_if_fast/real_results_summary_table.md
```

Current held-out test headline:

| method | n | chamfer | fscore | CD improvement vs coarse |
|---|---:|---:|---:|---:|
| partial input | 150 | 0.158159 | 0.175398 | -127.94% |
| visible anchor FPS(partial + coarse) | 150 | 0.072260 | 0.234357 | -4.14% |
| AdaPoinTr coarse | 150 | 0.069386 | 0.274315 | 0.00% |
| PoinTr-IF refined | 150 | 0.069320 | 0.274787 | 0.10% |
