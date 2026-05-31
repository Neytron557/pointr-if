# Fourth Pass MVC-PoinTr-IF Report

## Objective

The third-pass SEED/candidate-bank direction is stopped except for diagnostics. The fourth pass moves the improvement attempt into the completion backbone by fine-tuning AdaPoinTr with multi-view consistency.

Target protocol:

```text
resample_mode=fps
n_partial=2048
n_coarse=4096
n_gt=4096
n_output=4096
eval_seed=200570
input_seed=570
```

Current audited held-out test baseline from the third-pass FPS-4096 audit:

```text
AdaPoinTr coarse CD = 0.04441340666264296
5% target CD <= 0.04219235
```

No fourth-pass held-out test success is claimed until a validation-selected MVC checkpoint is evaluated on the held-out test split under this same protocol.

## Environment

```text
GPU: NVIDIA TITAN X (Pascal), 12288 MiB
PyTorch: 2.7.1+cu126
CUDA available: True
CUDA device: NVIDIA TITAN X (Pascal)
Backbone checkpoint: external/PoinTr/pretrained/AdaPoinTr_ps55.pth
Backbone config: external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml
```

## Implemented Artifacts

- `tools/build_multiview_groups.py`
- `tools/generate_train_projected_views.py`
- `tools/train_adapointr_multiview_consistency.py`
- `tools/export_finetuned_adapointr_predictions.py`
- `scripts/build_fourth_pass_groups.sh`
- `scripts/train_mvc_pointr_if.sh`
- `scripts/evaluate_mvc_pointr_if.sh`
- `scripts/run_fourth_pass_mvc_sweep.sh`
- `tests/test_fourth_pass_mvc_tools.py`

## Multi-View Group Audit

Command:

```bash
bash scripts/build_fourth_pass_groups.sh
```

Real grouped-view audit:

| split | rows | groups | min views | median views | max views | groups with >=2 views |
|---|---:|---:|---:|---:|---:|---:|
| train | 1200 | 1200 | 1 | 1 | 1 | 0 |
| val | 150 | 150 | 1 | 1 | 1 | 0 |

The local Projected ShapeNet subset has only one real projected partial per GT object. MVC therefore uses generated train-only partial views from train GT shapes. No val/test GT is used for view generation or training.

Generated train-only views:

```text
input groups: data/real_projected_shapenet55_groups/train_groups.json
augmented groups: data/real_projected_shapenet55_groups/train_groups_augmented.json
generated views: 4800
groups: 1200
members per augmented group: 5
points per generated view: 2048
image size: 64
```

The group audit is saved at `reports/real_projected_shapenet55_group_audit.json`. The generated-view summary is saved at `data/real_projected_shapenet55_groups/generated_train_views_summary.json`.

## MVC Training Implementation

The fine-tuning script loads AdaPoinTr from the official local config and checkpoint, then samples K partial views per GT object.

Loss:

```text
L = L_rec + lambda_target * L_target + lambda_self * L_self + lambda_missing * L_missing
```

Current implementation:

- `L_rec`: native AdaPoinTr `model.get_loss(...)`.
- `L_target`: Chamfer L1 between FPS(prediction, 4096) and FPS(GT, 4096).
- `L_self`: pairwise stop-gradient Chamfer between K predictions for the same GT.
- `L_missing`: set to zero because this AdaPoinTr output path exposes complete predictions, not explicit missing-region points.
- AMP and gradient clipping are supported.
- Default freeze policy is decoder/query-generator only; optional `decoder_plus_last_encoder` unfreezes the last encoder block with a smaller LR.
- Validation always compares the fine-tuned model prediction to the frozen AdaPoinTr coarse manifest under FPS-4096.

Important protocol fix:

- Metric FPS uses `eval_seed=200570`.
- AdaPoinTr inference partial sampling uses `input_seed=570`, matching the original coarse export.

## Smoke Checks

Focused tests:

```bash
.venv/bin/pytest -q tests/test_fourth_pass_mvc_tools.py
```

Result:

```text
4 passed
```

Full test suite:

```bash
.venv/bin/pytest -q
```

Result:

```text
27 passed
```

Bounded MVC smoke:

```bash
EPOCHS=1 MAX_TRAIN_BATCHES=1 MAX_VAL_SAMPLES=2 \
  bash scripts/train_mvc_pointr_if.sh outputs/fourth_pass/smoke_mvc
```

Result on two validation samples, for runtime validation only:

| method | n | CD | F-score |
|---|---:|---:|---:|
| coarse | 2 | 0.025047 | 0.857990 |
| refined | 2 | 0.029034 | 0.818417 |

Bounded no-consistency fine-tune smoke:

```bash
EPOCHS=1 MAX_TRAIN_BATCHES=2 MAX_VAL_SAMPLES=4 \
TRAIN_GROUPS=data/real_projected_shapenet55_groups/train_groups.json \
VIEWS_PER_OBJECT=1 LAMBDA_SELF=0.0 \
  bash scripts/train_mvc_pointr_if.sh outputs/fourth_pass/smoke_ft_cd
```

Result on four validation samples, for runtime validation only:

| method | n | CD | F-score |
|---|---:|---:|---:|
| coarse | 4 | 0.049032 | 0.534589 |
| refined | 4 | 0.050521 | 0.531509 |

The smoke results are not final claims; they only verify the training, checkpoint, and evaluation paths.

## Baseline Evaluator Audit

Command:

```bash
.venv/bin/python tools/export_finetuned_adapointr_predictions.py \
  --config external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml \
  --checkpoint external/PoinTr/pretrained/AdaPoinTr_ps55.pth \
  --manifest data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv \
  --out-dir outputs/fourth_pass/adapointr_baseline_eval_audit \
  --device cuda:0 \
  --batch-size 1 \
  --n-partial 2048 \
  --n-output 4096 \
  --n-gt 4096 \
  --eval-seed 200570 \
  --input-seed 570
```

Held-out test result:

| method | n | CD | F-score |
|---|---:|---:|---:|
| coarse manifest | 150 | 0.0444134067 | 0.593049 |
| direct original AdaPoinTr | 150 | 0.0444134231 | 0.593048 |

This reproduces the audited coarse baseline exactly on the coarse-manifest side. The direct original AdaPoinTr output is numerically equivalent to the saved manifest coarse output under `input_seed=570`.

Paired audit stats are saved under `outputs/fourth_pass/adapointr_baseline_eval_audit/stats`. As expected, direct original AdaPoinTr is not meaningfully different from the saved coarse manifest (`mean CD improvement = -0.0000%`, bootstrap 95% CI `[-0.0005%, 0.0004%]`).

## Full Run Commands

Run A: protocol-matched AdaPoinTr fine-tuning without consistency.

```bash
EPOCHS=8 \
TRAIN_GROUPS=data/real_projected_shapenet55_groups/train_groups.json \
VIEWS_PER_OBJECT=1 \
LAMBDA_SELF=0.0 \
bash scripts/train_mvc_pointr_if.sh outputs/fourth_pass/adapointr_ft_cd
```

Run B: MVC fine-tuning using train-only generated partial views.

```bash
EPOCHS=8 \
TRAIN_GROUPS=data/real_projected_shapenet55_groups/train_groups_augmented.json \
VIEWS_PER_OBJECT=2 \
LAMBDA_SELF=0.05 \
UNFREEZE=decoder_plus_last_encoder \
UNFREEZE_LAST_ENCODER_BLOCKS=1 \
LR_DECODER=2e-5 \
LR_ENCODER=1e-6 \
bash scripts/train_mvc_pointr_if.sh outputs/fourth_pass/adapointr_mvc_lam005_last_encoder
```

Evaluate the validation-selected checkpoint on held-out test:

```bash
bash scripts/evaluate_mvc_pointr_if.sh \
  outputs/fourth_pass/<run>/ckpt-best.pth \
  outputs/fourth_pass/<run>/test_eval \
  test
```

The evaluation script also writes paired statistics under `outputs/fourth_pass/<run>/test_eval/stats` using `tools/analyze_real_results_stats.py` with `baseline=coarse`, `candidate=refined`, `bootstrap=5000`, and `seed=570`.

Target gate:

```text
held-out test refined CD <= 0.04219
```

## Archive

Fourth-pass source, scripts, reports, and audit metrics are packaged at:

```text
deliverables/pointr_if_fourth_pass_mvc_tools.zip
```
