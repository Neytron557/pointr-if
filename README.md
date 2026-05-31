# PoinTr-IF: Real Point-Cloud Completion Refinement

This repository implements a lightweight PoinTr-IF refinement module for point-cloud completion. The method keeps PoinTr/AdaPoinTr as the frozen completion backbone, exports real coarse completions, then trains a small implicit/residual refiner from `(partial point cloud, coarse completion, complete ground truth)` triplets.

## Current Real Result

The current checked result is a real-data second-pass run, not a synthetic smoke test. The final method is `GMV-PoinTr-IF`, a gated multi-view self-structure refiner trained on frozen AdaPoinTr completions.

- Dataset: Projected ShapeNet55/34 under `external/PoinTr/data/ShapeNet55-34`.
- Data source: official ShapeNet55/34 complete point clouds plus projected partial point clouds.
- Split used: 1200 train samples, 150 validation samples, 150 held-out test samples.
- Categories: 53 train categories, 21 validation categories, 23 test categories.
- Backbone checkpoint: `external/PoinTr/pretrained/AdaPoinTr_ps55.pth`.
- Backbone config: `external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml`.
- Final refiner checkpoint: `outputs/real_projected_shapenet55_gmv_if/best_model.pt`.
- GPU: NVIDIA TITAN X (Pascal), 12 GB VRAM.
- PyTorch: `2.7.1+cu126`; CUDA was available during export, training, and evaluation.

Official PCN data was not available through the environment without manual gated access, so this run uses the prompt's Priority B real benchmark path: ShapeNet55/34 / Projected ShapeNet.

## Metrics

All methods below were evaluated on the same held-out real test sample IDs with the same Chamfer and F-score implementation. Final GMV evaluations use `--eval-seed 200570` so point resampling is fixed across training seeds.

| method | n | chamfer | fscore | CD improvement vs AdaPoinTr coarse | bootstrap 95% CI | paired p |
|---|---:|---:|---:|---:|---:|---:|
| Partial input | 150 | 0.158159 | 0.175398 | -127.94% | n/a | n/a |
| Visible anchor FPS(partial + coarse) | 150 | 0.072260 | 0.234357 | -4.14% | n/a | n/a |
| AdaPoinTr coarse | 150 | 0.069386 | 0.274315 | 0.00% | n/a | n/a |
| Naive PoinTr-IF seed 570 | 150 | 0.069320 | 0.274787 | 0.0957% | [-0.0457%, 0.2390%] | 0.1937 |
| Gated-local PoinTr-IF seed 570 | 150 | 0.069243 | 0.274894 | 0.2062% | [0.0841%, 0.3301%] | 0.001277 |
| GMV-PoinTr-IF seed 571 | 150 | 0.069219 | 0.276132 | 0.2405% | [0.0832%, 0.4234%] | 0.005839 |
| GMV-PoinTr-IF seed 570 | 150 | 0.069101 | 0.277758 | 0.4107% | [0.2389%, 0.6037%] | 1.33e-05 |

The old naive refiner is retained as a baseline because its gain was not statistically significant. The final gated multi-view refiner is still a lightweight post-processor, but its held-out improvement is larger and significant under paired bootstrap, paired t-test, and Wilcoxon tests.

## Fourth-Pass MVC Work

The third-pass SEED/candidate-bank post-processing direction is no longer treated as deployable. The fourth pass implements `MVC-PoinTr-IF`, which fine-tunes AdaPoinTr itself with multi-view consistency rather than refining exported points after inference.

The audited FPS-4096 held-out baseline is `0.044413` CD, so the fourth-pass success gate is `<=0.04219` held-out test CD. No fourth-pass success is claimed until a validation-selected MVC checkpoint reaches that held-out threshold under `n_partial=2048`, `n_coarse=4096`, `n_gt=4096`, `n_output=4096`, `eval_seed=200570`, and `input_seed=570`.

Implemented entrypoints:

```bash
bash scripts/build_fourth_pass_groups.sh
bash scripts/train_mvc_pointr_if.sh outputs/fourth_pass/adapointr_ft_cd
bash scripts/evaluate_mvc_pointr_if.sh outputs/fourth_pass/adapointr_ft_cd/ckpt-best.pth outputs/fourth_pass/adapointr_ft_cd/test_eval test
```

See `docs/FOURTH_PASS_MV_CONSISTENCY_REPORT.md` for the group audit, generated train-only view summary, smoke checks, and full run commands.

## Key Artifacts

- Coarse export manifests: `data/real_projected_shapenet55_adapointr_predictions/manifests/`.
- Export logs: `data/real_projected_shapenet55_adapointr_predictions/logs/`.
- Manifest validation reports: `reports/real_projected_shapenet55_*_manifest_validation.json`.
- Naive baseline run: `outputs/real_projected_shapenet55_adapointr_if_fast/`.
- Gated-local run: `outputs/real_projected_shapenet55_gated_local_if/`.
- Final GMV run: `outputs/real_projected_shapenet55_gmv_if/`.
- Final GMV replicate: `outputs/real_projected_shapenet55_gmv_if_seed571/`.
- Final paired stats: `outputs/real_projected_shapenet55_gmv_if/test_eval/stats/paired_stats.md`.
- Ranked qualitative grid: `outputs/real_projected_shapenet55_gmv_if/test_eval/ranked_qualitative.png`.
- Second-pass report: `docs/HARD_SECOND_PASS_REPORT.md`.

## Reproduce

Export AdaPoinTr coarse completions:

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

Train the final GMV refiner:

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

Evaluate test split:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pointr_if.evaluate \
  --manifest data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv \
  --checkpoint outputs/real_projected_shapenet55_gmv_if/best_model.pt \
  --out-dir outputs/real_projected_shapenet55_gmv_if/test_eval \
  --batch-size 4 \
  --num-workers 4 \
  --device cuda \
  --eval-seed 200570 \
  --save-predictions \
  --save-visualizations \
  --max-visualizations 12
```

Create paired statistics and ranked qualitative grid:

```bash
.venv/bin/python tools/analyze_real_results_stats.py \
  outputs/real_projected_shapenet55_gmv_if/test_eval/per_sample_metrics.csv \
  --out-dir outputs/real_projected_shapenet55_gmv_if/test_eval/stats \
  --baseline coarse \
  --candidate refined \
  --bootstrap 5000 \
  --seed 570

.venv/bin/python tools/make_ranked_qualitative_figures.py \
  --manifest data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv \
  --eval-dir outputs/real_projected_shapenet55_gmv_if/test_eval \
  --delta-csv outputs/real_projected_shapenet55_gmv_if/test_eval/stats/per_sample_delta.csv \
  --out outputs/real_projected_shapenet55_gmv_if/test_eval/ranked_qualitative.png \
  --per-group 3
```

## Debug-Only Synthetic Checks

The repository still includes synthetic smoke tests and tiny manifest checks for development:

```bash
source .venv/bin/activate
pytest -q
bash scripts/run_smoke.sh
bash scripts/run_ablation_synthetic.sh
```

Synthetic metrics are debug-only and should not be presented as final project results.

## Documentation Map

- `docs/EXPERIMENT_LOG.md`: real run commands, metrics, and artifact paths.
- `docs/HARD_SECOND_PASS_REPORT.md`: final second-pass result table, paired stats, data cap audit, and packaging notes.
- `docs/CODEX_STATUS.md`: latest environment, data, checkpoint, and result status.
- `docs/FINAL_REPORT_TEMPLATE.md`: report text with the current real-results table.
- `docs/SERVER_RUNBOOK.md`: server workflow for reproducing or extending the real run.
- `docs/PROFESSOR_FEEDBACK_RESPONSE.md`: framing for the project contribution.
- `docs/REFERENCES.md`: papers to cite.
