# Codex Status

**Last updated:** 2026-06-01

## Current Status

The final selected result is the A100 PCN Run B experiment: a backbone-feature-conditioned `feature_gmv` refiner trained on AdaPoinTr PCN decoder features. The checkpoint is tracked at `checkpoints/run_b/best.pt`.

Run A, the multi-view consistency AdaPoinTr fine-tuning path, was implemented and evaluated but failed because it degraded held-out PCN performance. Run A checkpoints are not tracked.

## Final Run B Test Result

| method | n | chamfer | fscore |
|---|---:|---:|---:|
| anchor | 1200 | 0.042830 | 0.596364 |
| coarse | 1200 | 0.043141 | 0.586850 |
| partial | 1200 | 0.151631 | 0.326790 |
| refined | 1200 | 0.041547 | 0.613169 |

Paired coarse-vs-refined result:

- Mean CD improvement: `3.6936%`.
- Bootstrap 95% CI: `[3.6073%, 3.7810%]`.
- Positive / negative / zero samples: `1186 / 14 / 0`.
- Mean F-score delta: `0.02631900`.

## Tracked Artifacts

- Best Run B checkpoint: `checkpoints/run_b/best.pt`.
- Run B result files: `results/a100_full_improvement/run_b_feature_refiner/`.
- Run A failure metrics: `results/a100_full_improvement/run_a_pcn_mvc/`.
- Reproduction report: `docs/A100_PCN_RESULTS.md`.
- A100 launcher: `scripts/train_a100_full_improvement.sh`.
- Best-checkpoint evaluator: `scripts/evaluate_run_b_best.sh`.

## Environment Used

- Project root: `/lustre/home/ziya/PoinTr/pointr_if_project`.
- Python: `.venv` created from `requirements.txt`.
- PyTorch: `2.11.0+cu128`.
- GPU: physical `gpu_id=1`, selected by `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.
- Visible CUDA device: `NVIDIA Graphics Device`, about 98 GB VRAM.
- CPU cap: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 NUM_THREADS=8 NUM_WORKERS=8`.

## Data Requirement

The gated PCN dataset is not tracked in git. To reproduce, place it at:

```text
data/ShapeNetCompletion
```

Then run:

```bash
bash scripts/setup_a100_venv.sh
RUN_A=0 RUN_B=1 FORCE_EXPORT=1 bash scripts/train_a100_full_improvement.sh outputs/a100_full_improvement_repro
```
