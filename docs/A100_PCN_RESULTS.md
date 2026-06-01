# A100 PCN Run B Results And Reproduction

This is the final deployable experiment for the repository. Run A was implemented and tested, but it degraded held-out PCN performance, so the repository keeps its metrics only and does not track its checkpoints. Run B is the selected result.

## Selected Result

- Dataset: PCN / ShapeNetCompletion, placed locally at `data/ShapeNetCompletion`.
- Backbone: AdaPoinTr PCN checkpoint, `external/PoinTr/pretrained/AdaPoinTr_PCN.pth`.
- Candidate: backbone-feature-conditioned `feature_gmv` refiner.
- Checkpoint: `checkpoints/run_b/best.pt`.
- Validation-selected epoch: `20`.
- Validation CD improvement: `3.887066%`.
- Held-out test samples: `1200`.

## Held-Out Test Metrics

| method | n | chamfer | fscore |
|---|---:|---:|---:|
| anchor | 1200 | 0.042830 | 0.596364 |
| coarse | 1200 | 0.043141 | 0.586850 |
| partial | 1200 | 0.151631 | 0.326790 |
| refined | 1200 | 0.041547 | 0.613169 |

Paired coarse-vs-refined statistics:

| metric | value |
|---|---:|
| paired samples | 1200 |
| baseline Chamfer | 0.043141 |
| candidate Chamfer | 0.041547 |
| mean CD delta | 0.00159346 |
| median CD delta | 0.00155813 |
| mean CD improvement | 3.6936% |
| bootstrap 95% CI | [3.6073%, 3.7810%] |
| positive / negative / zero samples | 1186 / 14 / 0 |
| positive fraction | 0.9883 |
| mean F-score delta | 0.02631900 |
| paired t-test p-value | 0.0 |
| Wilcoxon p-value | 2.737593353866877e-194 |

## Per-Category Improvement

| category | n | baseline CD | candidate CD | mean CD delta | improvement | positive | negative | zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 02691156 | 150 | 0.029054 | 0.027966 | 0.00108773 | 3.7438% | 149 | 1 | 0 |
| 02933112 | 150 | 0.055852 | 0.053793 | 0.00205938 | 3.6872% | 146 | 4 | 0 |
| 02958343 | 150 | 0.052337 | 0.050392 | 0.00194489 | 3.7161% | 150 | 0 | 0 |
| 03001627 | 150 | 0.045148 | 0.043532 | 0.00161610 | 3.5795% | 149 | 1 | 0 |
| 03636649 | 150 | 0.033420 | 0.032252 | 0.00116818 | 3.4954% | 146 | 4 | 0 |
| 04256520 | 150 | 0.054430 | 0.052374 | 0.00205546 | 3.7763% | 148 | 2 | 0 |
| 04379243 | 150 | 0.039263 | 0.037636 | 0.00162627 | 4.1420% | 149 | 1 | 0 |
| 04530566 | 150 | 0.035623 | 0.034433 | 0.00118968 | 3.3396% | 149 | 1 | 0 |

## Reproduce From A Fresh Clone

Clone with the PoinTr submodule:

```bash
git clone --recurse-submodules https://github.com/Neytron557/pointr-if.git
cd pointr-if
```

If the submodule was not cloned recursively:

```bash
git submodule update --init external/PoinTr
```

Place the gated PCN dataset at:

```text
data/ShapeNetCompletion
```

Expected layout:

```text
data/ShapeNetCompletion/train/partial/<taxonomy>/<model>/00.pcd
data/ShapeNetCompletion/train/complete/<taxonomy>/<model>.pcd
data/ShapeNetCompletion/val/partial/<taxonomy>/<model>/00.pcd
data/ShapeNetCompletion/val/complete/<taxonomy>/<model>.pcd
data/ShapeNetCompletion/test/partial/<taxonomy>/<model>/00.pcd
data/ShapeNetCompletion/test/complete/<taxonomy>/<model>.pcd
```

Create the environment and patch/build the external PoinTr runtime:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 bash scripts/setup_a100_venv.sh
```

Reproduce the selected Run B experiment from scratch:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
NUM_THREADS=8 NUM_WORKERS=8 \
RUN_A=0 RUN_B=1 \
RUN_B_HOURS=5.5 RUN_B_EPOCHS=60 \
RUN_B_BATCH_SIZE=16 RUN_B_EXPORT_BATCH_SIZE=16 \
FORCE_EXPORT=1 \
bash scripts/train_a100_full_improvement.sh outputs/a100_full_improvement_repro
```

Evaluate the tracked best checkpoint:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
NUM_THREADS=8 NUM_WORKERS=8 \
bash scripts/evaluate_run_b_best.sh outputs/reproduce_run_b_best
```

The final paired stats are written to:

```text
outputs/reproduce_run_b_best/test_eval/stats/paired_stats.md
```

## Hardware Notes

The final run used physical `gpu_id=1`, selected with:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
```

On the training server this maps to the 98 GB `NVIDIA Graphics Device`. CPU usage was capped with:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 NUM_THREADS=8 NUM_WORKERS=8
```
