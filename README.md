# PoinTr-IF: AdaPoinTr Feature-Conditioned Refinement

This repository implements and evaluates lightweight refiners for PoinTr/AdaPoinTr point-cloud completion. The final selected method is **Run B**, a backbone-feature-conditioned gated multi-view refiner trained on PCN / ShapeNetCompletion.

## Final Result

- Dataset: PCN / ShapeNetCompletion.
- Backbone: AdaPoinTr PCN checkpoint.
- Selected model: `feature_gmv`, conditioned on AdaPoinTr decoder features.
- Checkpoint: `checkpoints/run_b/best.pt`.
- Best validation epoch: `20`.
- Held-out test samples: `1200`.

| method | n | chamfer | fscore |
|---|---:|---:|---:|
| anchor | 1200 | 0.042830 | 0.596364 |
| coarse | 1200 | 0.043141 | 0.586850 |
| partial | 1200 | 0.151631 | 0.326790 |
| refined | 1200 | 0.041547 | 0.613169 |

Against frozen AdaPoinTr coarse completions, Run B improves held-out Chamfer by **3.6936%** with bootstrap 95% CI **[3.6073%, 3.7810%]**. It improves 1186 / 1200 test samples and gains `0.026319` mean F-score.

Run A, which fine-tuned AdaPoinTr with multi-view consistency, was implemented but failed: held-out CD degraded from `0.040271` to `0.048130` (`-19.5151%`). Its checkpoints are intentionally not tracked.

## Tracked Artifacts

- `checkpoints/run_b/best.pt`: deployable Run B checkpoint.
- `checkpoints/run_b/resolved_config.yaml`: config embedded in that checkpoint.
- `results/a100_full_improvement/run_b_feature_refiner/`: train metrics, held-out test metrics, paired stats, per-sample deltas, and per-category improvements.
- `results/a100_full_improvement/run_a_pcn_mvc/`: compact failure metrics for Run A, without checkpoints.
- `docs/A100_PCN_RESULTS.md`: final result report and reproduction instructions.

Large generated data are not tracked: exported decoder features, per-sample predicted point clouds, full `outputs/`, and the gated PCN dataset.

## Reproduce

Clone with the PoinTr submodule:

```bash
git clone --recurse-submodules https://github.com/Neytron557/pointr-if.git
cd pointr-if
```

Place the PCN gated dataset at:

```text
data/ShapeNetCompletion
```

Create the environment and build the required PoinTr runtime extension:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 bash scripts/setup_a100_venv.sh
```

Reproduce the selected Run B training and evaluation:

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

The final stats will be written to `outputs/reproduce_run_b_best/test_eval/stats/paired_stats.md`.

## Important Entrypoints

- `scripts/setup_a100_venv.sh`: creates `.venv`, initializes/patches `external/PoinTr`, installs dependencies, and builds Chamfer CUDA.
- `scripts/build_pcn_artifacts.sh`: builds PCN source manifests and multi-view group JSONs from `data/ShapeNetCompletion`.
- `tools/export_adapointr_features.py`: exports AdaPoinTr dense completions and aligned decoder features.
- `scripts/train_a100_full_improvement.sh`: runs the A100 PCN experiment pipeline.
- `scripts/evaluate_run_b_best.sh`: evaluates the tracked best Run B checkpoint.

## Development Checks

```bash
source .venv/bin/activate
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m compileall -q src tools tests
```

## Documentation Map

- `docs/A100_PCN_RESULTS.md`: final PCN/A100 report and reproduction runbook.
- `docs/MAIN_MODEL_IMPROVEMENT_PLAN.md` and `docs/MODEL_IMPROVEMENT_PLAN.md`: implementation plan that led to Run A and Run B.
- `docs/FOURTH_PASS_MV_CONSISTENCY_REPORT.md`: prior MVC work and Run A lineage.
- `docs/HARD_SECOND_PASS_REPORT.md`: older Projected ShapeNet55/34 result.
- `docs/EXPERIMENT_LOG.md`: historical experiment notes.
