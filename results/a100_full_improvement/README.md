# A100 PCN Results

These are the compact tracked artifacts for the final PCN experiment.

## Run B, Deployable Result

Path: `run_b_feature_refiner/`

- Model: backbone-feature-conditioned `feature_gmv` refiner.
- Checkpoint: `checkpoints/run_b/best.pt`.
- Best validation epoch: `20`.
- Held-out PCN test samples: `1200`.
- Baseline: frozen AdaPoinTr PCN coarse completion.
- Candidate: Run B refined completion.

| method | n | chamfer | fscore |
|---|---:|---:|---:|
| anchor | 1200 | 0.042830 | 0.596364 |
| coarse | 1200 | 0.043141 | 0.586850 |
| partial | 1200 | 0.151631 | 0.326790 |
| refined | 1200 | 0.041547 | 0.613169 |

Paired test result:

- Mean CD improvement: `3.6936%`.
- Bootstrap 95% CI: `[3.6073%, 3.7810%]`.
- Positive / negative / zero samples: `1186 / 14 / 0`.
- Mean F-score delta: `0.02631900`.
- Paired t-test p-value: `0.0`.
- Wilcoxon p-value: `2.737593353866877e-194`.

Important files:

- `run_b_feature_refiner/metrics.csv`: train/validation metrics.
- `run_b_feature_refiner/summary.json`: training summary.
- `run_b_feature_refiner/test_eval/per_sample_metrics.csv`: held-out test metrics for all methods.
- `run_b_feature_refiner/test_eval/stats/paired_stats.md`: final paired statistical report.
- `run_b_feature_refiner/test_eval/stats/per_sample_delta.csv`: paired per-sample deltas.
- `run_b_feature_refiner/test_eval/stats/per_category_improvement.csv`: per-category improvements.

## Run A Failure

Path: `run_a_pcn_mvc/`

Run A fine-tuned AdaPoinTr with multi-view consistency. It failed the selection criterion and degraded held-out performance:

- Baseline test CD: `0.040271`.
- Candidate test CD: `0.048130`.
- Mean CD improvement: `-19.5151%`.
- Positive / negative / zero samples: `104 / 1096 / 0`.

No Run A checkpoints are tracked.
