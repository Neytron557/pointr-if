# Final Report Draft

## Title

PoinTr-IF: Lightweight Geometry-Aware Refinement for PoinTr-Style Point Cloud Completion

## Abstract

We investigate whether a lightweight implicit/residual refinement module can improve PoinTr-style point-cloud completion. The method keeps AdaPoinTr frozen as a strong pretrained completion backbone, exports coarse completions on real Projected ShapeNet55/34 partial point clouds, and trains PoinTr-IF on real `(partial, coarse, complete)` triplets. The initial naive refiner improved held-out test Chamfer by only 0.0957% and was not statistically significant. A second-pass gated multi-view self-structure refiner, GMV-PoinTr-IF, improves held-out test Chamfer by 0.4107% over AdaPoinTr coarse output with a bootstrap 95% CI of [0.2389%, 0.6037%] and paired t-test p = 1.33e-05. The gain is still modest because AdaPoinTr is a strong baseline, but it is now measurable, repeatable under a second seed, and supported by paired statistics.

## Motivation

PoinTr and AdaPoinTr are strong point-cloud completion baselines, but they still optimize a finite point set. The project idea is to preserve the strength of the transformer completion backbone while adding a small geometry-aware refinement stage that can use local partial observations, coarse completion geometry, and near-surface occupancy-style supervision. This addresses the proposal feedback by making the implicit component concrete and evaluating it on real point-cloud completion data.

## Related Work

Discuss:

- PointNet and PointNet++ as point-cloud encoders.
- Occupancy Networks as implicit function foundations.
- IF-Net and Convolutional Occupancy Networks as implicit representations conditioned on local 3D features.
- PoinTr and AdaPoinTr as completion-specific transformer baselines.
- SnowflakeNet and SeedFormer as point-completion methods focused on local detail.

## Method

Inputs:

```text
partial point cloud P
coarse completion C from AdaPoinTr
complete ground truth G
```

The final GMV-PoinTr-IF model uses a lightweight encoder over the partial and coarse point clouds, source flags to distinguish observed and predicted points, local KNN geometry, six canonical orthographic self-structure views, and a tiny CNN over multi-view occupancy/depth maps. The refiner predicts gated residual offsets for the AdaPoinTr coarse points and trains with Chamfer supervision, conservative residual regularization, partial-preservation loss, and gate regularization.

Model sketch:

```text
concat(P, C, source flag) -> local/global features
six canonical self-structure maps -> tiny CNN -> queried view features
coarse point + geometry features -> gate and residual delta
C_refined = C + sigmoid(gate) * delta(C)
```

Loss:

```text
Chamfer(C_refined, G)
+ nearest-neighbor residual regularization
+ partial preservation loss
+ gate improvement-target BCE
+ gate sparsity and residual L2 penalties
```

The implementation also evaluates a visible-anchor baseline:

```text
FPS(partial union coarse)
```

This tests whether directly preserving observed real partial points improves the final point set.

## Experimental Setup

Dataset:

- Real Projected ShapeNet55/34.
- 1200 train samples, 150 validation samples, 150 held-out test samples.
- 53 train categories, 21 validation categories, 23 test categories.
- Official PCN was attempted first but was not available in the environment without manual gated access, so the experiment used the accepted ShapeNet55/34 path.

Backbone:

- AdaPoinTr pretrained on Projected ShapeNet55.
- Checkpoint: `external/PoinTr/pretrained/AdaPoinTr_ps55.pth`.
- Config: `external/PoinTr/cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml`.

Hardware:

- NVIDIA TITAN X (Pascal), 12 GB VRAM.
- PyTorch `2.7.1+cu126`.
- CUDA was used for export, training, and evaluation.

Training:

- Refiner only; AdaPoinTr was frozen and used for coarse export.
- 40 epochs for the final GMV refiner.
- Batch size 4.
- Seeds 570 and 571.
- Config: `configs/real_projected_shapenet55_gmv_if.yaml`.
- Deterministic evaluation seed: `--eval-seed 200570`.

Metrics:

- Chamfer distance, lower is better.
- F-score at project threshold `0.02`, higher is better.
- Per-sample and per-category metrics are saved.
- Improvement is computed as `100 * (coarse_cd - method_cd) / coarse_cd`.

## Results

Validation results, primary seed 570:

| Method | n | Chamfer | F-score | CD improvement vs coarse |
|---|---:|---:|---:|---:|
| Partial input | 150 | 0.167753 | 0.194933 | -136.57% |
| Visible anchor FPS(partial + coarse) | 150 | 0.073574 | 0.268183 | -3.75% |
| AdaPoinTr coarse | 150 | 0.070912 | 0.303461 | 0.00% |
| GMV-PoinTr-IF refined | 150 | 0.070621 | 0.306258 | 0.4102% |

Held-out test results:

| Method | n | Chamfer | F-score | CD improvement vs coarse | bootstrap 95% CI | paired t p |
|---|---:|---:|---:|---:|---:|---:|
| Partial input | 150 | 0.158159 | 0.175398 | -127.94% | n/a | n/a |
| Visible anchor FPS(partial + coarse) | 150 | 0.072260 | 0.234357 | -4.14% | n/a | n/a |
| AdaPoinTr coarse | 150 | 0.069386 | 0.274315 | 0.00% | n/a | n/a |
| Naive PoinTr-IF seed 570 | 150 | 0.069320 | 0.274787 | 0.0957% | [-0.0457%, 0.2390%] | 0.1937 |
| Gated-local PoinTr-IF seed 570 | 150 | 0.069243 | 0.274894 | 0.2062% | [0.0841%, 0.3301%] | 0.001277 |
| GMV-PoinTr-IF seed 571 | 150 | 0.069219 | 0.276132 | 0.2405% | [0.0832%, 0.4234%] | 0.005839 |
| GMV-PoinTr-IF seed 570 | 150 | 0.069101 | 0.277758 | 0.4107% | [0.2389%, 0.6037%] | 1.33e-05 |

Qualitative examples:

- `outputs/real_projected_shapenet55_gmv_if/test_eval/visualizations/`
- `outputs/real_projected_shapenet55_gmv_if/test_eval/ranked_qualitative.png`

Complete result files:

- `outputs/real_projected_shapenet55_gmv_if/val_eval/metrics.json`
- `outputs/real_projected_shapenet55_gmv_if/val_eval/stats/paired_stats.md`
- `outputs/real_projected_shapenet55_gmv_if/test_eval/metrics.json`
- `outputs/real_projected_shapenet55_gmv_if/test_eval/stats/paired_stats.md`
- `outputs/real_projected_shapenet55_gmv_if/test_eval/per_sample_metrics.csv`
- `outputs/real_projected_shapenet55_gmv_if/test_eval/category_metrics.csv`
- `docs/HARD_SECOND_PASS_REPORT.md`

## Discussion

The naive learned PoinTr-IF refiner gave a small, non-significant improvement over AdaPoinTr. The gated-local and GMV variants are stronger: GMV seed 570 improves held-out Chamfer by 0.4107% and seed 571 by 0.2405% under the same deterministic evaluation seed. This is a realistic outcome: AdaPoinTr is already a strong pretrained backbone, and the refiner is trained as a lightweight post-processing model rather than integrated end-to-end into the transformer. The result supports the idea that local implicit/residual refinement can improve a strong coarse completion, but it does not justify a claim of a large benchmark jump.

The visible-anchor baseline did not improve Chamfer on this subset. This is useful negative evidence: directly mixing observed partial points with AdaPoinTr coarse output can hurt global point-set balance even though the observed points are real surface samples. The final presentation should show this as an ablation, not as a successful component.

Fourth-pass status: the SEED/candidate-bank post-processing path is no longer treated as deployable because its oracle headroom was too small. The follow-up implementation is MVC-PoinTr-IF, which fine-tunes AdaPoinTr directly with train-only multi-view consistency. Its success gate is held-out test CD <= 0.04219 under the audited FPS-4096 protocol; no fourth-pass result should be reported as successful until a validation-selected checkpoint reaches that gate. See `docs/FOURTH_PASS_MV_CONSISTENCY_REPORT.md`.

The local projected data cap prevented scaling beyond 1200/150/150 in this environment, despite larger official split metadata. Future work should run on the full official projected set or PCN after gated download access, use mesh-derived occupancy labels where available, and integrate PoinTr/AdaPoinTr features directly rather than refining only exported point sets.

## Conclusion

The project keeps the original proposal idea while making it feasible: AdaPoinTr supplies a strong complete coarse point cloud, and GMV-PoinTr-IF provides a small geometry-aware gated refinement stage. On real Projected ShapeNet55/34 data, the refined output improves Chamfer and F-score over AdaPoinTr coarse output with paired statistical support. The contribution is best framed as a measured lightweight refinement result with honest limitations, not as a replacement for the PoinTr backbone.
