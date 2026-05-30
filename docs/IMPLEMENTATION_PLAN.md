# Implementation Plan for 2-2.5 Days

## Decision

Keep the core proposal idea, but simplify the implementation:

**Original proposal:** modify PoinTr end-to-end and add an Occupancy-Network-style implicit surface module trained with true mesh occupancy labels.

**Revised feasible project:** freeze PoinTr/AdaPoinTr as a coarse completion backbone and train a small **PoinTr-IF refiner** from point-cloud triplets. The refiner predicts:

1. A pseudo occupancy/near-surface field for arbitrary 3D query points.
2. Residual displacements for the coarse PoinTr output points.

This avoids full PoinTr surgery and avoids watertight mesh preprocessing, but still tests the central hypothesis: implicit surface supervision can improve point-only completion.

## Architecture

Input per sample:

- `partial`: observed partial point cloud.
- `coarse`: complete point cloud predicted by PoinTr or AdaPoinTr.
- `gt`: complete target point cloud.

Model:

1. Concatenate `partial` and `coarse` points with a source flag.
2. Encode them with a PointNet-style per-point MLP and global max pooling.
3. For a query point `q`, gather local KNN features from the partial+coarse set.
4. Decode `q` with Fourier features + global feature + local feature.
5. Output occupancy logit and residual displacement.
6. Refined completion is `coarse + predicted_delta(coarse)`.

Loss:

```text
L = lambda_cd * Chamfer(refined, gt)
  + lambda_occ * BCE(occupancy_logits, pseudo_occupancy_labels)
  + lambda_delta * SmoothL1(delta, nearest_gt - coarse)
  + lambda_partial * Chamfer(partial -> refined)
  + lambda_repulsion * local uniformity penalty
```

Pseudo occupancy label:

- Sample query points near GT surface and uniformly in a bounding box.
- Label as positive if the query is within a distance threshold of the complete GT point cloud.
- This is not exact inside/outside occupancy, but is feasible for PCN/ShapeNet point-cloud-only datasets and still teaches a continuous surface-aware field.

## Experiment matrix

Minimum deliverable:

| Experiment | Purpose |
|---|---|
| PoinTr or AdaPoinTr baseline | Establish coarse completion performance. |
| PoinTr-IF local+global | Main method. |
| Global-only refiner | Tests whether local implicit features matter. |
| No-occupancy-loss refiner | Tests whether implicit supervision matters beyond residual denoising. |
| Easy/medium/hard missing splits | Tests professor-facing hypothesis that hard missing regions benefit most. |

Optional if time remains:

- Compare PoinTr backbone vs AdaPoinTr backbone.
- Run category-wise analysis on chair/table/airplane/car.
- Visualize coarse vs refined surfaces.

## Day-by-day schedule

### Day 0 / first evening

- Install official PoinTr and verify pretrained baseline evaluation.
- Install this PoinTr-IF package.
- Run `bash scripts/run_smoke.sh` to verify the refiner code.
- Choose subset: PCN validation or ShapeNet-55 small subset.

### Day 1

- Generate coarse PoinTr/AdaPoinTr predictions for the subset.
- Create CSV manifests with columns `id,partial,coarse,gt`.
- Train `configs/titanx_fast_subset.yaml` for a quick sanity result.
- Run local+global, global-only, and no-occupancy ablations.

### Day 2

- Train the best setting longer using `configs/titanx_pcn_refiner.yaml` if GPU memory allows.
- Evaluate and export visualizations.
- Prepare final report: motivation, feedback response, method, experiments, limitations.

## Risk management

Status update, 2026-05-30: the CUDA/PoinTr setup did run successfully on the TITAN X server. The completed real experiment used Projected ShapeNet55/34, AdaPoinTr `AdaPoinTr_ps55.pth`, 1200 train samples, 150 validation samples, and 150 held-out test samples. Current metrics and artifact paths are recorded in `docs/EXPERIMENT_LOG.md`.

### Risk: official PoinTr setup/CUDA extensions fail

This risk was handled for the completed run by using compatibility fallbacks for the needed AdaPoinTr inference path and running export/training/evaluation on CUDA. Do not present synthetic-only fallback results as final project evidence.

### Risk: 12 GB GPU OOM

Use `configs/titanx_fast_subset.yaml`, reduce `n_query` to 512, and reduce batch size to 2.

### Risk: PoinTr-IF does not beat AdaPoinTr

Compare against original PoinTr as the proposal target, and present AdaPoinTr as a strong newer reference. The contribution can be stated as a lightweight implicit refiner that can attach to either backbone.

### Risk: occupancy loss does not improve CD

This is scientifically acceptable: report the ablation honestly. It may still improve F-score or qualitative continuity. In the final report, explain that point-cloud pseudo occupancy is weaker than true mesh inside/outside supervision and suggest watertight mesh occupancy as future work.
