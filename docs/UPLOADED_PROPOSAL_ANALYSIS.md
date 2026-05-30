# Uploaded Proposal Analysis

## Materials reviewed

The uploaded archive contained:

- `initial_proposal.txt`
- `feeback.txt`
- `PoinTr-IF_ Geometry-Aware Point Cloud Completion with Implicit Surface Refinement.pdf`
- `PoinTr-IF_ Geometry-Aware Point Cloud Completion with Implicit Surface Refinement.pptx`

## Proposal summary

The initial project idea was to use PoinTr as the base point-cloud completion architecture and add an implicit surface refinement module inspired by Occupancy Networks. PoinTr would predict missing points, then the implicit module would learn continuous surface/inside-outside structure to refine the completed shape. The proposed datasets were ShapeNet, PCN, and possibly KITTI. The proposal also listed a fallback: improve PoinTr through more realistic partial-view training if implicit refinement proved too difficult.

## Strengths

- The topic fits CS570's implicit-functions theme.
- The project has a clear baseline and proposed extension.
- The slides communicate the coarse-to-refined pipeline visually.
- The proposed comparison plan already includes PoinTr baseline vs implicit head, global vs local conditioning, and missing-rate experiments.

## Weaknesses

- The implementation plan is too ambitious for 2-2.5 days if interpreted as true Occupancy Network training with watertight mesh occupancy supervision.
- The proposal does not explain how occupancy labels will be generated from PCN/ShapeNet point-cloud data.
- It does not sufficiently cite implicit point-cloud reconstruction/completion methods such as IF-Net and Convolutional Occupancy Networks.
- It does not justify why PoinTr remains a good backbone when PointNet++, PointNeXt, Point Transformer, and AdaPoinTr exist.
- The slides promise inside/outside prediction, but the available datasets may only provide point clouds, not meshes; this mismatch needs to be addressed.

## Recommended revision

Keep the PoinTr-IF idea, but revise the contribution as follows:

> We add a lightweight implicit surface refinement head to PoinTr/AdaPoinTr. Instead of requiring watertight mesh occupancy labels, we train the implicit head with pseudo near-surface occupancy labels sampled around the complete target point cloud. The same decoder predicts residual corrections for PoinTr's coarse output points, so we can evaluate the final output with Chamfer distance and F-score.

This is the best trade-off between proposal consistency, professor feedback, and the 2-2.5 day deadline.

## What changed from the initial proposal

| Initial proposal | Revised feasible implementation |
|---|---|
| Full Occupancy-Network-style surface module | Lightweight implicit near-surface head |
| True inside/outside labels from meshes | Pseudo occupancy labels from complete point clouds |
| Modify PoinTr end-to-end | Freeze PoinTr/AdaPoinTr and train post-refiner |
| ShapeNet/PCN/KITTI full experiments | PCN/ShapeNet subset first; KITTI optional |
| Main risk hidden | Explicit fallback and ablation plan |

## Final recommendation

Do not abandon the idea. Use the implementation in this package as the main project. It directly addresses the professor's feedback while staying realistic for the deadline.
