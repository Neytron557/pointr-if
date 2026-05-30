# Response to Professor Feedback

## What the original proposal got right

The proposal has a coherent course-aligned idea: **PoinTr handles discrete point completion; an implicit module adds continuous surface awareness**. This fits the course topic on 3D shape representation and implicit functions and gives a natural experimental question: does implicit refinement improve point-cloud completion, especially in hard missing-region cases?

The presentation also had clear figures, a sensible dataset list, and a comparison plan: PoinTr baseline vs PoinTr + implicit head, global-only vs global+local conditioning, and different missing percentages.

## Main weaknesses from the feedback

### C1: Prior work is incomplete

The proposal cited older foundations such as PointNet and Occupancy Networks, but it did not sufficiently cover **implicit point-cloud reconstruction/completion** methods. Add these to the written report and presentation:

- **IF-Net / Implicit Functions in Feature Space**: relevant because it predicts implicit values at query points using global and local features from imperfect 3D inputs, including sparse or incomplete point clouds.
- **Convolutional Occupancy Networks**: relevant because it fixes a limitation of pure global-code implicit decoders by adding local/structured feature conditioning.
- **AdaPoinTr**: relevant because it is the newer official extension of PoinTr with adaptive denoising queries. It should be presented as a strong reference/backbone option, not ignored.
- **SnowflakeNet / SeedFormer**: relevant completion baselines focused on local detail recovery.

### C2: Why implicit representation helps completion

The original proposal said implicit representations are smoother, but did not make the advantage precise. The revised explanation should say:

- PoinTr predicts a finite set of points. This is directly measurable by Chamfer distance, but it has no explicit notion of whether nearby 3D query locations are on or off the object surface.
- An implicit head can supervise arbitrary 3D queries around the completed object, giving a denser surface-consistency signal than point-to-point losses alone.
- Local implicit conditioning can regularize holes, thin structures, and hard missing regions because the model learns a continuous near-surface field around both observed partial points and generated coarse points.
- The final evaluation can still be point-cloud based: the implicit module predicts residual corrections for coarse PoinTr points and is evaluated with Chamfer distance/F-score.

### C3: Why use PoinTr instead of PointNet++, PointNeXt, or Point Transformer?

The clean answer is:

- PointNet++ and PointNeXt are primarily point-cloud **understanding encoders** for classification/segmentation-style tasks; they are not complete point-cloud completion systems by themselves.
- Point Transformer is a general self-attention point-cloud architecture; using it would require building the completion decoder, training recipe, and benchmarks from scratch.
- PoinTr is completion-specific: it already has an encoder-decoder generation framework, public pretrained checkpoints, official PCN/ShapeNet/KITTI evaluation scripts, and a newer AdaPoinTr branch in the same codebase.
- Because the class project must finish in 2-2.5 days, PoinTr/AdaPoinTr is the best backbone to preserve proposal consistency and reduce implementation risk.

## Recommended project framing

Do **not** claim that we fully implemented Occupancy Networks on watertight meshes. Instead, claim:

> We implemented a lightweight implicit surface refinement module for PoinTr-style point-cloud completion. The module is trained from point-cloud-only supervision by sampling 3D query points around the complete target point cloud and assigning pseudo near-surface occupancy labels. The same implicit decoder also predicts residual corrections for PoinTr's coarse output, so the method remains directly evaluable with Chamfer distance and F-score.

This keeps the core idea while avoiding the largest timeline risk: preparing watertight ShapeNet meshes and exact inside/outside occupancy labels.

## What to say if results are mixed

A safe interpretation:

- If PoinTr-IF improves CD/F-score on hard missing cases, emphasize implicit local surface supervision as the likely reason.
- If overall CD improvement is small, report that the implicit head acts as a regularizer and improves some hard categories/qualitative surfaces but is sensitive to the quality of coarse PoinTr predictions.
- If it fails to beat AdaPoinTr, say AdaPoinTr is a stronger newer backbone, while the project contribution is the implicit refinement layer that can be attached to either PoinTr or AdaPoinTr.
