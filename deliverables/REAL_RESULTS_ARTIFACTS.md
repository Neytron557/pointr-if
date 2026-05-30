# Real Results Artifact Manifest

Archive target:

```text
deliverables/pointr_if_real_projected_shapenet55_results.zip
```

Included:

- Project code, configs, scripts, tools, tests, and docs.
- Real Projected ShapeNet55/34 selected-member JSON files.
- Real AdaPoinTr coarse prediction manifests, export summaries, command logs, and saved coarse `.npy` outputs.
- Full manifest validation reports for train, validation, and test splits.
- Trained PoinTr-IF checkpoint files, training logs, resolved configs, and metrics.
- Validation/test evaluation metrics, per-sample metrics, category metrics, saved refined/anchor predictions, visualizations, and qualitative grid.
- Minimal patched external PoinTr compatibility files and the AdaPoinTr Projected ShapeNet55 config needed to reproduce the run with an existing PoinTr checkout.

Excluded:

- `.venv`, `__pycache__`, pytest cache, and package build caches.
- Raw external ShapeNet55/34 data under `external/PoinTr/data`.
- Official pretrained AdaPoinTr checkpoint files under `external/PoinTr/pretrained`.
- Large downloaded raw archives under `data/downloads`.

The excluded data/checkpoints remain present locally at the paths recorded in `README.md`, `docs/EXPERIMENT_LOG.md`, and command logs.
