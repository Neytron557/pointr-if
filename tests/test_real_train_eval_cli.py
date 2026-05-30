from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from pointr_if.datasets import ManifestTripletDataset
from pointr_if.io import save_point_cloud
from pointr_if.models import ImplicitSurfaceRefiner


def _write_real_manifest(path: Path) -> None:
    points = {
        "partial": np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32),
        "coarse": np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]],
            dtype=np.float32,
        ),
        "gt": np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]],
            dtype=np.float32,
        ),
    }
    for name, arr in points.items():
        save_point_cloud(path.parent / f"{name}.npy", arr)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "category", "partial_path", "coarse_path", "gt_path", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample-1",
                "category": "03001627",
                "partial_path": "partial.npy",
                "coarse_path": "coarse.npy",
                "gt_path": "gt.npy",
                "split": "test",
            }
        )


def test_manifest_dataset_preserves_real_metadata(tmp_path):
    manifest = tmp_path / "triplets.csv"
    _write_real_manifest(manifest)

    dset = ManifestTripletDataset(manifest, n_partial=3, n_coarse=4, n_gt=4, normalize=False)
    sample = dset[0]

    assert sample["id"] == "sample-1"
    assert sample["sample_id"] == "sample-1"
    assert sample["category"] == "03001627"
    assert sample["split"] == "test"


def test_train_cli_maps_fixed_prompt_args():
    from pointr_if.train import apply_cli_overrides

    cfg = {"dataset": {"type": "synthetic"}, "train": {}}
    updated = apply_cli_overrides(
        cfg,
        train_manifest="train.csv",
        val_manifest="val.csv",
        epochs=20,
        batch_size=4,
        num_workers=2,
        amp=True,
        seed=570,
    )

    assert updated["seed"] == 570
    assert updated["dataset"]["type"] == "manifest"
    assert updated["dataset"]["manifest"] == "train.csv"
    assert updated["dataset"]["val_manifest"] == "val.csv"
    assert updated["train"]["epochs"] == 20
    assert updated["train"]["batch_size"] == 4
    assert updated["train"]["val_batch_size"] == 4
    assert updated["train"]["num_workers"] == 2
    assert updated["train"]["amp"] is True


def test_evaluate_manifest_writes_real_metric_outputs(tmp_path):
    from pointr_if.evaluate import evaluate_manifest

    manifest = tmp_path / "triplets.csv"
    _write_real_manifest(manifest)
    cfg = {
        "dataset": {"n_partial": 3, "n_coarse": 4, "n_gt": 4, "normalize": False},
        "model": {"hidden_dim": 8, "point_feature_dim": 6, "global_dim": 10, "fourier_bands": 1, "k": 2},
        "train": {"device": "cpu", "num_workers": 0, "batch_size": 1},
        "eval": {"fscore_threshold": 0.05},
    }
    model = ImplicitSurfaceRefiner(hidden_dim=8, point_feature_dim=6, global_dim=10, fourier_bands=1, k=2)
    ckpt = tmp_path / "model.pt"
    torch.save({"cfg": cfg, "model": model.state_dict(), "epoch": 1}, ckpt)

    summary = evaluate_manifest(
        manifest=manifest,
        checkpoint=ckpt,
        out_dir=tmp_path / "eval",
        batch_size=1,
        num_workers=0,
        device="cpu",
        save_predictions=True,
        save_visualizations=False,
        eval_seed=123,
    )

    assert summary["overall"]["n_samples"] == 1
    assert {"partial", "coarse", "anchor", "refined"}.issubset(summary["overall"]["methods"])
    assert (tmp_path / "eval" / "per_sample_metrics.csv").exists()
    assert (tmp_path / "eval" / "category_metrics.csv").exists()
    assert (tmp_path / "eval" / "summary_table.md").exists()
    assert (tmp_path / "eval" / "predictions" / "sample-1_refined.npy").exists()
    assert '"eval_seed": 123' in (tmp_path / "eval" / "command_log.txt").read_text(encoding="utf-8")
