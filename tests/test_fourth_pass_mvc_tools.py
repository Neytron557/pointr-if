from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path("tools").resolve()))

from build_multiview_groups import build_groups_for_manifest
from build_pcn_artifacts import build_pcn_artifacts
from generate_train_projected_views import project_visible_points
from train_adapointr_multiview_consistency import build_optimizer, set_trainable_modules
from pointr_if.io import load_point_cloud, save_point_cloud


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "category", "partial_path", "coarse_path", "gt_path", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_multiview_groups_groups_by_gt_path(tmp_path: Path) -> None:
    gt = tmp_path / "gt.npy"
    partial_a = tmp_path / "partial_a.npy"
    partial_b = tmp_path / "partial_b.npy"
    coarse = tmp_path / "coarse.npy"
    for path, offset in ((gt, 0.0), (partial_a, 0.1), (partial_b, 0.2), (coarse, 0.3)):
        save_point_cloud(path, np.full((4, 3), offset, dtype=np.float32))
    manifest = tmp_path / "train.csv"
    _write_manifest(
        manifest,
        [
            {
                "sample_id": "train_03001627_model_view0",
                "category": "03001627",
                "partial_path": str(partial_a),
                "coarse_path": str(coarse),
                "gt_path": str(gt),
                "split": "train",
            },
            {
                "sample_id": "train_03001627_model_view1",
                "category": "03001627",
                "partial_path": str(partial_b),
                "coarse_path": str(coarse),
                "gt_path": str(gt),
                "split": "train",
            },
        ],
    )

    groups, audit = build_groups_for_manifest(manifest)

    assert audit["n_rows"] == 2
    assert audit["n_groups"] == 1
    assert audit["views_per_group"]["max"] == 2
    assert groups[0]["group_id"] == "03001627-model"
    assert [member["sample_id"] for member in groups[0]["members"]] == [
        "train_03001627_model_view0",
        "train_03001627_model_view1",
    ]


def test_project_visible_points_is_deterministic_and_finite() -> None:
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)), axis=-1)
    points = grid.reshape(-1, 3).astype(np.float32)

    first = project_visible_points(points, view_direction=np.array([1.0, 0.5, 0.25]), image_size=8, n_points=32)
    second = project_visible_points(points, view_direction=np.array([1.0, 0.5, 0.25]), image_size=8, n_points=32)

    assert first.shape == (32, 3)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)


def test_build_pcn_artifacts_writes_groups_and_source_manifests(tmp_path: Path) -> None:
    root = tmp_path / "ShapeNetCompletion"
    category = "03001627"
    model = "abc"
    for split, views in (("train", 8), ("val", 1), ("test", 1)):
        complete_dir = root / split / "complete" / category
        partial_dir = root / split / "partial" / category / model
        complete_dir.mkdir(parents=True)
        partial_dir.mkdir(parents=True)
        (complete_dir / f"{model}.pcd").write_text(
            "\n".join(
                [
                    "# .PCD v0.7",
                    "VERSION 0.7",
                    "FIELDS x y z",
                    "SIZE 4 4 4",
                    "TYPE F F F",
                    "COUNT 1 1 1",
                    "WIDTH 1",
                    "HEIGHT 1",
                    "POINTS 1",
                    "DATA ascii",
                    "0 0 0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        for view in range(views):
            (partial_dir / f"{view:02d}.pcd").write_text((complete_dir / f"{model}.pcd").read_text(), encoding="utf-8")
    category_json = tmp_path / "PCN.json"
    category_json.write_text(
        json.dumps([{"taxonomy_id": category, "taxonomy_name": "chair", "train": [model], "val": [model], "test": [model]}]),
        encoding="utf-8",
    )

    summary = build_pcn_artifacts(pcn_root=root, category_json=category_json, out_dir=tmp_path / "out")

    assert summary["splits"]["train"]["n_groups"] == 1
    assert summary["splits"]["train"]["views_per_group"] == 8
    assert Path(summary["splits"]["val"]["source_manifest"]).exists()


class _DummyAda(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.coarse_pred = torch.nn.Linear(1, 1)
        self.base_model.mlp_query = torch.nn.Linear(1, 1)
        self.base_model.query_ranking = torch.nn.Linear(1, 1)
        self.base_model.decoder = torch.nn.Module()
        self.base_model.decoder.blocks = torch.nn.Module()
        self.base_model.decoder.blocks.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
        self.base_model.encoder = torch.nn.Module()
        self.base_model.encoder.blocks = torch.nn.Module()
        self.base_model.encoder.blocks.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
        self.increase_dim = torch.nn.Linear(1, 1)
        self.reduce_map = torch.nn.Linear(1, 1)
        self.decode_head = torch.nn.Linear(1, 1)


def test_set_trainable_modules_freezes_encoder_by_default() -> None:
    model = _DummyAda()

    set_trainable_modules(model, unfreeze="decoder_only", unfreeze_last_encoder_blocks=0)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    assert "base_model.coarse_pred.weight" in trainable
    assert "base_model.decoder.blocks.blocks.0.weight" in trainable
    assert "base_model.encoder.blocks.blocks.1.weight" not in trainable

    set_trainable_modules(model, unfreeze="decoder_plus_last_encoder", unfreeze_last_encoder_blocks=1)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}

    assert "base_model.encoder.blocks.blocks.1.weight" in trainable
    assert "base_model.encoder.blocks.blocks.0.weight" not in trainable


def test_build_optimizer_rejects_unfrozen_encoder_without_lr() -> None:
    model = _DummyAda()
    set_trainable_modules(model, unfreeze="decoder_plus_last_encoder", unfreeze_last_encoder_blocks=1)

    with pytest.raises(RuntimeError, match="lr_encoder"):
        build_optimizer(model, lr_decoder=1e-5, lr_encoder=0.0, weight_decay=0.0)
