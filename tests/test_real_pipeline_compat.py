from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from pointr_if.io import read_manifest, save_point_cloud


def test_real_manifest_columns_are_normalized(tmp_path):
    partial = tmp_path / "partial.npy"
    coarse = tmp_path / "coarse.npy"
    gt = tmp_path / "gt.npy"
    points = np.zeros((4, 3), dtype=np.float32)
    for path in (partial, coarse, gt):
        save_point_cloud(path, points)

    manifest = tmp_path / "triplets.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "category", "partial_path", "coarse_path", "gt_path", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample-1",
                "category": "chair",
                "partial_path": partial.name,
                "coarse_path": coarse.name,
                "gt_path": gt.name,
                "split": "val",
            }
        )

    rows = read_manifest(manifest)

    assert rows[0]["id"] == "sample-1"
    assert rows[0]["sample_id"] == "sample-1"
    assert rows[0]["category"] == "chair"
    assert Path(rows[0]["partial"]).is_absolute()
    assert rows[0]["partial"] == rows[0]["partial_path"]
    assert rows[0]["coarse"] == rows[0]["coarse_path"]
    assert rows[0]["gt"] == rows[0]["gt_path"]


def test_pointnet2_fallback_fps_and_gather_shapes():
    module_path = Path("external/PoinTr/pointnet2_ops/pointnet2_utils.py")
    spec = importlib.util.spec_from_file_location("pointr_pointnet2_fallback", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    xyz = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]],
        dtype=torch.float32,
    )
    idx = module.furthest_point_sample(xyz, 2)
    gathered = module.gather_operation(xyz.transpose(1, 2).contiguous(), idx)

    assert idx.shape == (1, 2)
    assert gathered.shape == (1, 3, 2)
    assert torch.equal(idx[:, :1], torch.zeros((1, 1), dtype=torch.long))


def test_projected_shapenet_selected_rows_map_paths(tmp_path):
    module_path = Path("tools/export_pointr_predictions.py")
    spec = importlib.util.spec_from_file_location("export_pointr_predictions", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            [
                {
                    "taxonomy_id": "03001627",
                    "model_id": "abc",
                    "gt_member": "shapenet_pc/03001627-abc.npy",
                    "partial_member": "pcd/03001627/abc/models/0.pcd",
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = module._rows_from_selected_json(selected, tmp_path / "data", "val")

    assert rows[0].sample_id == "val_03001627_abc_view0"
    assert rows[0].category == "03001627"
    assert rows[0].partial_path == tmp_path / "data/pcd/03001627/abc/models/0.pcd"
    assert rows[0].gt_path == tmp_path / "data/shapenet_pc/03001627-abc.npy"
