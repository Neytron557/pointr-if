import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from pointr_if.io import load_point_cloud, save_point_cloud


def _write_cloud(path, offset=0.0):
    points = np.array(
        [
            [0.0 + offset, 0.0, 0.0],
            [1.0 + offset, 0.0, 0.0],
            [0.0 + offset, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    save_point_cloud(path, points)
    return points


def test_validate_triplet_manifest_cli(tmp_path):
    partial = tmp_path / "partial.npy"
    coarse = tmp_path / "coarse.npy"
    gt = tmp_path / "gt.npy"
    _write_cloud(partial)
    _write_cloud(coarse, 0.1)
    _write_cloud(gt, 0.2)
    manifest = tmp_path / "triplets.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "partial", "coarse", "gt"])
        writer.writeheader()
        writer.writerow({"id": "sample", "partial": partial.name, "coarse": coarse.name, "gt": gt.name})

    out_json = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/validate_triplet_manifest.py",
            str(manifest),
            "--max-samples",
            "1",
            "--json-out",
            str(out_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["n_rows"] == 1
    assert summary["checked_rows"] == 1
    assert summary["samples"][0]["partial"]["shape"] == [3, 3]
    assert json.loads(out_json.read_text(encoding="utf-8"))["n_rows"] == 1


def test_validate_triplet_manifest_accepts_real_schema(tmp_path):
    partial = tmp_path / "partial.npy"
    coarse = tmp_path / "coarse.npy"
    gt = tmp_path / "gt.npy"
    _write_cloud(partial)
    _write_cloud(coarse, 0.1)
    _write_cloud(gt, 0.2)
    manifest = tmp_path / "real_triplets.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "category", "partial_path", "coarse_path", "gt_path", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample",
                "category": "chair",
                "partial_path": partial.name,
                "coarse_path": coarse.name,
                "gt_path": gt.name,
                "split": "val",
            }
        )

    result = subprocess.run(
        [
            sys.executable,
            "tools/validate_triplet_manifest.py",
            str(manifest),
            "--max-samples",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    assert summary["splits"] == {"val": 1}
    assert summary["categories"] == {"chair": 1}


def test_convert_pointr_predictions_matches_and_copies(tmp_path):
    partial_dir = tmp_path / "partial"
    coarse_dir = tmp_path / "coarse"
    gt_dir = tmp_path / "gt"
    for directory in (partial_dir, coarse_dir, gt_dir):
        directory.mkdir()
    _write_cloud(partial_dir / "chair_000_partial.xyz")
    _write_cloud(coarse_dir / "chair_000_pred.npy", 0.1)
    _write_cloud(gt_dir / "chair_000_complete.pcd", 0.2)

    manifest = tmp_path / "manifest.csv"
    converted = tmp_path / "converted"
    subprocess.run(
        [
            sys.executable,
            "tools/convert_pointr_predictions.py",
            "--partial-dir",
            str(partial_dir),
            "--coarse-dir",
            str(coarse_dir),
            "--gt-dir",
            str(gt_dir),
            "--output-manifest",
            str(manifest),
            "--split",
            "val",
            "--converted-root",
            str(converted),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with manifest.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["id"] == "val_chair_000"
    assert load_point_cloud(rows[0]["partial"]).shape == (3, 3)
    assert rows[0]["partial"].endswith(".npy")


def test_load_point_cloud_h5_uses_common_keys(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "cloud.h5"
    points = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    with h5py.File(path, "w") as f:
        f.create_dataset("complete", data=points)

    loaded = load_point_cloud(path)
    np.testing.assert_allclose(loaded, points[0])


def test_load_point_cloud_npz_accepts_batched_arrays(tmp_path):
    path = tmp_path / "batched.npz"
    points = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    np.savez_compressed(path, points=points)

    loaded = load_point_cloud(path)
    np.testing.assert_allclose(loaded, points[0])
