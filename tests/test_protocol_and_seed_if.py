from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pointr_if.datasets import ManifestTripletDataset
from pointr_if.io import save_point_cloud
from pointr_if.models import SeedPointrIFRefiner, build_model_from_config


def _line_cloud(n: int, offset: float = 0.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, dtype=np.float32) + offset
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)


def _write_triplet_manifest(root: Path, n: int = 8) -> Path:
    partial = root / "partial.npy"
    coarse = root / "coarse.npy"
    gt = root / "gt.npy"
    save_point_cloud(partial, _line_cloud(n, 0.0))
    save_point_cloud(coarse, _line_cloud(n, 0.1))
    save_point_cloud(gt, _line_cloud(n, 0.2))
    manifest = root / "triplets.csv"
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
                "split": "test",
            }
        )
    return manifest


def test_manifest_dataset_supports_fps_and_none_resampling(tmp_path: Path) -> None:
    manifest = _write_triplet_manifest(tmp_path, n=8)

    fps = ManifestTripletDataset(
        manifest,
        n_partial=4,
        n_coarse=4,
        n_gt=4,
        normalize=False,
        resample_mode="fps",
    )[0]
    none = ManifestTripletDataset(
        manifest,
        n_partial=None,
        n_coarse=None,
        n_gt=None,
        normalize=False,
        resample_mode="none",
    )[0]

    assert fps["partial"].shape == (4, 3)
    assert none["coarse"].shape == (8, 3)
    assert torch.allclose(fps["partial"][0], torch.tensor([0.0, 0.0, 0.0]))
    assert (fps["partial"] == torch.tensor([1.0, 0.0, 0.0])).all(dim=1).any()


def test_evaluate_manifest_records_protocol_overrides(tmp_path: Path) -> None:
    from pointr_if.evaluate import evaluate_manifest

    manifest = _write_triplet_manifest(tmp_path, n=8)
    cfg = {
        "dataset": {"n_partial": 4, "n_coarse": 4, "n_gt": 4, "normalize": False},
        "model": {"hidden_dim": 8, "point_feature_dim": 6, "global_dim": 10, "fourier_bands": 1, "k": 2},
        "train": {"device": "cpu", "num_workers": 0, "batch_size": 1},
    }
    model = build_model_from_config(cfg)
    ckpt = tmp_path / "model.pt"
    torch.save({"cfg": cfg, "model": model.state_dict(), "epoch": 1}, ckpt)

    evaluate_manifest(
        manifest=manifest,
        checkpoint=ckpt,
        out_dir=tmp_path / "eval",
        batch_size=1,
        num_workers=0,
        device="cpu",
        eval_seed=77,
        resample_mode="fps",
        n_partial=4,
        n_coarse=4,
        n_gt=4,
        n_output=4,
    )

    command_log = json.loads((tmp_path / "eval" / "command_log.txt").read_text(encoding="utf-8"))
    assert command_log["resample_mode"] == "fps"
    assert command_log["n_output"] == 4


def test_seed_refiner_forward_outputs_dense_cloud() -> None:
    model = SeedPointrIFRefiner(
        hidden_dim=16,
        global_dim=20,
        source_embed_dim=4,
        fourier_bands=1,
        k=3,
        max_sources=6,
        candidate_points=5,
        output_points=12,
        expansion_factor=3,
        delta_scale=0.02,
    )
    partial = torch.rand(2, 7, 3)
    coarse = torch.rand(2, 5, 3)
    candidates = torch.rand(2, 3, 5, 3)
    source_ids = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)

    out = model(partial=partial, coarse=coarse, candidates=candidates, source_ids=source_ids)

    assert out["refined"].shape == (2, 12, 3)
    assert out["point_confidence"].shape == (2, 15)
    assert out["source_logits"].shape[:2] == (2, 15)
    assert out["residual_delta"].shape == (2, 15, 3)
    assert torch.isfinite(out["refined"]).all()


def test_candidate_manifest_and_oracle_cli(tmp_path: Path) -> None:
    manifest = _write_triplet_manifest(tmp_path, n=6)
    cand_root = tmp_path / "candidates"
    id_dir = cand_root / "identity"
    good_dir = cand_root / "good"
    id_dir.mkdir(parents=True)
    good_dir.mkdir(parents=True)
    save_point_cloud(id_dir / "sample-1.npy", _line_cloud(6, 0.1))
    save_point_cloud(good_dir / "sample-1.npy", _line_cloud(6, 0.2))

    candidate_manifest = tmp_path / "candidate_manifest.csv"
    subprocess.run(
        [
            sys.executable,
            "tools/build_candidate_manifest.py",
            "--triplet-manifest",
            str(manifest),
            "--source",
            f"identity={id_dir}",
            "--source",
            f"good={good_dir}",
            "--out-manifest",
            str(candidate_manifest),
            "--out-root",
            str(tmp_path / "bank"),
            "--candidate-points",
            "6",
            "--resample-mode",
            "none",
            "--no-include-symmetry",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    oracle_dir = tmp_path / "oracle"
    subprocess.run(
        [
            sys.executable,
            "tools/evaluate_candidate_bank_oracle.py",
            "--candidate-manifest",
            str(candidate_manifest),
            "--out-dir",
            str(oracle_dir),
            "--n-gt",
            "6",
            "--resample-mode",
            "none",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((oracle_dir / "oracle_summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["methods"]["good"]["chamfer"] == pytest.approx(0.0, abs=1e-7)
    assert summary["oracle"]["sample_oracle"]["chamfer"] == pytest.approx(0.0, abs=1e-7)
