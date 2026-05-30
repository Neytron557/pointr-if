#!/usr/bin/env python
"""Export AdaPoinTr/PoinTr test-time augmentation candidates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from export_pointr_predictions import ExportSample, _load_pointr_model, _rows_from_projected_split, _rows_from_selected_json, _select_rows, _validate_inputs
from pointr_if.io import load_point_cloud, save_point_cloud
from pointr_if.point_ops import resample_points_np, set_seed


@dataclass(frozen=True)
class Transform:
    name: str
    matrix: np.ndarray
    input_scale: float = 1.0
    dropout_keep: float = 1.0
    jitter_sigma: float = 0.0


def _rotation(axis: int, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    mat = np.eye(3, dtype=np.float32)
    axes = [0, 1, 2]
    axes.remove(axis)
    i, j = axes
    mat[i, i] = c
    mat[i, j] = -s
    mat[j, i] = s
    mat[j, j] = c
    return mat


def _transforms(names: list[str]) -> list[Transform]:
    all_transforms = [Transform("identity", np.eye(3, dtype=np.float32))]
    for scale in (0.98, 1.02):
        all_transforms.append(Transform(f"scale_{scale:.2f}".replace(".", "p"), np.eye(3, dtype=np.float32), input_scale=scale))
    for axis_name, axis in (("x", 0), ("y", 1), ("z", 2)):
        for degrees in (-10, -5, 5, 10):
            all_transforms.append(Transform(f"rot_{axis_name}_{degrees:+d}".replace("+", "p").replace("-", "m"), _rotation(axis, degrees)))
    for axes_name, signs in (
        ("mirror_x", (-1, 1, 1)),
        ("mirror_y", (1, -1, 1)),
        ("mirror_z", (1, 1, -1)),
        ("mirror_xy", (-1, -1, 1)),
        ("mirror_xz", (-1, 1, -1)),
        ("mirror_yz", (1, -1, -1)),
    ):
        all_transforms.append(Transform(axes_name, np.diag(np.asarray(signs, dtype=np.float32))))
    for keep in (0.9, 0.8):
        all_transforms.append(Transform(f"dropout_{int(keep * 100)}", np.eye(3, dtype=np.float32), dropout_keep=keep))
    for sigma in (0.002, 0.005):
        all_transforms.append(Transform(f"jitter_{str(sigma).replace('.', 'p')}", np.eye(3, dtype=np.float32), jitter_sigma=sigma))
    if not names or names == ["all"]:
        return all_transforms
    wanted = set(names)
    selected = [t for t in all_transforms if t.name in wanted]
    missing = wanted - {t.name for t in selected}
    if missing:
        raise ValueError(f"Unknown TTA transforms: {sorted(missing)}")
    return selected


def _apply_transform(points: np.ndarray, transform: Transform, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32)
    if transform.dropout_keep < 1.0:
        keep = max(1, int(round(out.shape[0] * transform.dropout_keep)))
        idx = rng.choice(out.shape[0], size=keep, replace=False)
        out = out[idx]
    out = out @ transform.matrix.T
    out = out * transform.input_scale
    if transform.jitter_sigma > 0.0:
        out = out + rng.normal(0.0, transform.jitter_sigma, size=out.shape).astype(np.float32)
    return out.astype(np.float32)


def _inverse_transform(points: np.ndarray, transform: Transform) -> np.ndarray:
    inv = np.linalg.inv(transform.matrix).astype(np.float32)
    out = np.asarray(points, dtype=np.float32) / max(transform.input_scale, 1e-8)
    return (out @ inv.T).astype(np.float32)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "category", "source_name", "transform_name", "candidate_path", "split"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_tta(args: argparse.Namespace) -> dict:
    pointr_root = Path(args.pointr_root).resolve()
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    split_name = args.split_name or args.split
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)

    if args.selected_json:
        rows = _rows_from_selected_json(Path(args.selected_json).resolve(), data_root, split_name)
    else:
        rows = _rows_from_projected_split(data_root, args.split, split_name)
    rows = _select_rows(rows, args.offset, args.limit)
    _validate_inputs(rows)
    transforms = _transforms(args.transforms)
    model = _load_pointr_model(pointr_root, config, checkpoint, device)
    manifest_rows = []

    for transform in transforms:
        for start in tqdm(range(0, len(rows), args.batch_size), desc=f"{split_name} {transform.name}"):
            batch_rows: list[ExportSample] = rows[start : start + args.batch_size]
            partials = []
            for row_index, row in enumerate(batch_rows, start=start):
                rng = np.random.default_rng(args.seed + row_index)
                partial = load_point_cloud(row.partial_path)
                partial = resample_points_np(partial, args.input_points, rng, mode="random")
                partial = _apply_transform(partial, transform, rng)
                partial = resample_points_np(partial, args.input_points, rng, mode="random")
                partials.append(partial)
            batch = torch.from_numpy(np.stack(partials, axis=0)).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred = model(batch)[-1].detach().cpu().numpy()
            for row, points in zip(batch_rows, pred):
                restored = _inverse_transform(points, transform)
                candidate_path = out_root / "predictions" / split_name / transform.name / row.category / row.model_id / "coarse.npy"
                save_point_cloud(candidate_path, restored)
                manifest_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "category": row.category,
                        "source_name": f"tta_{transform.name}",
                        "transform_name": transform.name,
                        "candidate_path": str(candidate_path),
                        "split": split_name,
                    }
                )

    manifest_path = out_root / "manifests" / f"{split_name}_tta_candidates.csv"
    _write_manifest(manifest_path, manifest_rows)
    summary = {
        "split": args.split,
        "split_name": split_name,
        "n_samples": len(rows),
        "n_transforms": len(transforms),
        "manifest": str(manifest_path),
        "out_root": str(out_root),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{split_name}_tta_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (log_dir / "command_log.txt").open("a", encoding="utf-8") as f:
        f.write(summary["command"] + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointr-root", default="external/PoinTr")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--selected-json", default=None)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-points", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=570)
    parser.add_argument("--transforms", nargs="+", default=["identity", "scale_0p98", "scale_1p02", "mirror_x", "mirror_y", "mirror_z"])
    export_tta(parser.parse_args())


if __name__ == "__main__":
    main()
