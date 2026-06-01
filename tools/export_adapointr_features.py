#!/usr/bin/env python
"""Export AdaPoinTr completions and decoder features from a source manifest."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
POINTR_ROOT = ROOT / "external" / "PoinTr"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(POINTR_ROOT))

from pointr_if.io import load_point_cloud, resolve_project_path, save_point_cloud
from pointr_if.point_ops import resample_points_np, set_seed


def _load_pointr_config(config: str | Path):
    from utils.config import cfg_from_yaml_file

    config_path = Path(config).resolve()
    cwd = Path.cwd()
    try:
        os.chdir(POINTR_ROOT)
        rel = config_path.relative_to(POINTR_ROOT) if config_path.is_relative_to(POINTR_ROOT) else config_path
        return cfg_from_yaml_file(str(rel))
    finally:
        os.chdir(cwd)


def load_adapointr(config: str | Path, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    from models import build_model_from_cfg

    cfg = _load_pointr_config(config)
    cwd = Path.cwd()
    try:
        os.chdir(POINTR_ROOT)
        model = build_model_from_cfg(cfg.model)
    finally:
        os.chdir(cwd)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("base_model") or ckpt.get("model")
    if state is None:
        raise RuntimeError(f"Checkpoint {checkpoint} does not contain base_model/model weights")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def forward_with_decoder_features(model: torch.nn.Module, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, coarse_point_cloud, denoise_length = model.base_model(xyz)
    if int(denoise_length) != 0:
        raise RuntimeError("Feature export expects AdaPoinTr eval mode with denoise_length=0")
    bsz, n_query, channels = q.shape
    global_feature = model.increase_dim(q.transpose(1, 2)).transpose(1, 2)
    global_feature = torch.max(global_feature, dim=1)[0]
    rebuild_feature = torch.cat(
        [
            global_feature.unsqueeze(-2).expand(-1, n_query, -1),
            q,
            coarse_point_cloud,
        ],
        dim=-1,
    )
    if model.decoder_type == "fold":
        reduced = model.reduce_map(rebuild_feature.reshape(bsz * n_query, -1))
        relative_xyz = model.decode_head(reduced).reshape(bsz, n_query, 3, -1)
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)
    else:
        reduced = model.reduce_map(rebuild_feature)
        relative_xyz = model.decode_head(reduced)
        rebuild_points = relative_xyz + coarse_point_cloud.unsqueeze(-2)
    factor = int(rebuild_points.shape[2])
    dense_points = rebuild_points.reshape(bsz, n_query * factor, 3).contiguous()
    dense_features = q.unsqueeze(2).expand(-1, -1, factor, -1).reshape(bsz, n_query * factor, channels).contiguous()
    return dense_points, dense_features


def _read_source_manifest(path: str | Path) -> list[dict[str, str]]:
    path = resolve_project_path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"sample_id", "category", "partial_path", "gt_path", "split"}
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    base = path.parent
    for row in rows:
        row["partial_path"] = str(resolve_project_path(row["partial_path"], base=base))
        row["gt_path"] = str(resolve_project_path(row["gt_path"], base=base))
    return rows


def _select_aligned(points: np.ndarray, features: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    total = int(points.shape[0])
    if total == n:
        return points.astype(np.float32), features
    if total < n:
        repeats = int(np.ceil(n / total))
        return np.tile(points, (repeats, 1))[:n].astype(np.float32), np.tile(features, (repeats, 1))[:n]
    idx = rng.choice(total, size=n, replace=False)
    return points[idx].astype(np.float32), features[idx]


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["sample_id", "category", "partial_path", "coarse_path", "gt_path", "feature_path", "split"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    set_seed(int(args.seed))
    rows = _read_source_manifest(args.source_manifest)
    rows = rows[int(args.offset) :]
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError("No source rows selected for export")
    out_root = Path(args.out_root)
    model = load_adapointr(args.config, args.checkpoint, device)
    manifest_rows: list[dict[str, Any]] = []
    feature_dtype = np.float16 if args.feature_dtype == "float16" else np.float32

    for start in tqdm(range(0, len(rows), int(args.batch_size)), desc=f"export {Path(args.source_manifest).stem}"):
        batch_rows = rows[start : start + int(args.batch_size)]
        partials = []
        for local_idx, row in enumerate(batch_rows):
            row_index = int(args.offset) + start + local_idx
            rng = np.random.default_rng(int(args.seed) + row_index)
            partial = load_point_cloud(row["partial_path"])
            partials.append(resample_points_np(partial, int(args.input_points), rng, mode="random"))
        batch = torch.from_numpy(np.stack(partials, axis=0)).to(device=device, dtype=torch.float32)
        dense_points, dense_features = forward_with_decoder_features(model, batch)
        dense_points_np = dense_points.detach().cpu().numpy()
        dense_features_np = dense_features.detach().cpu().numpy()
        for local_idx, row in enumerate(batch_rows):
            row_index = int(args.offset) + start + local_idx
            rng = np.random.default_rng(int(args.seed) + 1000000 + row_index)
            points, features = _select_aligned(dense_points_np[local_idx], dense_features_np[local_idx], int(args.output_points), rng)
            split = row.get("split") or "export"
            category = row["category"]
            model_id = row.get("model_id") or Path(row["gt_path"]).stem
            pred_dir = out_root / "predictions" / split / category / model_id
            feat_dir = out_root / "features" / split / category / model_id
            coarse_path = (pred_dir / "coarse.npy").resolve()
            feature_path = (feat_dir / "decoder_features.npz").resolve()
            save_point_cloud(coarse_path, points)
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(feature_path, features=features.astype(feature_dtype))
            manifest_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "category": category,
                    "partial_path": row["partial_path"],
                    "coarse_path": str(coarse_path),
                    "gt_path": row["gt_path"],
                    "feature_path": str(feature_path),
                    "split": split,
                }
            )

    split_name = args.split_name or rows[0].get("split") or "export"
    manifest_path = out_root / "manifests" / f"{split_name}_triplets.csv"
    _write_manifest(manifest_path, manifest_rows)
    summary = {
        "source_manifest": str(resolve_project_path(args.source_manifest)),
        "manifest": str(manifest_path),
        "n_samples": len(manifest_rows),
        "input_points": int(args.input_points),
        "output_points": int(args.output_points),
        "feature_dtype": args.feature_dtype,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{split_name}_feature_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--input-points", type=int, default=2048)
    parser.add_argument("--output-points", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=570)
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def main() -> None:
    export(parse_args())


if __name__ == "__main__":
    main()
