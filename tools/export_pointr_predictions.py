#!/usr/bin/env python
"""Export real PoinTr/AdaPoinTr completions into PoinTr-IF triplet manifests."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointr_if.io import load_point_cloud, save_point_cloud
from pointr_if.point_ops import resample_points_np, set_seed


@dataclass
class ExportSample:
    sample_id: str
    category: str
    taxonomy_id: str
    model_id: str
    partial_path: Path
    gt_path: Path


def _load_pointr_model(pointr_root: Path, config: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(pointr_root))
    cwd = Path.cwd()
    try:
        os.chdir(pointr_root)
        from utils.config import cfg_from_yaml_file
        from models import build_model_from_cfg

        cfg = cfg_from_yaml_file(str(config if config.is_absolute() else config.relative_to(pointr_root)))
        model = build_model_from_cfg(cfg.model)
    finally:
        os.chdir(cwd)

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model") or ckpt.get("base_model")
    if state is None:
        raise RuntimeError(f"Checkpoint {checkpoint} does not contain 'model' or 'base_model' weights")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def _rows_from_selected_json(path: Path, data_root: Path, split_name: str) -> List[ExportSample]:
    selected = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in selected:
        taxonomy_id = row["taxonomy_id"]
        model_id = row["model_id"]
        rows.append(
            ExportSample(
                sample_id=f"{split_name}_{taxonomy_id}_{model_id}_view0",
                category=taxonomy_id,
                taxonomy_id=taxonomy_id,
                model_id=model_id,
                partial_path=data_root / row["partial_member"],
                gt_path=data_root / row["gt_member"],
            )
        )
    return rows


def _rows_from_projected_split(data_root: Path, split: str, split_name: str) -> List[ExportSample]:
    list_file = data_root / "Projected_ShapeNet-55_noise" / f"{split}.txt"
    if not list_file.exists():
        raise FileNotFoundError(f"Projected ShapeNet split file not found: {list_file}")
    rows = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        stem = Path(line).stem
        taxonomy_id, model_id = stem.split("-", 1)
        rows.append(
            ExportSample(
                sample_id=f"{split_name}_{taxonomy_id}_{model_id}_view0",
                category=taxonomy_id,
                taxonomy_id=taxonomy_id,
                model_id=model_id,
                partial_path=data_root / "pcd" / taxonomy_id / model_id / "models" / "0.pcd",
                gt_path=data_root / line,
            )
        )
    return rows


def _select_rows(rows: List[ExportSample], offset: int, limit: int | None) -> List[ExportSample]:
    start = max(0, int(offset))
    end = None if limit is None else start + max(0, int(limit))
    return rows[start:end]


def _validate_inputs(rows: Iterable[ExportSample]) -> None:
    missing = []
    for row in rows:
        for field, path in (("partial_path", row.partial_path), ("gt_path", row.gt_path)):
            if not path.exists():
                missing.append({"sample_id": row.sample_id, "field": field, "path": str(path)})
    if missing:
        preview = "\n".join(f"{m['sample_id']} {m['field']} {m['path']}" for m in missing[:10])
        raise FileNotFoundError(f"{len(missing)} input point-cloud files are missing. First missing:\n{preview}")


def _write_manifest(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "category", "partial_path", "coarse_path", "gt_path", "split"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_predictions(args: argparse.Namespace) -> dict:
    pointr_root = Path(args.pointr_root).resolve()
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    split_name = args.split_name or args.split
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    set_seed(args.seed)

    if args.selected_json:
        rows = _rows_from_selected_json(Path(args.selected_json).resolve(), data_root, split_name)
    else:
        rows = _rows_from_projected_split(data_root, args.split, split_name)
    rows = _select_rows(rows, args.offset, args.limit)
    if not rows:
        raise ValueError("No samples selected for export")
    _validate_inputs(rows)

    model = _load_pointr_model(pointr_root, config, checkpoint, device)
    prediction_root = out_root / "predictions" / split_name
    manifest_rows = []
    failures = []

    for start in tqdm(range(0, len(rows), args.batch_size), desc=f"export {split_name}"):
        batch_rows = rows[start : start + args.batch_size]
        partials = []
        for row_index, row in enumerate(batch_rows, start=start):
            rng = np.random.default_rng(args.seed + row_index)
            partial = load_point_cloud(row.partial_path)
            partial = resample_points_np(partial, args.input_points, rng)
            partials.append(partial)
        batch = torch.from_numpy(np.stack(partials, axis=0)).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            pred = model(batch)[-1].detach().cpu().numpy()
        for row, points in zip(batch_rows, pred):
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                failures.append({"sample_id": row.sample_id, "reason": f"invalid prediction shape/values {points.shape}"})
                continue
            coarse_path = prediction_root / row.category / row.model_id / "coarse.npy"
            save_point_cloud(coarse_path, points.astype(np.float32))
            manifest_rows.append(
                {
                    "sample_id": row.sample_id,
                    "category": row.category,
                    "partial_path": str(row.partial_path),
                    "coarse_path": str(coarse_path),
                    "gt_path": str(row.gt_path),
                    "split": split_name,
                }
            )

    if failures:
        raise RuntimeError(f"{len(failures)} predictions failed numeric validation: {failures[:5]}")

    manifest_path = out_root / "manifests" / f"{split_name}_triplets.csv"
    _write_manifest(manifest_path, manifest_rows)
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "split_name": split_name,
        "n_samples": len(manifest_rows),
        "input_points": args.input_points,
        "prediction_points": int(pred.shape[1]) if manifest_rows else 0,
        "pointr_root": str(pointr_root),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "device": str(device),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{split_name}_export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (log_dir / "command_log.txt").open("a", encoding="utf-8") as f:
        f.write(summary["command"] + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="projected-shapenet55", choices=["projected-shapenet55"])
    parser.add_argument("--pointr-root", default="external/PoinTr")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", required=True, help="Source split name, e.g. train or test")
    parser.add_argument("--split-name", default=None, help="Output split name, e.g. val when slicing test")
    parser.add_argument("--selected-json", default=None, help="Optional selected-members JSON produced by the subset builder")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0, help="Accepted for command compatibility; export is synchronous")
    parser.add_argument("--input-points", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=570)
    return parser.parse_args()


def main() -> None:
    export_predictions(parse_args())


if __name__ == "__main__":
    main()
