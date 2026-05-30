#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointr_if.io import load_point_cloud, read_manifest
from pointr_if.point_ops import resample_points_np


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:180]


def _parse_source(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Source must be name=path, got {spec!r}")
    name, root = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Source name is empty in {spec!r}")
    return name, Path(root).expanduser().resolve()


def _model_id_from_sample(sample_id: str, category: str | None = None) -> str | None:
    parts = sample_id.split("_")
    if category and category in parts:
        idx = parts.index(category)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(parts) >= 3:
        return parts[2]
    return None


def _candidate_paths(root: Path, sample_id: str, category: str | None = None) -> Iterable[Path]:
    safe = _safe_name(sample_id)
    model_id = _model_id_from_sample(sample_id, category)
    stems = [sample_id, safe, f"{safe}_refined", f"{safe}_coarse", f"{safe}_anchor"]
    for stem in stems:
        for suffix in (".npy", ".npz", ".pcd", ".txt", ".xyz"):
            yield root / f"{stem}{suffix}"
    if category and model_id:
        for filename in ("coarse.npy", "refined.npy", "pred.npy", "points.npy"):
            yield root / category / model_id / filename
            yield root / "predictions" / category / model_id / filename
            for split in ("train", "val", "test"):
                yield root / split / category / model_id / filename
                yield root / "predictions" / split / category / model_id / filename
        for transform_dir in (root.iterdir() if root.exists() else []):
            if transform_dir.is_dir():
                for filename in ("coarse.npy", "refined.npy", "pred.npy", "points.npy"):
                    yield transform_dir / category / model_id / filename
    for suffix in (".npy", ".npz", ".pcd", ".txt", ".xyz"):
        yield from root.rglob(f"{safe}{suffix}")
        yield from root.rglob(f"{safe}_refined{suffix}")


def _load_source(root: Path, sample_id: str, category: str | None = None) -> np.ndarray:
    for path in _candidate_paths(root, sample_id, category):
        if path.exists():
            return load_point_cloud(path)
    raise FileNotFoundError(f"No candidate file found for {sample_id} under {root}")


def _normalize_like_gt(points: np.ndarray, gt: np.ndarray) -> np.ndarray:
    center = gt.mean(axis=0, keepdims=True)
    scale = np.sqrt(((gt - center) ** 2).sum(axis=1)).max().clip(1e-8)
    return ((points - center) / scale).astype(np.float32)


def _symmetry_candidates(partial: np.ndarray, coarse: np.ndarray) -> list[tuple[str, np.ndarray]]:
    out = []
    for axis, name in enumerate(("mirror_x", "mirror_y", "mirror_z")):
        for base_name, cloud in (("partial", partial), ("coarse", coarse)):
            mirrored = cloud.copy()
            mirrored[:, axis] *= -1.0
            merged = np.concatenate([cloud, mirrored], axis=0)
            radius = np.linalg.norm(merged, axis=1)
            merged = merged[radius <= 1.35]
            out.append((f"sym_{base_name}_{name}", merged.astype(np.float32)))
    return out


def build_manifest(args: argparse.Namespace) -> dict:
    rows = read_manifest(args.triplet_manifest)
    source_specs = [_parse_source(spec) for spec in args.source]
    normalized_sources = set(args.normalized_source or [])
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_rows = []
    failures = []
    rng = np.random.default_rng(args.seed)

    for row_index, row in enumerate(rows):
        sample_id = row.get("sample_id") or row.get("id")
        gt_raw = load_point_cloud(row["gt"])
        partial = _normalize_like_gt(load_point_cloud(row["partial"]), gt_raw)
        coarse = _normalize_like_gt(load_point_cloud(row["coarse"]), gt_raw)
        candidates: list[np.ndarray] = []
        source_names: list[str] = []

        candidates.append(resample_points_np(coarse, args.candidate_points, rng, mode=args.resample_mode))
        source_names.append("adapointr_identity")
        if args.include_symmetry:
            for name, cloud in _symmetry_candidates(partial, coarse):
                candidates.append(resample_points_np(cloud, args.candidate_points, rng, mode=args.resample_mode))
                source_names.append(name)

        for source_name, root in source_specs:
            try:
                points = _load_source(root, sample_id, row.get("category", ""))
                if source_name not in normalized_sources:
                    points = _normalize_like_gt(points, gt_raw)
                candidates.append(resample_points_np(points, args.candidate_points, rng, mode=args.resample_mode))
                source_names.append(source_name)
            except Exception as exc:  # noqa: BLE001 - report all candidate-source failures.
                if args.allow_missing_sources:
                    failures.append({"sample_id": sample_id, "source": source_name, "error": str(exc)})
                    continue
                raise

        candidate_path = out_root / f"{_safe_name(sample_id)}.npz"
        np.savez_compressed(
            candidate_path,
            candidates=np.stack(candidates, axis=0).astype(np.float32),
            source_names=np.asarray(source_names),
        )
        out_rows.append(
            {
                "sample_id": sample_id,
                "category": row.get("category", ""),
                "partial_path": row["partial"],
                "coarse_path": row["coarse"],
                "gt_path": row["gt"],
                "candidate_npz": str(candidate_path),
                "split": row.get("split", ""),
            }
        )

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "category", "partial_path", "coarse_path", "gt_path", "candidate_npz", "split"]
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {
        "triplet_manifest": str(args.triplet_manifest),
        "out_manifest": str(out_manifest),
        "out_root": str(out_root),
        "n_rows": len(out_rows),
        "n_sources": len(source_names) if out_rows else 0,
        "source_specs": [name for name, _ in source_specs],
        "include_symmetry": bool(args.include_symmetry),
        "candidate_points": int(args.candidate_points),
        "resample_mode": args.resample_mode,
        "failures": failures,
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    (out_manifest.parent / "candidate_manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a per-sample SEED candidate-bank manifest.")
    parser.add_argument("--triplet-manifest", required=True)
    parser.add_argument("--source", action="append", default=[], help="Candidate source as name=directory")
    parser.add_argument("--normalized-source", action="append", default=[], help="Source name already in normalized GT coordinates")
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--candidate-points", type=int, default=1024)
    parser.add_argument("--resample-mode", choices=["random", "fps", "none"], default="fps")
    parser.add_argument("--seed", type=int, default=570)
    parser.add_argument("--include-symmetry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing-sources", action="store_true")
    build_manifest(parser.parse_args())


if __name__ == "__main__":
    main()
