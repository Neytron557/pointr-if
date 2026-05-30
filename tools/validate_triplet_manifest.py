#!/usr/bin/env python
"""Validate a PoinTr-IF triplet manifest before training."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Allow running from a fresh checkout before editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointr_if.io import load_point_cloud, read_manifest


LEGACY_REQUIRED_COLUMNS = {"id", "partial", "coarse", "gt"}
REAL_REQUIRED_COLUMNS = {"sample_id", "category", "partial_path", "coarse_path", "gt_path", "split"}


def _stats(points: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(arr)
    return {
        "shape": list(arr.shape),
        "finite": bool(finite.all()),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "min": float(np.nanmin(arr)) if arr.size else math.nan,
        "max": float(np.nanmax(arr)) if arr.size else math.nan,
    }


def _read_header(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def validate_manifest(path: str | Path, max_samples: int = 5) -> Dict[str, Any]:
    manifest = Path(path)
    header = _read_header(manifest)
    header_set = set(header)
    if not (LEGACY_REQUIRED_COLUMNS.issubset(header_set) or REAL_REQUIRED_COLUMNS.issubset(header_set)):
        raise ValueError(
            f"Manifest {manifest} must contain either legacy columns {sorted(LEGACY_REQUIRED_COLUMNS)} "
            f"or real-result columns {sorted(REAL_REQUIRED_COLUMNS)}; found {header}"
        )

    rows = read_manifest(manifest)
    split_counts = Counter(row.get("split", "unknown") or "unknown" for row in rows)
    category_counts = Counter(row.get("category", "unknown") or "unknown" for row in rows)
    summary: Dict[str, Any] = {
        "manifest": str(manifest),
        "n_rows": len(rows),
        "splits": dict(sorted(split_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "path_checked_rows": len(rows),
        "path_checked_files": 0,
        "checked_rows": 0,
        "missing_files": [],
        "samples": [],
    }

    for row_index, row in enumerate(rows):
        for key in ("partial", "coarse", "gt"):
            cloud_path = Path(row[key])
            summary["path_checked_files"] += 1
            if not cloud_path.exists():
                summary["missing_files"].append({"row_index": row_index, "id": row.get("id", ""), "field": key, "path": str(cloud_path)})
    if summary["missing_files"]:
        raise FileNotFoundError(f"Manifest has {len(summary['missing_files'])} missing files")

    for row_index, row in enumerate(rows[: max(0, max_samples)]):
        sample: Dict[str, Any] = {"row_index": row_index, "id": row.get("id", "")}
        for key in ("partial", "coarse", "gt"):
            cloud_path = Path(row[key])
            points = load_point_cloud(cloud_path)
            if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
                raise ValueError(f"{key} for row {row_index} must be non-empty [N, >=3], got {points.shape}: {cloud_path}")
            stats = _stats(points[:, :3])
            if not stats["finite"]:
                raise ValueError(f"{key} for row {row_index} contains NaN/Inf values: {cloud_path}")
            sample[key] = {"path": str(cloud_path), **stats}
        summary["samples"].append(sample)
        summary["checked_rows"] += 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a PoinTr-IF CSV manifest")
    parser.add_argument("manifest")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--json-out", default=None, help="Optional path for a JSON validation summary")
    args = parser.parse_args()

    summary = validate_manifest(args.manifest, max_samples=args.max_samples)
    print(json.dumps(summary, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
