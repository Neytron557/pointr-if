#!/usr/bin/env python
"""Build a PoinTr-IF triplet dataset.

Two modes:
1. Copy/convert a CSV manifest with columns id,partial,coarse,gt into .npz files.
2. Create a manifest by matching files from partial/coarse/gt roots by basename.

For PCN-style directories, creating a CSV manifest manually is the safest option
because partial views and PoinTr output names may differ across repositories.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow running without installing the package.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointr_if.io import find_pc_files, load_point_cloud, read_manifest, write_manifest


def rel_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return str(rel.with_suffix(""))


def basename_key(path: Path) -> str:
    return path.stem


def create_manifest(partial_root: Path, coarse_root: Path, gt_root: Path, out_csv: Path, match: str = "basename") -> None:
    key_fn = basename_key if match == "basename" else lambda p: rel_key(p, partial_root)
    partials = find_pc_files(partial_root)
    coarse_files = find_pc_files(coarse_root)
    gt_files = find_pc_files(gt_root)

    coarse_by_key = {basename_key(p) if match == "basename" else str(p.relative_to(coarse_root).with_suffix("")): p for p in coarse_files}
    gt_by_key = {basename_key(p) if match == "basename" else str(p.relative_to(gt_root).with_suffix("")): p for p in gt_files}
    rows = []
    misses = 0
    for p in partials:
        key = basename_key(p) if match == "basename" else str(p.relative_to(partial_root).with_suffix(""))
        if key in coarse_by_key and key in gt_by_key:
            rows.append({"id": key.replace("/", "_"), "partial": str(p), "coarse": str(coarse_by_key[key]), "gt": str(gt_by_key[key])})
        else:
            misses += 1
    write_manifest(out_csv, rows)
    print(f"Wrote {len(rows)} matched rows to {out_csv}; skipped {misses} partial files without matches.")


def convert_manifest_to_npz(manifest: Path, out_dir: Path, max_samples: int | None = None) -> None:
    rows = read_manifest(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_samples is not None:
        rows = rows[:max_samples]
    for i, row in enumerate(rows):
        partial = load_point_cloud(row["partial"])
        coarse = load_point_cloud(row["coarse"])
        gt = load_point_cloud(row["gt"])
        sample_id = row.get("id") or f"sample_{i:06d}"
        np.savez_compressed(out_dir / f"{sample_id}.npz", partial=partial, coarse=coarse, gt=gt)
    print(f"Wrote {len(rows)} .npz triplets to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_manifest = sub.add_parser("manifest", help="Match roots and create a CSV manifest")
    p_manifest.add_argument("--partial-root", required=True)
    p_manifest.add_argument("--coarse-root", required=True)
    p_manifest.add_argument("--gt-root", required=True)
    p_manifest.add_argument("--out-csv", required=True)
    p_manifest.add_argument("--match", choices=["basename", "relative"], default="basename")

    p_npz = sub.add_parser("npz", help="Convert a manifest into .npz triplets")
    p_npz.add_argument("--manifest", required=True)
    p_npz.add_argument("--out-dir", required=True)
    p_npz.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "manifest":
        create_manifest(Path(args.partial_root), Path(args.coarse_root), Path(args.gt_root), Path(args.out_csv), match=args.match)
    elif args.cmd == "npz":
        convert_manifest_to_npz(Path(args.manifest), Path(args.out_dir), args.max_samples)


if __name__ == "__main__":
    main()
