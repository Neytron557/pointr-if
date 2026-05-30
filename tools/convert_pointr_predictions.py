#!/usr/bin/env python
"""Create PoinTr-IF triplet manifests from PoinTr/AdaPoinTr prediction folders."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

# Allow running before editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointr_if.io import find_pc_files, load_point_cloud, save_point_cloud, write_manifest


def _key(path: Path, root: Path, mode: str) -> str:
    if mode == "relative":
        return str(path.relative_to(root).with_suffix(""))
    name = path.stem
    for suffix in ("_coarse", "_pred", "_prediction", "_complete", "_gt", "_partial", "_input"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _index(root: Path, mode: str) -> Dict[str, Path]:
    files = find_pc_files(root)
    indexed: Dict[str, Path] = {}
    duplicates: List[str] = []
    for path in files:
        key = _key(path, root, mode)
        if key in indexed:
            duplicates.append(key)
            continue
        indexed[key] = path
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:8])
        raise ValueError(f"Duplicate keys under {root} using match={mode}: {preview}")
    return indexed


def _maybe_copy(rows: Iterable[Dict[str, str]], converted_root: Path | None) -> List[Dict[str, str]]:
    rows = list(rows)
    if converted_root is None:
        return rows
    converted_root.mkdir(parents=True, exist_ok=True)
    converted_rows = []
    for row in rows:
        converted = {"id": row["id"]}
        for field in ("partial", "coarse", "gt"):
            src = Path(row[field])
            dst = converted_root / field / f"{row['id']}.npy"
            points = load_point_cloud(src)
            save_point_cloud(dst, points)
            converted[field] = str(dst.resolve())
        converted_rows.append(converted)
    return converted_rows


def build_manifest(
    partial_dir: str | Path,
    coarse_dir: str | Path,
    gt_dir: str | Path,
    output_manifest: str | Path,
    split: str,
    limit: int | None = None,
    match: str = "basename",
    converted_root: str | Path | None = None,
) -> List[Dict[str, str]]:
    partial_root = Path(partial_dir)
    coarse_root = Path(coarse_dir)
    gt_root = Path(gt_dir)
    if match not in {"basename", "relative"}:
        raise ValueError("match must be 'basename' or 'relative'")

    partials = _index(partial_root, match)
    coarse = _index(coarse_root, match)
    gt = _index(gt_root, match)

    rows = []
    missing = []
    for key in sorted(partials):
        if key not in coarse or key not in gt:
            missing.append(key)
            continue
        sample_id = f"{split}_{key}".replace("/", "_")
        rows.append(
            {
                "id": sample_id,
                "partial": str(partials[key].resolve()),
                "coarse": str(coarse[key].resolve()),
                "gt": str(gt[key].resolve()),
            }
        )
        if limit is not None and len(rows) >= limit:
            break

    converted = _maybe_copy(rows, Path(converted_root) if converted_root else None)
    write_manifest(output_manifest, converted)
    print(f"Wrote {len(converted)} rows to {output_manifest}")
    print(f"Indexed partial={len(partials)} coarse={len(coarse)} gt={len(gt)}; skipped missing matches={len(missing)}")
    if missing:
        print("First missing keys: " + ", ".join(missing[:10]))
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PoinTr/AdaPoinTr prediction folders to a PoinTr-IF manifest")
    parser.add_argument("--partial-dir", required=True)
    parser.add_argument("--coarse-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--match", choices=["basename", "relative"], default="basename")
    parser.add_argument(
        "--converted-root",
        default=None,
        help="Optional output root for normalized .npy copies; useful when source files are mixed formats.",
    )
    args = parser.parse_args()
    build_manifest(
        partial_dir=args.partial_dir,
        coarse_dir=args.coarse_dir,
        gt_dir=args.gt_dir,
        output_manifest=args.output_manifest,
        split=args.split,
        limit=args.limit,
        match=args.match,
        converted_root=args.converted_root,
    )


if __name__ == "__main__":
    main()
