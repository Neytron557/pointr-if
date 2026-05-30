#!/usr/bin/env python
"""Build a submission archive without datasets, environments, or checkpoints."""
from __future__ import annotations

import argparse
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ALWAYS_INCLUDE_FILES = {
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "Makefile",
}

INCLUDE_TOP_LEVEL_DIRS = {
    "configs",
    "docs",
    "reports",
    "scripts",
    "src",
    "tests",
    "tools",
}

INCLUDE_DATA_PARTS = {
    "data/real_projected_shapenet55_adapointr_predictions/manifests",
    "data/real_projected_shapenet55_adapointr_predictions/logs",
}

OUTPUT_INCLUDE_SUFFIXES = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".png",
    ".txt",
    ".yaml",
    ".yml",
}

OUTPUT_INCLUDE_NAMES = {
    "command_log.txt",
    "metrics.csv",
    "metrics.json",
    "paired_stats.md",
    "paired_stats.json",
    "per_category_improvement.csv",
    "per_sample_delta.csv",
    "per_sample_metrics.csv",
    "qualitative_grid.png",
    "resolved_config.yaml",
    "summary.json",
    "summary_table.md",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "external",
    "node_modules",
}

EXCLUDED_SUFFIXES = {
    ".ckpt",
    ".npz",
    ".npy",
    ".pcd",
    ".pth",
    ".pt",
    ".tar",
    ".tgz",
    ".zip",
}


@dataclass(frozen=True)
class ArchiveSummary:
    out_path: Path
    n_files: int
    size_bytes: int


def _as_posix(path: Path) -> str:
    return path.as_posix()


def _is_under(path: Path, parent: str) -> bool:
    rel = _as_posix(path)
    return rel == parent or rel.startswith(parent + "/")


def _has_excluded_part(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def _include_path(rel: Path) -> bool:
    rel_posix = _as_posix(rel)
    if _has_excluded_part(rel):
        return False
    if rel.name.startswith(".") and rel.name not in {".gitignore"}:
        return False
    if rel.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if rel_posix in ALWAYS_INCLUDE_FILES:
        return True
    if rel.parts and rel.parts[0] in INCLUDE_TOP_LEVEL_DIRS:
        return True
    if any(_is_under(rel, parent) for parent in INCLUDE_DATA_PARTS):
        return True
    if rel.parts and rel.parts[0] == "outputs":
        if rel.name in OUTPUT_INCLUDE_NAMES:
            return True
        return rel.suffix.lower() in OUTPUT_INCLUDE_SUFFIXES and "predictions" not in rel.parts
    if rel.parts and rel.parts[0] == "deliverables":
        return rel.suffix.lower() in {".md", ".txt"} or rel.name == "archive_listing.txt"
    return False


def iter_archive_files(root: Path, out_path: Path) -> Iterable[Path]:
    root = root.resolve()
    out_path = out_path.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_obj = Path(dirpath)
        rel_dir = dirpath_obj.relative_to(root)
        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if name not in EXCLUDED_DIR_NAMES and not (rel_dir / name).match("outputs/*/test_eval/predictions")
        ]
        for filename in sorted(filenames):
            path = dirpath_obj / filename
            if path.resolve() == out_path:
                continue
            rel = path.relative_to(root)
            if _include_path(rel):
                yield rel


def build_archive(root: str | Path, out: str | Path) -> ArchiveSummary:
    root_path = Path(root).resolve()
    out_path = Path(out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = list(iter_archive_files(root_path, out_path))
    if not files:
        raise RuntimeError(f"No files matched archive include rules under {root_path}")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            zf.write(root_path / rel, _as_posix(rel))
    return ArchiveSummary(out_path=out_path, n_files=len(files), size_bytes=out_path.stat().st_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root to archive")
    parser.add_argument("--out", type=Path, required=True, help="Output .zip path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_archive(args.root, args.out)
    print(f"Wrote {summary.out_path} ({summary.n_files} files, {summary.size_bytes} bytes)")


if __name__ == "__main__":
    main()
