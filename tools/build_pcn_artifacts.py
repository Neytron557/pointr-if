#!/usr/bin/env python
"""Build PCN source manifests and multi-view groups for A100 experiments."""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any


def _partial_path(root: Path, split: str, taxonomy_id: str, model_id: str, view: int) -> Path:
    return root / split / "partial" / taxonomy_id / model_id / f"{view:02d}.pcd"


def _complete_path(root: Path, split: str, taxonomy_id: str, model_id: str) -> Path:
    return root / split / "complete" / taxonomy_id / f"{model_id}.pcd"


def _sample_id(split: str, taxonomy_id: str, model_id: str, view: int) -> str:
    return f"{split}_{taxonomy_id}_{model_id}_view{view:02d}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pcn_artifacts(
    *,
    pcn_root: str | Path,
    category_json: str | Path,
    out_dir: str | Path,
    train_source_view: int = 0,
    validate_paths: bool = True,
) -> dict[str, Any]:
    root = Path(pcn_root).resolve()
    out = Path(out_dir)
    categories = json.loads(Path(category_json).read_text(encoding="utf-8"))
    source_fields = ["sample_id", "category", "taxonomy_id", "model_id", "view_id", "partial_path", "gt_path", "split"]
    summaries: dict[str, Any] = {}
    missing: list[str] = []

    for split in ("train", "val", "test"):
        source_rows: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        n_views = 8 if split == "train" else 1
        for category in categories:
            taxonomy_id = category["taxonomy_id"]
            for model_id in category.get(split, []):
                gt_path = _complete_path(root, split, taxonomy_id, model_id)
                members = []
                for view in range(n_views):
                    partial_path = _partial_path(root, split, taxonomy_id, model_id, view)
                    if validate_paths:
                        if not partial_path.exists():
                            missing.append(str(partial_path))
                        if not gt_path.exists():
                            missing.append(str(gt_path))
                    member = {
                        "partial_path": str(partial_path),
                        "coarse_path": "",
                        "sample_id": _sample_id(split, taxonomy_id, model_id, view),
                        "generated": False,
                    }
                    members.append(member)
                    if split != "train" or view == int(train_source_view):
                        source_rows.append(
                            {
                                "sample_id": member["sample_id"],
                                "category": taxonomy_id,
                                "taxonomy_id": taxonomy_id,
                                "model_id": model_id,
                                "view_id": f"{view:02d}",
                                "partial_path": str(partial_path),
                                "gt_path": str(gt_path),
                                "split": split,
                            }
                        )
                groups.append(
                    {
                        "group_id": f"{taxonomy_id}-{model_id}",
                        "category": taxonomy_id,
                        "gt_path": str(gt_path),
                        "members": members,
                    }
                )
        source_path = out / f"{split}_sources.csv"
        groups_path = out / f"{split}_groups.json"
        _write_csv(source_path, source_rows, source_fields)
        groups_path.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
        summaries[split] = {
            "source_manifest": str(source_path),
            "groups_path": str(groups_path),
            "n_source_rows": len(source_rows),
            "n_groups": len(groups),
            "views_per_group": n_views,
        }

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"{len(missing)} PCN files are missing. First missing:\n{preview}")

    summary = {
        "pcn_root": str(root),
        "category_json": str(Path(category_json).resolve()),
        "out_dir": str(out.resolve()),
        "train_source_view": int(train_source_view),
        "splits": summaries,
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "pcn_artifact_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcn-root", default="data/ShapeNetCompletion")
    parser.add_argument("--category-json", default="external/PoinTr/data/PCN/PCN.json")
    parser.add_argument("--out-dir", default="data/pcn_a100")
    parser.add_argument("--train-source-view", type=int, default=0)
    parser.add_argument("--no-validate-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_pcn_artifacts(
        pcn_root=args.pcn_root,
        category_json=args.category_json,
        out_dir=args.out_dir,
        train_source_view=args.train_source_view,
        validate_paths=not args.no_validate_paths,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
