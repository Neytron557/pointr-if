#!/usr/bin/env python
"""Build object-level multi-view groups from PoinTr-IF triplet manifests."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointr_if.io import read_manifest


def _parse_category_model(row: dict[str, str]) -> tuple[str, str]:
    category = row.get("category", "")
    sample_id = row.get("sample_id") or row.get("id", "")
    if category and f"_{category}_" in sample_id:
        tail = sample_id.split(f"_{category}_", 1)[1]
        model_id = tail.rsplit("_view", 1)[0]
        return category, model_id
    gt_stem = Path(row["gt"]).stem
    if "-" in gt_stem:
        category_from_gt, model_id = gt_stem.split("-", 1)
        return category or category_from_gt, model_id
    return category, gt_stem


def _group_id(row: dict[str, str]) -> str:
    category, model_id = _parse_category_model(row)
    if category and model_id:
        return f"{category}-{model_id}"
    return str(Path(row["gt"]).resolve())


def _views_summary(counts: list[int]) -> dict[str, Any]:
    if not counts:
        return {"min": 0, "median": 0, "max": 0, "ge2": 0, "ge3": 0, "ge4": 0}
    return {
        "min": int(min(counts)),
        "median": float(median(counts)),
        "max": int(max(counts)),
        "ge2": int(sum(c >= 2 for c in counts)),
        "ge3": int(sum(c >= 3 for c in counts)),
        "ge4": int(sum(c >= 4 for c in counts)),
    }


def build_groups_for_manifest(manifest: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_manifest(manifest)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_group_id(row)].append(row)

    groups: list[dict[str, Any]] = []
    for group_id, group_rows in sorted(grouped.items()):
        first = group_rows[0]
        category, _model_id = _parse_category_model(first)
        groups.append(
            {
                "group_id": group_id,
                "category": category,
                "gt_path": first["gt"],
                "members": [
                    {
                        "partial_path": row["partial"],
                        "coarse_path": row["coarse"],
                        "sample_id": row.get("sample_id") or row.get("id") or Path(row["partial"]).stem,
                        "generated": False,
                    }
                    for row in group_rows
                ],
            }
        )

    counts = [len(group["members"]) for group in groups]
    categories = Counter(str(group["category"]) for group in groups)
    audit = {
        "manifest": str(manifest),
        "n_rows": len(rows),
        "n_groups": len(groups),
        "views_per_group": _views_summary(counts),
        "view_count_histogram": dict(sorted(Counter(counts).items())),
        "categories": dict(categories.most_common()),
    }
    return groups, audit


def write_group_outputs(
    *,
    train_manifest: str | Path,
    val_manifest: str | Path,
    out_dir: str | Path,
    audit_out: str | Path,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_groups, train_audit = build_groups_for_manifest(train_manifest)
    val_groups, val_audit = build_groups_for_manifest(val_manifest)
    train_path = out / "train_groups.json"
    val_path = out / "val_groups.json"
    train_path.write_text(json.dumps(train_groups, indent=2) + "\n", encoding="utf-8")
    val_path.write_text(json.dumps(val_groups, indent=2) + "\n", encoding="utf-8")
    summary = {
        "train": {**train_audit, "groups_path": str(train_path)},
        "val": {**val_audit, "groups_path": str(val_path)},
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    audit_path = Path(audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--out-dir", default="data/real_projected_shapenet55_groups")
    parser.add_argument("--audit-out", default="reports/real_projected_shapenet55_group_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_group_outputs(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        out_dir=args.out_dir,
        audit_out=args.audit_out,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
