#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointr_if.io import load_point_cloud, read_manifest


def _summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": None, "median": None, "max": None, "n_unique": 0, "unique_head": []}
    unique = sorted(set(values))
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "n_unique": len(unique),
        "unique_head": unique[:20],
    }


def audit_manifest(path: str | Path, max_samples: int | None = None) -> dict[str, Any]:
    rows = read_manifest(path)
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    counts: dict[str, list[int]] = defaultdict(list)
    categories: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    samples = []
    for row in rows:
        categories[str(row.get("category", ""))] += 1
        splits[str(row.get("split", ""))] += 1
        sample = {"sample_id": row.get("sample_id") or row.get("id")}
        for key in ("partial", "coarse", "gt"):
            n = int(load_point_cloud(row[key]).shape[0])
            counts[key].append(n)
            sample[key] = n
        samples.append(sample)
    return {
        "manifest": str(path),
        "n_rows": len(rows),
        "counts": {key: _summary(values) for key, values in counts.items()},
        "categories": dict(categories),
        "splits": dict(splits),
        "samples_head": samples[:20],
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw point counts in a triplet manifest.")
    parser.add_argument("manifest")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out", "--json-out", dest="out", default=None)
    args = parser.parse_args()

    summary = audit_manifest(args.manifest, max_samples=args.max_samples)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
