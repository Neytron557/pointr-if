#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


FIELDS = [
    "run",
    "best_refined_cd",
    "coarse_cd",
    "refined_cd",
    "cd_improvement_pct",
    "coarse_fscore",
    "refined_fscore",
    "fscore_gain",
    "n_samples",
    "checkpoint",
    "summary_path",
]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_summary(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    last = data.get("last_val", {})
    eval_summary = data if "refined_cd" in data and "last_val" not in data else {}
    metrics = eval_summary or last
    checkpoint = data.get("checkpoint")
    if checkpoint is None:
        best = path.parent / "best_model.pt"
        checkpoint = str(best) if best.exists() else ""
    return {
        "run": path.parent.name,
        "best_refined_cd": data.get("best_refined_cd", metrics.get("refined_cd")),
        "coarse_cd": metrics.get("coarse_cd"),
        "refined_cd": metrics.get("refined_cd"),
        "cd_improvement_pct": metrics.get("cd_improvement_pct"),
        "coarse_fscore": metrics.get("coarse_fscore"),
        "refined_fscore": metrics.get("refined_fscore"),
        "fscore_gain": metrics.get("fscore_gain"),
        "n_samples": metrics.get("n_samples"),
        "checkpoint": checkpoint,
        "summary_path": str(path),
    }


def write_csv(rows: Iterable[Dict[str, Any]], out: Path) -> None:
    rows = list(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: Iterable[Dict[str, Any]], out: Path) -> None:
    rows = list(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Run",
        "Best refined CD",
        "Coarse CD",
        "Refined CD",
        "CD improvement %",
        "Coarse F-score",
        "Refined F-score",
        "F-score gain",
        "N",
        "Checkpoint",
    ]
    keys = [
        "run",
        "best_refined_cd",
        "coarse_cd",
        "refined_cd",
        "cd_improvement_pct",
        "coarse_fscore",
        "refined_fscore",
        "fscore_gain",
        "n_samples",
        "checkpoint",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key in keys) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PoinTr-IF train/eval summary JSON files")
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--out", default="outputs/summary_table.csv")
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    rows = [_load_summary(Path(path)) for path in args.summaries]
    out = Path(args.out)
    write_csv(rows, out)
    print(out)
    if args.markdown_out:
        md_out = Path(args.markdown_out)
        write_markdown(rows, md_out)
        print(md_out)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
