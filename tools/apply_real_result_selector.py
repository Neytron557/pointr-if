#!/usr/bin/env python
"""Apply a validation-learned coarse/refined selector to real evaluation metrics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def learn_category_policy(
    train_delta_csv: str | Path,
    *,
    min_samples: int = 1,
    min_mean_delta: float = 0.0,
    default_choice: str = "refined",
) -> dict[str, Any]:
    rows = _read_csv(Path(train_delta_csv))
    by_category: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(float(row["cd_delta"]))
    categories: dict[str, Any] = {}
    for category, deltas in sorted(by_category.items()):
        mean_delta = sum(deltas) / len(deltas)
        choose_candidate = len(deltas) >= int(min_samples) and mean_delta > float(min_mean_delta)
        categories[category] = {
            "choice": "refined" if choose_candidate else "coarse",
            "n_samples": len(deltas),
            "mean_cd_delta": mean_delta,
        }
    return {
        "type": "category_mean_delta_threshold",
        "default_choice": default_choice,
        "min_samples": int(min_samples),
        "min_mean_delta": float(min_mean_delta),
        "categories": categories,
    }


def _pivot_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    pivot: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pivot[str(row["sample_id"])][str(row["method"])] = row
    return dict(pivot)


def apply_policy(
    eval_metrics_csv: str | Path,
    policy: dict[str, Any],
    *,
    baseline: str = "coarse",
    candidate: str = "refined",
) -> dict[str, Any]:
    rows = _read_csv(Path(eval_metrics_csv))
    pivot = _pivot_metrics(rows)
    per_sample = []
    selected_cd = []
    selected_fscore = []
    baseline_cd = []
    baseline_fscore = []
    candidate_count = 0
    for sample_id, methods in sorted(pivot.items()):
        if baseline not in methods or candidate not in methods:
            continue
        base = methods[baseline]
        cand = methods[candidate]
        category = str(base.get("category") or cand.get("category") or "")
        choice = policy.get("categories", {}).get(category, {}).get("choice", policy.get("default_choice", baseline))
        if choice not in {baseline, candidate}:
            choice = baseline
        chosen = cand if choice == candidate else base
        candidate_count += int(choice == candidate)
        base_cd = float(base["chamfer"])
        cand_cd = float(cand["chamfer"])
        chosen_cd = float(chosen["chamfer"])
        selected_cd.append(chosen_cd)
        selected_fscore.append(float(chosen["fscore"]))
        baseline_cd.append(base_cd)
        baseline_fscore.append(float(base["fscore"]))
        per_sample.append(
            {
                "sample_id": sample_id,
                "category": category,
                "choice": choice,
                "baseline_chamfer": base_cd,
                "candidate_chamfer": cand_cd,
                "selected_chamfer": chosen_cd,
                "selected_vs_baseline_delta": base_cd - chosen_cd,
                "baseline_fscore": float(base["fscore"]),
                "candidate_fscore": float(cand["fscore"]),
                "selected_fscore": float(chosen["fscore"]),
            }
        )
    if not per_sample:
        raise ValueError(f"No paired {baseline}/{candidate} rows found in {eval_metrics_csv}")
    mean_baseline_cd = sum(baseline_cd) / len(baseline_cd)
    mean_selected_cd = sum(selected_cd) / len(selected_cd)
    summary = {
        "n_samples": len(per_sample),
        "baseline_method": baseline,
        "candidate_method": candidate,
        "selected_candidate_count": candidate_count,
        "selected_baseline_count": len(per_sample) - candidate_count,
        "baseline_chamfer": mean_baseline_cd,
        "selected_chamfer": mean_selected_cd,
        "selected_fscore": sum(selected_fscore) / len(selected_fscore),
        "baseline_fscore": sum(baseline_fscore) / len(baseline_fscore),
        "selected_vs_baseline_cd_improvement_percent": 100.0 * (mean_baseline_cd - mean_selected_cd) / max(mean_baseline_cd, 1e-12),
    }
    return {"policy": policy, "summary": summary, "per_sample": per_sample}


def write_outputs(result: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selector_metrics.json").write_text(json.dumps({"policy": result["policy"], "summary": result["summary"]}, indent=2) + "\n", encoding="utf-8")
    _write_csv(
        out / "selector_per_sample.csv",
        result["per_sample"],
        [
            "sample_id",
            "category",
            "choice",
            "baseline_chamfer",
            "candidate_chamfer",
            "selected_chamfer",
            "selected_vs_baseline_delta",
            "baseline_fscore",
            "candidate_fscore",
            "selected_fscore",
        ],
    )
    s = result["summary"]
    lines = [
        "# Selector Baseline",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| samples | {s['n_samples']} |",
        f"| candidate chosen | {s['selected_candidate_count']} |",
        f"| baseline chosen | {s['selected_baseline_count']} |",
        f"| baseline Chamfer | {s['baseline_chamfer']:.6f} |",
        f"| selected Chamfer | {s['selected_chamfer']:.6f} |",
        f"| selected CD improvement | {s['selected_vs_baseline_cd_improvement_percent']:.4f}% |",
        f"| baseline F-score | {s['baseline_fscore']:.6f} |",
        f"| selected F-score | {s['selected_fscore']:.6f} |",
        "",
    ]
    (out / "selector_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learn-delta-csv", required=True, type=Path, help="Validation per_sample_delta.csv used to learn category choices")
    parser.add_argument("--eval-metrics-csv", required=True, type=Path, help="Evaluation per_sample_metrics.csv to score the selector")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--baseline", default="coarse")
    parser.add_argument("--candidate", default="refined")
    parser.add_argument("--default-choice", choices=["coarse", "refined"], default="refined")
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--min-mean-delta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = learn_category_policy(
        args.learn_delta_csv,
        min_samples=args.min_samples,
        min_mean_delta=args.min_mean_delta,
        default_choice=args.default_choice,
    )
    result = apply_policy(args.eval_metrics_csv, policy, baseline=args.baseline, candidate=args.candidate)
    write_outputs(result, args.out_dir)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
