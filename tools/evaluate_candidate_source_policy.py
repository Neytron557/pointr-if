#!/usr/bin/env python
"""Evaluate a validation-learned candidate source policy on held-out metrics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


METRIC_FIELDS = ("cd_pred_to_gt", "cd_gt_to_pred", "chamfer", "precision", "recall", "fscore")


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    for row in rows:
        for key in METRIC_FIELDS:
            if key in row and row[key] != "":
                row[key] = float(row[key])
        if "n_samples" in row and row["n_samples"] != "":
            row["n_samples"] = int(row["n_samples"])
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _allowed(method: str, exclude: set[str]) -> bool:
    if method in exclude:
        return False
    return "oracle" not in method.lower()


def learn_category_policy(
    val_category_rows: List[Dict[str, Any]],
    *,
    baseline_method: str,
    min_val_samples: int,
    exclude_methods: set[str],
) -> Dict[str, str]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in val_category_rows:
        method = str(row["method"])
        if _allowed(method, exclude_methods):
            by_category[str(row["category"])].append(row)

    policy: Dict[str, str] = {}
    for category, rows in by_category.items():
        baseline_rows = [row for row in rows if row["method"] == baseline_method]
        if not baseline_rows or int(baseline_rows[0].get("n_samples", 0)) < min_val_samples:
            policy[category] = baseline_method
            continue
        best = min(rows, key=lambda row: float(row["chamfer"]))
        policy[category] = str(best["method"])
    return policy


def apply_policy(
    test_rows: List[Dict[str, Any]],
    policy: Dict[str, str],
    *,
    baseline_method: str,
) -> Dict[str, Any]:
    by_sample: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in test_rows:
        by_sample[str(row["sample_id"])][str(row["method"])] = row

    selected_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    for sample_id, method_rows in sorted(by_sample.items()):
        if baseline_method not in method_rows:
            raise ValueError(f"Sample {sample_id!r} is missing baseline method {baseline_method!r}")
        baseline = method_rows[baseline_method]
        category = str(baseline["category"])
        wanted = policy.get(category, baseline_method)
        chosen_method = wanted if wanted in method_rows else baseline_method
        chosen = method_rows[chosen_method]
        selected_rows.append(
            {
                "sample_id": sample_id,
                "category": category,
                "split": chosen["split"],
                "chosen_method": chosen_method,
                "fallback_used": chosen_method != wanted,
                "baseline_chamfer": float(baseline["chamfer"]),
                "chosen_chamfer": float(chosen["chamfer"]),
                "cd_delta": float(baseline["chamfer"]) - float(chosen["chamfer"]),
                "baseline_fscore": float(baseline["fscore"]),
                "chosen_fscore": float(chosen["fscore"]),
                "fscore_delta": float(chosen["fscore"]) - float(baseline["fscore"]),
            }
        )
        baseline_rows.append(baseline)

    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_category[str(row["category"])].append(row)

    category_rows = []
    for category, rows in sorted(by_category.items()):
        base_cd = _mean(float(row["baseline_chamfer"]) for row in rows)
        chosen_cd = _mean(float(row["chosen_chamfer"]) for row in rows)
        category_rows.append(
            {
                "category": category,
                "n_samples": len(rows),
                "policy_method": policy.get(category, baseline_method),
                "baseline_chamfer": base_cd,
                "chosen_chamfer": chosen_cd,
                "cd_improvement_percent": 100.0 * (base_cd - chosen_cd) / max(base_cd, 1e-12),
                "fscore_delta": _mean(float(row["fscore_delta"]) for row in rows),
                "positive_count": sum(1 for row in rows if float(row["cd_delta"]) > 1e-12),
                "negative_count": sum(1 for row in rows if float(row["cd_delta"]) < -1e-12),
                "zero_count": sum(1 for row in rows if abs(float(row["cd_delta"])) <= 1e-12),
            }
        )

    baseline_cd = _mean(float(row["chamfer"]) for row in baseline_rows)
    chosen_cd = _mean(float(row["chosen_chamfer"]) for row in selected_rows)
    baseline_f = _mean(float(row["fscore"]) for row in baseline_rows)
    chosen_f = _mean(float(row["chosen_fscore"]) for row in selected_rows)
    return {
        "summary": {
            "n_samples": len(selected_rows),
            "baseline_method": baseline_method,
            "baseline_chamfer": baseline_cd,
            "policy_chamfer": chosen_cd,
            "cd_improvement_percent": 100.0 * (baseline_cd - chosen_cd) / max(baseline_cd, 1e-12),
            "baseline_fscore": baseline_f,
            "policy_fscore": chosen_f,
            "fscore_delta": chosen_f - baseline_f,
            "positive_count": sum(1 for row in selected_rows if float(row["cd_delta"]) > 1e-12),
            "negative_count": sum(1 for row in selected_rows if float(row["cd_delta"]) < -1e-12),
            "zero_count": sum(1 for row in selected_rows if abs(float(row["cd_delta"])) <= 1e-12),
        },
        "per_sample": selected_rows,
        "per_category": category_rows,
    }


def _markdown(result: Dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Candidate Source Policy",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| samples | {summary['n_samples']} |",
        f"| baseline Chamfer | {summary['baseline_chamfer']:.6f} |",
        f"| policy Chamfer | {summary['policy_chamfer']:.6f} |",
        f"| CD improvement | {summary['cd_improvement_percent']:.4f}% |",
        f"| baseline F-score | {summary['baseline_fscore']:.6f} |",
        f"| policy F-score | {summary['policy_fscore']:.6f} |",
        f"| F-score delta | {summary['fscore_delta']:.8f} |",
        f"| positive / negative / zero | {summary['positive_count']} / {summary['negative_count']} / {summary['zero_count']} |",
        "",
        "## Per Category",
        "",
        "| category | n | policy | baseline CD | policy CD | CD gain | positive | negative | zero |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_category"]:
        lines.append(
            f"| {row['category']} | {row['n_samples']} | {row['policy_method']} | "
            f"{row['baseline_chamfer']:.6f} | {row['chosen_chamfer']:.6f} | "
            f"{row['cd_improvement_percent']:.4f}% | {row['positive_count']} | "
            f"{row['negative_count']} | {row['zero_count']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-category-metrics", type=Path, required=True)
    parser.add_argument("--test-per-sample-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-method", default="adapointr_coarse")
    parser.add_argument("--min-val-samples", type=int, default=3)
    parser.add_argument(
        "--exclude-method",
        action="append",
        default=[],
        help="Method to exclude from the deployable policy. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude = set(args.exclude_method)
    val_rows = _read_csv(args.val_category_metrics)
    test_rows = _read_csv(args.test_per_sample_metrics)
    policy = learn_category_policy(
        val_rows,
        baseline_method=args.baseline_method,
        min_val_samples=args.min_val_samples,
        exclude_methods=exclude,
    )
    result = apply_policy(test_rows, policy, baseline_method=args.baseline_method)
    payload = {
        "policy": policy,
        "inputs": {
            "val_category_metrics": str(args.val_category_metrics),
            "test_per_sample_metrics": str(args.test_per_sample_metrics),
            "baseline_method": args.baseline_method,
            "min_val_samples": args.min_val_samples,
            "exclude_methods": sorted(exclude),
        },
        **{k: v for k, v in result.items() if k != "per_sample"},
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_policy_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "source_policy_summary.md").write_text(_markdown(result), encoding="utf-8")
    _write_csv(
        out / "source_policy_per_sample.csv",
        result["per_sample"],
        [
            "sample_id",
            "category",
            "split",
            "chosen_method",
            "fallback_used",
            "baseline_chamfer",
            "chosen_chamfer",
            "cd_delta",
            "baseline_fscore",
            "chosen_fscore",
            "fscore_delta",
        ],
    )
    _write_csv(
        out / "source_policy_per_category.csv",
        result["per_category"],
        [
            "category",
            "n_samples",
            "policy_method",
            "baseline_chamfer",
            "chosen_chamfer",
            "cd_improvement_percent",
            "fscore_delta",
            "positive_count",
            "negative_count",
            "zero_count",
        ],
    )
    print(_markdown(result))


if __name__ == "__main__":
    main()
