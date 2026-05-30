#!/usr/bin/env python
"""Paired statistical analysis for real PoinTr-IF evaluation outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List

import numpy as np


NUMERIC_FIELDS = ("chamfer", "fscore", "cd_pred_to_gt", "cd_gt_to_pred", "precision", "recall")


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"sample_id", "category", "split", "method", "chamfer", "fscore"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    for row in rows:
        for field in NUMERIC_FIELDS:
            if field in row and row[field] != "":
                row[field] = float(row[field])
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else math.nan


def _summarize_methods(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    summary = {}
    for method, method_rows in sorted(by_method.items()):
        summary[method] = {
            "n_samples": len({str(row["sample_id"]) for row in method_rows}),
            "chamfer": _mean(float(row["chamfer"]) for row in method_rows),
            "fscore": _mean(float(row["fscore"]) for row in method_rows),
        }
    return summary


def _pivot_by_sample(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    pivot: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        sample_id = str(row["sample_id"])
        method = str(row["method"])
        if method in pivot[sample_id]:
            raise ValueError(f"Duplicate method row for sample={sample_id!r}, method={method!r}")
        pivot[sample_id][method] = row
    return dict(pivot)


def _paired_test_stats(baseline_values: np.ndarray, candidate_values: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scipy_available": False,
        "paired_ttest_pvalue": None,
        "paired_ttest_statistic": None,
        "wilcoxon_pvalue": None,
        "wilcoxon_statistic": None,
        "test_note": "",
    }
    try:
        from scipy import stats  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        out["test_note"] = f"scipy unavailable: {exc}"
        return out

    out["scipy_available"] = True
    ttest = stats.ttest_rel(baseline_values, candidate_values)
    out["paired_ttest_statistic"] = float(ttest.statistic)
    out["paired_ttest_pvalue"] = float(ttest.pvalue)
    try:
        wilcoxon = stats.wilcoxon(baseline_values, candidate_values, zero_method="wilcox", alternative="two-sided")
        out["wilcoxon_statistic"] = float(wilcoxon.statistic)
        out["wilcoxon_pvalue"] = float(wilcoxon.pvalue)
    except ValueError as exc:
        out["test_note"] = f"wilcoxon unavailable for these deltas: {exc}"
    return out


def _bootstrap_improvement(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    if n_bootstrap <= 0:
        return {
            "n_bootstrap": 0,
            "ci95_cd_improvement_percent": [math.nan, math.nan],
            "ci95_mean_cd_delta": [math.nan, math.nan],
        }
    rng = np.random.default_rng(seed)
    n = len(baseline_values)
    improvement = np.empty(n_bootstrap, dtype=np.float64)
    delta = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        base = baseline_values[idx]
        cand = candidate_values[idx]
        delta[i] = float(np.mean(base - cand))
        improvement[i] = 100.0 * float(np.mean(base) - np.mean(cand)) / max(float(np.mean(base)), 1e-12)
    return {
        "n_bootstrap": int(n_bootstrap),
        "ci95_cd_improvement_percent": [float(np.percentile(improvement, 2.5)), float(np.percentile(improvement, 97.5))],
        "ci95_mean_cd_delta": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
    }


def _build_per_sample_delta(
    pivot: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    baseline: str,
    candidate: str,
) -> List[Dict[str, Any]]:
    rows = []
    for sample_id, methods in sorted(pivot.items()):
        if baseline not in methods or candidate not in methods:
            continue
        base = methods[baseline]
        cand = methods[candidate]
        base_cd = float(base["chamfer"])
        cand_cd = float(cand["chamfer"])
        delta = base_cd - cand_cd
        if abs(delta) <= 1e-12:
            outcome = "same"
        elif delta > 0:
            outcome = "positive"
        else:
            outcome = "negative"
        rows.append(
            {
                "sample_id": sample_id,
                "category": cand.get("category") or base.get("category", ""),
                "split": cand.get("split") or base.get("split", ""),
                "baseline_method": baseline,
                "candidate_method": candidate,
                "baseline_chamfer": base_cd,
                "candidate_chamfer": cand_cd,
                "cd_delta": delta,
                "cd_improvement_percent": 100.0 * delta / max(base_cd, 1e-12),
                "baseline_fscore": float(base["fscore"]),
                "candidate_fscore": float(cand["fscore"]),
                "fscore_delta": float(cand["fscore"]) - float(base["fscore"]),
                "outcome": outcome,
            }
        )
    if not rows:
        raise ValueError(f"No paired samples found for baseline={baseline!r}, candidate={candidate!r}")
    return rows


def _category_rows(per_sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        by_category[str(row["category"])].append(row)
    out = []
    for category, rows in sorted(by_category.items()):
        positives = sum(1 for row in rows if row["outcome"] == "positive")
        negatives = sum(1 for row in rows if row["outcome"] == "negative")
        zeros = sum(1 for row in rows if row["outcome"] == "same")
        base_cd = _mean(float(row["baseline_chamfer"]) for row in rows)
        cand_cd = _mean(float(row["candidate_chamfer"]) for row in rows)
        out.append(
            {
                "category": category,
                "n_samples": len(rows),
                "baseline_chamfer": base_cd,
                "candidate_chamfer": cand_cd,
                "mean_cd_delta": _mean(float(row["cd_delta"]) for row in rows),
                "median_cd_delta": float(median(float(row["cd_delta"]) for row in rows)),
                "mean_cd_improvement_percent": 100.0 * (base_cd - cand_cd) / max(base_cd, 1e-12),
                "mean_fscore_delta": _mean(float(row["fscore_delta"]) for row in rows),
                "positive_count": positives,
                "negative_count": negatives,
                "zero_count": zeros,
            }
        )
    return out


def _oracle_summary(per_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    chosen_cd = []
    chosen_fscore = []
    refined_chosen = 0
    for row in per_sample:
        if float(row["candidate_chamfer"]) < float(row["baseline_chamfer"]):
            chosen_cd.append(float(row["candidate_chamfer"]))
            chosen_fscore.append(float(row["candidate_fscore"]))
            refined_chosen += 1
        else:
            chosen_cd.append(float(row["baseline_chamfer"]))
            chosen_fscore.append(float(row["baseline_fscore"]))
    return {
        "label": "oracle_non_deployable_choose_best_per_sample_cd",
        "n_samples": len(per_sample),
        "chamfer": _mean(chosen_cd),
        "fscore": _mean(chosen_fscore),
        "candidate_chosen_count": refined_chosen,
        "baseline_chosen_count": len(per_sample) - refined_chosen,
    }


def analyze_metrics_csv(
    metrics_csv: str | Path,
    *,
    baseline: str = "coarse",
    candidate: str = "refined",
    bootstrap: int = 5000,
    seed: int = 570,
) -> Dict[str, Any]:
    path = Path(metrics_csv)
    rows = _read_rows(path)
    pivot = _pivot_by_sample(rows)
    per_sample = _build_per_sample_delta(pivot, baseline=baseline, candidate=candidate)

    baseline_values = np.asarray([float(row["baseline_chamfer"]) for row in per_sample], dtype=np.float64)
    candidate_values = np.asarray([float(row["candidate_chamfer"]) for row in per_sample], dtype=np.float64)
    deltas = baseline_values - candidate_values

    positive = int(np.sum(deltas > 1e-12))
    negative = int(np.sum(deltas < -1e-12))
    zero = int(len(deltas) - positive - negative)
    mean_base = float(np.mean(baseline_values))
    mean_candidate = float(np.mean(candidate_values))
    paired = {
        "baseline_method": baseline,
        "candidate_method": candidate,
        "n_samples": len(per_sample),
        "baseline_chamfer": mean_base,
        "candidate_chamfer": mean_candidate,
        "mean_cd_delta": float(np.mean(deltas)),
        "median_cd_delta": float(np.median(deltas)),
        "mean_cd_improvement_percent": 100.0 * (mean_base - mean_candidate) / max(mean_base, 1e-12),
        "median_sample_cd_improvement_percent": float(np.median([float(row["cd_improvement_percent"]) for row in per_sample])),
        "mean_fscore_delta": _mean(float(row["fscore_delta"]) for row in per_sample),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "positive_fraction": positive / len(per_sample),
    }
    boot = _bootstrap_improvement(baseline_values, candidate_values, n_bootstrap=bootstrap, seed=seed)
    paired["bootstrap_ci95_cd_improvement_percent"] = boot["ci95_cd_improvement_percent"]
    paired["bootstrap_ci95_mean_cd_delta"] = boot["ci95_mean_cd_delta"]
    paired["n_bootstrap"] = boot["n_bootstrap"]
    paired.update(_paired_test_stats(baseline_values, candidate_values))

    return {
        "source": str(path),
        "aggregate_methods": _summarize_methods(rows),
        "paired": paired,
        "oracle": _oracle_summary(per_sample),
        "per_category": _category_rows(per_sample),
        "per_sample_delta": per_sample,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(result: Dict[str, Any]) -> str:
    paired = result["paired"]
    ci = paired["bootstrap_ci95_cd_improvement_percent"]
    lines = [
        "# Paired Real-Result Statistics",
        "",
        f"Source: `{result['source']}`",
        "",
        "## Aggregate Methods",
        "",
        "| method | n | chamfer | fscore |",
        "|---|---:|---:|---:|",
    ]
    for method, values in result["aggregate_methods"].items():
        lines.append(f"| {method} | {int(values['n_samples'])} | {values['chamfer']:.6f} | {values['fscore']:.6f} |")
    lines.extend(
        [
            "",
            "## Paired Delta",
            "",
            f"Baseline: `{paired['baseline_method']}`",
            f"Candidate: `{paired['candidate_method']}`",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| paired samples | {paired['n_samples']} |",
            f"| baseline Chamfer | {paired['baseline_chamfer']:.6f} |",
            f"| candidate Chamfer | {paired['candidate_chamfer']:.6f} |",
            f"| mean CD delta | {paired['mean_cd_delta']:.8f} |",
            f"| median CD delta | {paired['median_cd_delta']:.8f} |",
            f"| mean CD improvement | {paired['mean_cd_improvement_percent']:.4f}% |",
            f"| bootstrap 95% CI | [{ci[0]:.4f}%, {ci[1]:.4f}%] |",
            f"| positive / negative / zero samples | {paired['positive_count']} / {paired['negative_count']} / {paired['zero_count']} |",
            f"| positive fraction | {paired['positive_fraction']:.4f} |",
            f"| mean F-score delta | {paired['mean_fscore_delta']:.8f} |",
            f"| paired t-test p-value | {paired['paired_ttest_pvalue']} |",
            f"| Wilcoxon p-value | {paired['wilcoxon_pvalue']} |",
            "",
            "## Oracle Upper Bound",
            "",
            "This row is non-deployable because it chooses the better of baseline and candidate using ground truth per sample.",
            "",
            "| label | n | chamfer | fscore | candidate chosen | baseline chosen |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    oracle = result["oracle"]
    lines.append(
        f"| Oracle | {oracle['n_samples']} | {oracle['chamfer']:.6f} | {oracle['fscore']:.6f} | "
        f"{oracle['candidate_chosen_count']} | {oracle['baseline_chosen_count']} |"
    )
    lines.extend(
        [
            "",
            "## Per-Category Improvement",
            "",
            "| category | n | baseline CD | candidate CD | mean CD delta | improvement | positive | negative | zero |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["per_category"]:
        lines.append(
            f"| {row['category']} | {row['n_samples']} | {row['baseline_chamfer']:.6f} | "
            f"{row['candidate_chamfer']:.6f} | {row['mean_cd_delta']:.8f} | "
            f"{row['mean_cd_improvement_percent']:.4f}% | {row['positive_count']} | {row['negative_count']} | {row['zero_count']} |"
        )
    return "\n".join(lines) + "\n"


def write_stats_outputs(result: Dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_payload = {k: v for k, v in result.items() if k != "per_sample_delta"}
    (out / "paired_stats.json").write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
    (out / "paired_stats.md").write_text(_markdown(result), encoding="utf-8")
    _write_csv(
        out / "per_sample_delta.csv",
        result["per_sample_delta"],
        [
            "sample_id",
            "category",
            "split",
            "baseline_method",
            "candidate_method",
            "baseline_chamfer",
            "candidate_chamfer",
            "cd_delta",
            "cd_improvement_percent",
            "baseline_fscore",
            "candidate_fscore",
            "fscore_delta",
            "outcome",
        ],
    )
    _write_csv(
        out / "per_category_improvement.csv",
        result["per_category"],
        [
            "category",
            "n_samples",
            "baseline_chamfer",
            "candidate_chamfer",
            "mean_cd_delta",
            "median_cd_delta",
            "mean_cd_improvement_percent",
            "mean_fscore_delta",
            "positive_count",
            "negative_count",
            "zero_count",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", type=Path, help="per_sample_metrics.csv from pointr_if.evaluate")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="coarse")
    parser.add_argument("--candidate", default="refined")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=570)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_metrics_csv(
        args.metrics_csv,
        baseline=args.baseline,
        candidate=args.candidate,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    write_stats_outputs(result, args.out_dir)
    print(_markdown(result))


if __name__ == "__main__":
    main()
