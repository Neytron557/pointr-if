from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from analyze_real_results_stats import analyze_metrics_csv, write_stats_outputs


def _write_metrics(path: Path) -> None:
    rows = [
        ("a", "cat1", "test", "coarse", 1.00, 0.10),
        ("a", "cat1", "test", "refined", 0.80, 0.20),
        ("a", "cat1", "test", "partial", 2.00, 0.05),
        ("b", "cat1", "test", "coarse", 1.00, 0.10),
        ("b", "cat1", "test", "refined", 1.20, 0.05),
        ("b", "cat1", "test", "partial", 2.10, 0.04),
        ("c", "cat2", "test", "coarse", 2.00, 0.30),
        ("c", "cat2", "test", "refined", 1.00, 0.50),
        ("c", "cat2", "test", "partial", 3.00, 0.02),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "category", "split", "method", "chamfer", "fscore"],
        )
        writer.writeheader()
        for sample_id, category, split, method, chamfer, fscore in rows:
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "split": split,
                    "method": method,
                    "chamfer": chamfer,
                    "fscore": fscore,
                }
            )


def test_analyze_metrics_csv_computes_paired_deltas(tmp_path: Path) -> None:
    metrics = tmp_path / "per_sample_metrics.csv"
    _write_metrics(metrics)

    result = analyze_metrics_csv(metrics, baseline="coarse", candidate="refined", bootstrap=200, seed=7)

    paired = result["paired"]
    assert paired["n_samples"] == 3
    assert paired["positive_count"] == 2
    assert paired["negative_count"] == 1
    assert paired["zero_count"] == 0
    assert paired["mean_cd_delta"] == (1.0 + 1.0 + 2.0 - 0.8 - 1.2 - 1.0) / 3
    assert paired["median_cd_delta"] == pytest.approx(0.2)
    assert paired["mean_cd_improvement_percent"] > 0
    assert paired["bootstrap_ci95_cd_improvement_percent"][0] <= paired["bootstrap_ci95_cd_improvement_percent"][1]

    oracle = result["oracle"]
    assert oracle["label"] == "oracle_non_deployable_choose_best_per_sample_cd"
    assert oracle["chamfer"] == (0.8 + 1.0 + 1.0) / 3


def test_write_stats_outputs_creates_expected_files(tmp_path: Path) -> None:
    metrics = tmp_path / "per_sample_metrics.csv"
    out_dir = tmp_path / "stats"
    _write_metrics(metrics)
    result = analyze_metrics_csv(metrics, baseline="coarse", candidate="refined", bootstrap=50, seed=9)

    write_stats_outputs(result, out_dir)

    assert (out_dir / "paired_stats.json").exists()
    assert (out_dir / "paired_stats.md").exists()
    assert (out_dir / "per_category_improvement.csv").exists()
    assert (out_dir / "per_sample_delta.csv").exists()

    payload = json.loads((out_dir / "paired_stats.json").read_text(encoding="utf-8"))
    assert payload["paired"]["positive_count"] == 2
    assert "cat1" in (out_dir / "per_category_improvement.csv").read_text(encoding="utf-8")
    assert "Oracle" in (out_dir / "paired_stats.md").read_text(encoding="utf-8")
