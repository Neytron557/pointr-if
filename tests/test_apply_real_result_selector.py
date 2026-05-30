from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from apply_real_result_selector import apply_policy, learn_category_policy


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_category_selector_learns_from_validation_and_scores_eval(tmp_path: Path) -> None:
    deltas = tmp_path / "val_delta.csv"
    _write_csv(
        deltas,
        [
            {"sample_id": "a", "category": "good", "cd_delta": 0.2},
            {"sample_id": "b", "category": "bad", "cd_delta": -0.1},
        ],
        ["sample_id", "category", "cd_delta"],
    )
    metrics = tmp_path / "test_metrics.csv"
    _write_csv(
        metrics,
        [
            {"sample_id": "x", "category": "good", "method": "coarse", "chamfer": 1.0, "fscore": 0.1},
            {"sample_id": "x", "category": "good", "method": "refined", "chamfer": 0.8, "fscore": 0.2},
            {"sample_id": "y", "category": "bad", "method": "coarse", "chamfer": 2.0, "fscore": 0.3},
            {"sample_id": "y", "category": "bad", "method": "refined", "chamfer": 3.0, "fscore": 0.1},
        ],
        ["sample_id", "category", "method", "chamfer", "fscore"],
    )

    policy = learn_category_policy(deltas, default_choice="refined")
    result = apply_policy(metrics, policy)

    assert policy["categories"]["good"]["choice"] == "refined"
    assert policy["categories"]["bad"]["choice"] == "coarse"
    assert result["summary"]["selected_candidate_count"] == 1
    assert result["summary"]["selected_chamfer"] == 1.4
