#!/usr/bin/env python
"""Render best/median/worst real qualitative examples by refinement delta."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pointr_if.datasets import ManifestTripletDataset
from pointr_if.io import load_point_cloud


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:180]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _select_ranked(delta_rows: list[dict[str, Any]], per_group: int) -> list[tuple[str, dict[str, Any]]]:
    rows = sorted(delta_rows, key=lambda row: float(row["cd_delta"]))
    if not rows:
        raise ValueError("No delta rows available")
    per_group = max(1, int(per_group))
    median_delta = float(np.median([float(row["cd_delta"]) for row in rows]))
    worst = rows[:per_group]
    best = list(reversed(rows[-per_group:]))
    median = sorted(rows, key=lambda row: abs(float(row["cd_delta"]) - median_delta))[:per_group]
    selected: list[tuple[str, dict[str, Any]]] = []
    seen = set()
    for label, group in (("best", best), ("median", median), ("worst", worst)):
        for row in group:
            key = (label, row["sample_id"])
            if key not in seen:
                selected.append((label, row))
                seen.add(key)
    return selected


def _scatter(ax, points: np.ndarray, title: str, lim: float) -> None:
    pts = np.asarray(points, dtype=np.float32)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.3, alpha=0.85)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.view_init(elev=24, azim=-58)


def make_ranked_figure(
    *,
    manifest: str | Path,
    eval_dir: str | Path,
    delta_csv: str | Path,
    out: str | Path,
    per_group: int = 3,
    n_partial: int = 2048,
    n_coarse: int = 2048,
    n_gt: int = 2048,
    normalize: bool = True,
    seed: int = 200570,
) -> Path:
    eval_path = Path(eval_dir)
    selected = _select_ranked(_read_csv(Path(delta_csv)), per_group)
    dataset = ManifestTripletDataset(
        manifest,
        n_partial=n_partial,
        n_coarse=n_coarse,
        n_gt=n_gt,
        normalize=normalize,
        seed=seed,
    )
    index_by_sample = {str(row.get("sample_id") or row.get("id")): idx for idx, row in enumerate(dataset.rows)}
    rows = len(selected)
    columns = ["partial", "AdaPoinTr", "PoinTr-IF", "ground truth"]
    fig = plt.figure(figsize=(4.0 * len(columns), 2.55 * rows))
    for row_idx, (group, delta_row) in enumerate(selected):
        sample_id = str(delta_row["sample_id"])
        if sample_id not in index_by_sample:
            continue
        item = dataset[index_by_sample[sample_id]]
        refined_path = eval_path / "predictions" / f"{_safe_name(sample_id)}_refined.npy"
        refined = load_point_cloud(refined_path) if refined_path.exists() else item["coarse"].numpy()
        clouds = [
            item["partial"].numpy() if isinstance(item["partial"], torch.Tensor) else np.asarray(item["partial"]),
            item["coarse"].numpy() if isinstance(item["coarse"], torch.Tensor) else np.asarray(item["coarse"]),
            refined,
            item["gt"].numpy() if isinstance(item["gt"], torch.Tensor) else np.asarray(item["gt"]),
        ]
        lim = max(1e-6, max(float(np.abs(cloud).max()) for cloud in clouds))
        delta = float(delta_row["cd_delta"])
        pct = float(delta_row["cd_improvement_percent"])
        coarse_cd = float(delta_row["baseline_chamfer"])
        refined_cd = float(delta_row["candidate_chamfer"])
        for col_idx, (cloud, title) in enumerate(zip(clouds, columns), start=1):
            ax = fig.add_subplot(rows, len(columns), row_idx * len(columns) + col_idx, projection="3d")
            label = title
            if col_idx == 1:
                label = f"{group}: {sample_id[:34]}"
            if title == "AdaPoinTr":
                label = f"AdaPoinTr CD {coarse_cd:.5f}"
            if title == "PoinTr-IF":
                label = f"PoinTr-IF CD {refined_cd:.5f}, delta {delta:+.5f} ({pct:+.2f}%)"
            _scatter(ax, np.asarray(cloud), label, lim)
    fig.tight_layout()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--delta-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--n-partial", type=int, default=2048)
    parser.add_argument("--n-coarse", type=int, default=2048)
    parser.add_argument("--n-gt", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=200570)
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = make_ranked_figure(
        manifest=args.manifest,
        eval_dir=args.eval_dir,
        delta_csv=args.delta_csv,
        out=args.out,
        per_group=args.per_group,
        n_partial=args.n_partial,
        n_coarse=args.n_coarse,
        n_gt=args.n_gt,
        normalize=not args.no_normalize,
        seed=args.seed,
    )
    print(path)


if __name__ == "__main__":
    main()
