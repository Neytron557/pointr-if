#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from analyze_real_results_stats import analyze_metrics_csv, write_stats_outputs
from pointr_if.io import load_point_cloud, read_manifest
from pointr_if.point_ops import chamfer_components, farthest_point_sample, fscore, resample_points_np


def _normalize_like_gt(points: np.ndarray, gt: np.ndarray) -> np.ndarray:
    center = gt.mean(axis=0, keepdims=True)
    scale = np.sqrt(((gt - center) ** 2).sum(axis=1)).max().clip(1e-8)
    return ((points - center) / scale).astype(np.float32)


def _metric_row(sample_id: str, category: str, split: str, method: str, pred: torch.Tensor, gt: torch.Tensor, threshold: float) -> dict[str, Any]:
    comps = chamfer_components(pred, gt)
    fs = fscore(pred, gt, threshold=threshold)
    return {
        "sample_id": sample_id,
        "category": category,
        "split": split,
        "method": method,
        "cd_pred_to_gt": comps["cd_pred_to_gt"],
        "cd_gt_to_pred": comps["cd_gt_to_pred"],
        "chamfer": comps["cd"],
        "precision": fs["precision"],
        "recall": fs["recall"],
        "fscore": fs["fscore"],
    }


def _mean_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["cd_pred_to_gt", "cd_gt_to_pred", "chamfer", "precision", "recall", "fscore"]
    if not rows:
        return {key: float("nan") for key in keys} | {"n_samples": 0}
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys} | {"n_samples": len(rows)}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _point_oracle_cloud(candidates: torch.Tensor, gt: torch.Tensor, max_points: int) -> torch.Tensor:
    flat = candidates.reshape(1, -1, 3)
    if max_points > 0 and flat.shape[1] > max_points:
        flat = farthest_point_sample(flat, max_points)
    d = torch.cdist(gt, flat, p=2)
    idx = d.argmin(dim=2)
    return flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_manifest(args.candidate_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for candidate oracle but unavailable")
    rng = np.random.default_rng(args.seed)
    threshold = float(args.fscore_threshold)
    metric_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    point_oracle_rows: list[dict[str, Any]] = []
    source_wins: Counter[str] = Counter()

    for row in tqdm(rows, desc="candidate oracle"):
        sample_id = row.get("sample_id") or row.get("id")
        category = row.get("category", "")
        split = row.get("split", "")
        gt_raw = load_point_cloud(row["gt"])
        gt_np = _normalize_like_gt(gt_raw, gt_raw)
        gt_np = resample_points_np(gt_np, None if args.resample_mode == "none" else args.n_gt, rng, mode=args.resample_mode)
        coarse_np = _normalize_like_gt(load_point_cloud(row["coarse"]), gt_raw)
        coarse_np = resample_points_np(coarse_np, None if args.resample_mode == "none" else args.n_gt, rng, mode=args.resample_mode)
        payload = np.load(row["candidate_npz"], allow_pickle=True)
        candidates_np = np.asarray(payload["candidates"], dtype=np.float32)
        source_names = [str(x) for x in payload["source_names"].tolist()]

        gt = torch.from_numpy(gt_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        coarse = torch.from_numpy(coarse_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        coarse_metrics = _metric_row(sample_id, category, split, "adapointr_coarse", coarse, gt, threshold)
        metric_rows.append(coarse_metrics)

        candidate_scores = []
        candidate_tensors = []
        for name, candidate_np in zip(source_names, candidates_np):
            pred_np = resample_points_np(candidate_np, None if args.resample_mode == "none" else args.n_output, rng, mode=args.resample_mode)
            pred = torch.from_numpy(pred_np).to(device=device, dtype=torch.float32).unsqueeze(0)
            row_metrics = _metric_row(sample_id, category, split, name, pred, gt, threshold)
            metric_rows.append(row_metrics)
            candidate_scores.append((name, float(row_metrics["chamfer"]), row_metrics))
            candidate_tensors.append(pred)

        concat = torch.cat(candidate_tensors, dim=1)
        fused = farthest_point_sample(concat, int(args.n_output))
        fused_metrics = _metric_row(sample_id, category, split, "all_candidates_fps", fused, gt, threshold)
        metric_rows.append(fused_metrics)
        candidate_scores.append(("all_candidates_fps", float(fused_metrics["chamfer"]), fused_metrics))

        best_name, _best_cd, best_metrics = min(
            candidate_scores + [("adapointr_coarse", float(coarse_metrics["chamfer"]), coarse_metrics)],
            key=lambda item: item[1],
        )
        source_wins[best_name] += 1
        oracle_rows.append({**best_metrics, "method": "sample_oracle"})

        if int(args.point_oracle_points) != 0:
            point_oracle = _point_oracle_cloud(torch.cat(candidate_tensors, dim=1), gt, max_points=int(args.point_oracle_points))
            if point_oracle.shape[1] != int(args.n_output):
                point_oracle = farthest_point_sample(point_oracle, int(args.n_output))
            point_oracle_rows.append(_metric_row(sample_id, category, split, "point_oracle_approx", point_oracle, gt, threshold))

    all_rows = metric_rows + oracle_rows + point_oracle_rows
    fields = ["sample_id", "category", "split", "method", "cd_pred_to_gt", "cd_gt_to_pred", "chamfer", "precision", "recall", "fscore"]
    _write_csv(out_dir / "per_sample_metrics.csv", all_rows, fields)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_method[str(row["method"])].append(row)
        by_category_method[(str(row["category"]), str(row["method"]))].append(row)
    methods = {method: _mean_rows(group) for method, group in sorted(by_method.items())}
    category_rows = [{"category": cat, "method": method, **_mean_rows(group)} for (cat, method), group in sorted(by_category_method.items())]
    _write_csv(
        out_dir / "category_metrics.csv",
        category_rows,
        ["category", "method", "n_samples", "cd_pred_to_gt", "cd_gt_to_pred", "chamfer", "precision", "recall", "fscore"],
    )
    baseline_cd = methods["adapointr_coarse"]["chamfer"]
    summary = {
        "overall": {
            "n_samples": len(rows),
            "methods": methods,
            "improvements_vs_adapointr_percent": {
                method: 100.0 * (baseline_cd - metrics["chamfer"]) / max(baseline_cd, 1e-9)
                for method, metrics in methods.items()
            },
        },
        "oracle": {
            "sample_oracle": methods.get("sample_oracle", {}),
            "point_oracle_approx": methods.get("point_oracle_approx", {}),
            "source_wins": dict(source_wins),
        },
        "protocol": {
            "resample_mode": args.resample_mode,
            "n_gt": args.n_gt,
            "n_output": args.n_output,
            "fscore_threshold": args.fscore_threshold,
            "point_oracle_points": args.point_oracle_points,
        },
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    (out_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(summary["overall"], indent=2) + "\n", encoding="utf-8")
    try:
        stats = analyze_metrics_csv(out_dir / "per_sample_metrics.csv", baseline="adapointr_coarse", candidate="sample_oracle", bootstrap=1000, seed=args.seed)
        write_stats_outputs(stats, out_dir / "stats")
    except Exception as exc:  # noqa: BLE001
        (out_dir / "stats_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
    (out_dir / "command_log.txt").write_text(json.dumps({"command": summary["command"], **summary["protocol"]}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate-bank baselines and non-deployable oracle.")
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resample-mode", choices=["random", "fps", "none"], default="fps")
    parser.add_argument("--n-gt", type=int, default=4096)
    parser.add_argument("--n-output", type=int, default=4096)
    parser.add_argument("--fscore-threshold", type=float, default=0.02)
    parser.add_argument("--point-oracle-points", type=int, default=1024, help="Downsample concatenated bank before approximate point oracle; set 0 to skip.")
    parser.add_argument("--seed", type=int, default=200570)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
