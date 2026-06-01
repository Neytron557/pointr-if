from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import CandidateBankDataset, ManifestTripletDataset
from .io import save_point_cloud
from .models import build_model_from_config
from .point_ops import chamfer_components, farthest_point_sample, fscore, to_device


def _load_checkpoint(path: str | Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _resolve_device(device: str | None, cfg: Dict[str, Any]) -> torch.device:
    requested = device or cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for evaluation, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def _as_list(value: Any, n: int, default: str = "") -> List[str]:
    if value is None:
        return [default] * n
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] * n


def _metric_row(sample_id: str, category: str, split: str, method: str, pred: torch.Tensor, gt: torch.Tensor, threshold: float) -> Dict[str, Any]:
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


def _mean_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(rows)
    numeric = ["cd_pred_to_gt", "cd_gt_to_pred", "chamfer", "precision", "recall", "fscore"]
    if not rows:
        return {k: float("nan") for k in numeric} | {"n_samples": 0}
    return {k: sum(float(r[k]) for r in rows) / len(rows) for k in numeric} | {"n_samples": len(rows)}


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:180]


def _forward_model(model: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "candidates" in batch:
        return model(
            partial=batch["partial"],
            coarse=batch["coarse"],
            candidates=batch["candidates"],
            source_ids=batch.get("source_ids"),
        )
    if "coarse_features" in batch:
        return model(batch["partial"], batch["coarse"], coarse_features=batch["coarse_features"])
    return model(batch["partial"], batch["coarse"])


def _scatter(ax, points: torch.Tensor, title: str) -> None:
    pts = points.detach().cpu().numpy()
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    lim = max(1e-6, float(abs(pts).max()))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def _save_visualization(path: Path, sample_id: str, partial: torch.Tensor, coarse: torch.Tensor, anchor: torch.Tensor, refined: torch.Tensor, gt: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 4))
    panels = [
        (partial, "partial"),
        (coarse, "AdaPoinTr"),
        (anchor, "visible anchor"),
        (refined, "PoinTr-IF"),
        (gt, "ground truth"),
    ]
    for index, (cloud, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), index, projection="3d")
        _scatter(ax, cloud, title)
    fig.suptitle(sample_id)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_category_method: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
        by_category_method[(str(row["category"]), str(row["method"]))].append(row)

    methods = {method: _mean_rows(method_rows) for method, method_rows in sorted(by_method.items())}
    coarse_cd = methods.get("coarse", {}).get("chamfer", float("nan"))
    improvements = {}
    for method in ("partial", "anchor", "refined"):
        method_cd = methods.get(method, {}).get("chamfer", float("nan"))
        improvements[f"{method}_vs_coarse_cd_percent"] = 100.0 * (coarse_cd - method_cd) / max(coarse_cd, 1e-9)

    category_rows = []
    for (category, method), group in sorted(by_category_method.items()):
        category_rows.append({"category": category, "method": method, **_mean_rows(group)})

    return {
        "overall": {
            "n_samples": max((int(v["n_samples"]) for v in methods.values()), default=0),
            "methods": methods,
            "improvements": improvements,
        },
        "category_rows": category_rows,
    }


def _write_summary_table(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "| method | n | chamfer | cd_pred_to_gt | cd_gt_to_pred | fscore | precision | recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in sorted(summary["overall"]["methods"].items()):
        lines.append(
            "| {method} | {n_samples} | {chamfer:.6f} | {cd_pred_to_gt:.6f} | "
            "{cd_gt_to_pred:.6f} | {fscore:.6f} | {precision:.6f} | {recall:.6f} |".format(
                method=method,
                **metrics,
            )
        )
    lines.append("")
    lines.append("| comparison | CD improvement vs coarse |")
    lines.append("|---|---:|")
    for key, value in sorted(summary["overall"]["improvements"].items()):
        lines.append(f"| {key} | {value:.2f}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate_manifest(
    *,
    manifest: str | Path,
    checkpoint: str | Path,
    out_dir: str | Path,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str | None = None,
    save_predictions: bool = False,
    save_visualizations: bool = False,
    max_visualizations: int = 12,
    max_samples: int | None = None,
    eval_seed: int | None = None,
    resample_mode: str | None = None,
    n_partial: int | None = None,
    n_coarse: int | None = None,
    n_gt: int | None = None,
    n_output: int | None = None,
    candidate_manifest: str | Path | None = None,
) -> Dict[str, Any]:
    ckpt = _load_checkpoint(checkpoint)
    cfg = ckpt["cfg"]
    num_threads = int(cfg.get("train", {}).get("num_threads", 4))
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    dev = _resolve_device(device, cfg)
    dcfg = cfg.get("dataset", {})
    resolved_eval_seed = int(eval_seed) if eval_seed is not None else int(cfg.get("seed", 0)) + 200000
    resolved_resample_mode = str(resample_mode or dcfg.get("resample_mode", "random")).lower()
    resolved_n_partial = n_partial if n_partial is not None else int(dcfg.get("n_partial", 2048))
    resolved_n_coarse = n_coarse if n_coarse is not None else int(dcfg.get("n_coarse", 2048))
    resolved_n_gt = n_gt if n_gt is not None else int(dcfg.get("n_gt", 2048))
    if resolved_resample_mode == "none":
        resolved_n_partial = resolved_n_coarse = resolved_n_gt = None
    dset_cls = CandidateBankDataset if candidate_manifest is not None else ManifestTripletDataset
    dset = dset_cls(
        candidate_manifest or manifest,
        n_partial=resolved_n_partial,
        n_coarse=resolved_n_coarse,
        n_gt=resolved_n_gt,
        normalize=bool(dcfg.get("normalize", True)),
        seed=resolved_eval_seed,
        resample_mode=resolved_resample_mode,
        **(
            {
                "candidate_points": dcfg.get("candidate_points"),
                "max_sources": dcfg.get("max_sources"),
            }
            if candidate_manifest is not None
            else {}
        ),
    )
    loader = DataLoader(
        dset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=dev.type == "cuda",
        drop_last=False,
    )
    model = build_model_from_config(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(dev).eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(cfg.get("eval", {}).get("fscore_threshold", 0.03))
    rows: List[Dict[str, Any]] = []
    seen = 0
    visual_count = 0
    for batch in tqdm(loader, desc=f"evaluating {Path(manifest).stem}"):
        n_batch = int(batch["gt"].shape[0])
        ids = _as_list(batch.get("sample_id") or batch.get("id"), n_batch)
        categories = _as_list(batch.get("category"), n_batch)
        splits = _as_list(batch.get("split"), n_batch)
        batch = to_device(batch, dev)
        out = _forward_model(model, batch)
        refined = out["refined"]
        resolved_n_output = int(n_output) if n_output is not None else int(refined.shape[1])
        if refined.shape[1] != resolved_n_output:
            refined = farthest_point_sample(refined, resolved_n_output)
        coarse_eval = batch["coarse"]
        if coarse_eval.shape[1] != resolved_n_output:
            coarse_eval = farthest_point_sample(coarse_eval, resolved_n_output)
        anchor = farthest_point_sample(torch.cat([batch["partial"], batch["coarse"]], dim=1), resolved_n_output)

        for b in range(n_batch):
            if max_samples is not None and seen >= max_samples:
                break
            sample_id = ids[b]
            category = categories[b]
            split = splits[b]
            clouds = {
                "partial": batch["partial"][b : b + 1],
                "coarse": coarse_eval[b : b + 1],
                "anchor": anchor[b : b + 1],
                "refined": refined[b : b + 1],
            }
            gt = batch["gt"][b : b + 1]
            for method, pred in clouds.items():
                rows.append(_metric_row(sample_id, category, split, method, pred, gt, threshold))
            if save_predictions:
                pred_dir = out_dir / "predictions"
                save_point_cloud(pred_dir / f"{_safe_name(sample_id)}_refined.npy", refined[b].detach().cpu().numpy())
                save_point_cloud(pred_dir / f"{_safe_name(sample_id)}_anchor.npy", anchor[b].detach().cpu().numpy())
            if save_visualizations and visual_count < int(max_visualizations):
                _save_visualization(
                    out_dir / "visualizations" / f"{visual_count:03d}_{_safe_name(sample_id)}.png",
                    sample_id,
                    batch["partial"][b],
                    coarse_eval[b],
                    anchor[b],
                    refined[b],
                    batch["gt"][b],
                )
                visual_count += 1
            seen += 1
        if max_samples is not None and seen >= max_samples:
            break

    metric_fields = [
        "sample_id",
        "category",
        "split",
        "method",
        "cd_pred_to_gt",
        "cd_gt_to_pred",
        "chamfer",
        "precision",
        "recall",
        "fscore",
    ]
    _write_csv(out_dir / "per_sample_metrics.csv", rows, metric_fields)
    summary = _summarize(rows)
    category_fields = ["category", "method", "n_samples", "cd_pred_to_gt", "cd_gt_to_pred", "chamfer", "precision", "recall", "fscore"]
    _write_csv(out_dir / "category_metrics.csv", summary["category_rows"], category_fields)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary["overall"], f, indent=2)
    with (out_dir / "category_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary["category_rows"], f, indent=2)
    _write_summary_table(out_dir / "summary_table.md", summary)
    with (out_dir / "checkpoint_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    command = " ".join(shlex.quote(part) for part in [sys.executable, "-m", "pointr_if.evaluate", *sys.argv[1:]])
    (out_dir / "command_log.txt").write_text(
        json.dumps(
            {
                "command": command,
                "manifest": str(manifest),
                "checkpoint": str(checkpoint),
                "device": str(dev),
                "eval_seed": resolved_eval_seed,
                "resample_mode": resolved_resample_mode,
                "n_partial": resolved_n_partial,
                "n_coarse": resolved_n_coarse,
                "n_gt": resolved_n_gt,
                "n_output": n_output,
                "candidate_manifest": str(candidate_manifest) if candidate_manifest is not None else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PoinTr-IF on a real triplet manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", default="outputs/eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--max-visualizations", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Seed for deterministic eval point resampling. Defaults to checkpoint seed + 200000.",
    )
    parser.add_argument("--resample-mode", choices=["random", "fps", "none"], default=None)
    parser.add_argument("--n-partial", type=int, default=None)
    parser.add_argument("--n-coarse", type=int, default=None)
    parser.add_argument("--n-gt", type=int, default=None)
    parser.add_argument("--n-output", type=int, default=None)
    parser.add_argument("--candidate-manifest", default=None)
    args = parser.parse_args()

    summary = evaluate_manifest(
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        save_predictions=args.save_predictions,
        save_visualizations=args.save_visualizations,
        max_visualizations=args.max_visualizations,
        max_samples=args.max_samples,
        eval_seed=args.eval_seed,
        resample_mode=args.resample_mode,
        n_partial=args.n_partial,
        n_coarse=args.n_coarse,
        n_gt=args.n_gt,
        n_output=args.n_output,
        candidate_manifest=args.candidate_manifest,
    )
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
