from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .datasets import make_dataset
from .models import build_model_from_config
from .point_ops import (
    chamfer_components,
    chamfer_distance,
    chamfer_distance_per_sample,
    fscore,
    nearest_delta,
    pairwise_dist,
    partial_preservation_loss,
    repulsion_loss,
    sample_occupancy_queries,
    set_seed,
    to_device,
)


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def save_config(cfg: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def make_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    train_set = make_dataset(cfg, split="train")
    val_set = make_dataset(cfg, split="val")
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg.get("val_batch_size", batch_size)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def compute_losses(model, batch: Dict, cfg: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
    if "candidates" in batch:
        return compute_seed_losses(model, batch, cfg)

    loss_cfg = cfg.get("loss", {})
    lambda_occ = float(loss_cfg.get("lambda_occ", 0.2))
    lambda_repulsion = float(loss_cfg.get("lambda_repulsion", 0.0))
    lambda_gate_bce = float(loss_cfg.get("lambda_gate_bce", 0.0))
    lambda_gate_sparsity = float(loss_cfg.get("lambda_gate_sparsity", 0.0))
    lambda_delta_l2 = float(loss_cfg.get("lambda_delta_l2", 0.0))
    partial = batch["partial"]
    coarse = batch["coarse"]
    gt = batch["gt"]
    qcfg = cfg.get("queries", {})
    query, occ_labels = None, None
    if lambda_occ != 0.0:
        query, occ_labels = sample_occupancy_queries(
            gt,
            n_query=int(qcfg.get("n_query", 512)),
            threshold=float(qcfg.get("threshold", 0.035)),
            near_ratio=float(qcfg.get("near_ratio", 0.65)),
            near_std=float(qcfg.get("near_std", 0.035)),
            box_scale=float(qcfg.get("box_scale", 1.25)),
        )
    out = model(partial=partial, coarse=coarse, query=query)
    refined = out["refined"]

    loss_cd = chamfer_distance(refined, gt, squared=bool(loss_cfg.get("squared_cd", False)))
    if occ_labels is not None:
        loss_occ = F.binary_cross_entropy_with_logits(out["occ_logits"], occ_labels)
    else:
        loss_occ = refined.new_tensor(0.0)

    target_delta = nearest_delta(coarse, gt).detach()
    loss_delta = F.smooth_l1_loss(out["coarse_delta"], target_delta)
    loss_partial = partial_preservation_loss(refined, partial)
    if lambda_repulsion != 0.0:
        loss_repulse = repulsion_loss(
            refined,
            k=int(loss_cfg.get("repulsion_k", 5)),
            radius=float(loss_cfg.get("repulsion_radius", 0.025)),
        )
    else:
        loss_repulse = refined.new_tensor(0.0)
    raw_delta = out.get("raw_delta", out["coarse_delta"])
    loss_delta_l2 = raw_delta.square().mean()
    if "gate_logits" in out:
        with torch.no_grad():
            coarse_to_gt = pairwise_dist(coarse, gt).min(dim=2).values
            refined_to_gt = pairwise_dist(refined, gt).min(dim=2).values
            gate_margin = float(loss_cfg.get("gate_margin", 0.0))
            gate_target = (refined_to_gt + gate_margin < coarse_to_gt).float()
        loss_gate_bce = F.binary_cross_entropy_with_logits(out["gate_logits"], gate_target)
        loss_gate_sparsity = out["gate"].mean()
        gate_mean = float(out["gate"].detach().mean().cpu())
        gate_positive_target = float(gate_target.detach().mean().cpu())
    else:
        loss_gate_bce = refined.new_tensor(0.0)
        loss_gate_sparsity = refined.new_tensor(0.0)
        gate_mean = 0.0
        gate_positive_target = 0.0

    total = (
        float(loss_cfg.get("lambda_cd", 1.0)) * loss_cd
        + lambda_occ * loss_occ
        + float(loss_cfg.get("lambda_delta", 0.5)) * loss_delta
        + float(loss_cfg.get("lambda_partial", 0.05)) * loss_partial
        + lambda_repulsion * loss_repulse
        + lambda_gate_bce * loss_gate_bce
        + lambda_gate_sparsity * loss_gate_sparsity
        + lambda_delta_l2 * loss_delta_l2
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "loss_cd": float(loss_cd.detach().cpu()),
        "loss_occ": float(loss_occ.detach().cpu()),
        "loss_delta": float(loss_delta.detach().cpu()),
        "loss_partial": float(loss_partial.detach().cpu()),
        "loss_repulsion": float(loss_repulse.detach().cpu()),
        "loss_gate_bce": float(loss_gate_bce.detach().cpu()),
        "loss_gate_sparsity": float(loss_gate_sparsity.detach().cpu()),
        "loss_delta_l2": float(loss_delta_l2.detach().cpu()),
        "gate_mean": gate_mean,
        "gate_positive_target": gate_positive_target,
    }
    return total, logs


def _normalized_hard_weights(coarse_cd: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0.0:
        return torch.ones_like(coarse_cd)
    span = coarse_cd.max() - coarse_cd.min()
    if float(span.detach().cpu()) > 1e-8:
        normalized = (coarse_cd - coarse_cd.min()) / span.clamp_min(1e-8)
    else:
        normalized = coarse_cd / coarse_cd.mean().clamp_min(1e-8)
    return 1.0 + float(alpha) * normalized.detach()


def compute_seed_losses(model, batch: Dict, cfg: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_cfg = cfg.get("loss", {})
    partial = batch["partial"]
    coarse = batch["coarse"]
    gt = batch["gt"]
    out = model(
        partial=partial,
        coarse=coarse,
        candidates=batch["candidates"],
        source_ids=batch.get("source_ids"),
    )
    refined = out["refined"]
    squared = bool(loss_cfg.get("squared_cd", False))
    refined_cd = chamfer_distance_per_sample(refined, gt, squared=squared)
    with torch.no_grad():
        coarse_eval = coarse
        if coarse_eval.shape[1] != refined.shape[1]:
            from .point_ops import farthest_point_sample

            coarse_eval = farthest_point_sample(coarse_eval, refined.shape[1])
        coarse_cd = chamfer_distance_per_sample(coarse_eval, gt, squared=squared)
        sample_weights = _normalized_hard_weights(coarse_cd, float(loss_cfg.get("hard_weight_alpha", 0.0)))
    loss_cd = (refined_cd * sample_weights).mean() / sample_weights.mean().clamp_min(1e-8)

    candidate_points = out["candidate_points"]
    confidence_threshold = float(loss_cfg.get("confidence_threshold", 0.03))
    with torch.no_grad():
        candidate_to_gt = pairwise_dist(candidate_points, gt).min(dim=2).values
        confidence_target = (candidate_to_gt < confidence_threshold).float()
    loss_candidate_conf = F.binary_cross_entropy_with_logits(out["point_confidence"], confidence_target)

    loss_partial = partial_preservation_loss(refined, partial)
    lambda_repulsion = float(loss_cfg.get("lambda_repulsion", 0.01))
    if lambda_repulsion != 0.0:
        loss_repulse = repulsion_loss(
            refined,
            k=int(loss_cfg.get("repulsion_k", 5)),
            radius=float(loss_cfg.get("repulsion_radius", 0.025)),
        )
    else:
        loss_repulse = refined.new_tensor(0.0)
    loss_delta_l2 = out["residual_delta"].square().mean()
    probs = torch.softmax(out["source_logits"], dim=-1).clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    loss_source_entropy = -entropy
    loss_fscore_proxy = torch.relu(confidence_threshold - pairwise_dist(gt, refined).min(dim=2).values).neg().mean()

    total = (
        float(loss_cfg.get("lambda_cd", 1.0)) * loss_cd
        + float(loss_cfg.get("lambda_candidate_conf", 0.05)) * loss_candidate_conf
        + float(loss_cfg.get("lambda_partial", 0.03)) * loss_partial
        + lambda_repulsion * loss_repulse
        + float(loss_cfg.get("lambda_delta_l2", 0.05)) * loss_delta_l2
        + float(loss_cfg.get("lambda_source_entropy", 0.001)) * loss_source_entropy
        + float(loss_cfg.get("lambda_fscore_proxy", 0.0)) * loss_fscore_proxy
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "loss_cd": float(loss_cd.detach().cpu()),
        "loss_occ": 0.0,
        "loss_delta": 0.0,
        "loss_partial": float(loss_partial.detach().cpu()),
        "loss_repulsion": float(loss_repulse.detach().cpu()),
        "loss_gate_bce": 0.0,
        "loss_gate_sparsity": 0.0,
        "loss_delta_l2": float(loss_delta_l2.detach().cpu()),
        "gate_mean": 0.0,
        "gate_positive_target": 0.0,
        "loss_candidate_conf": float(loss_candidate_conf.detach().cpu()),
        "loss_source_entropy": float(loss_source_entropy.detach().cpu()),
        "loss_fscore_proxy": float(loss_fscore_proxy.detach().cpu()),
        "hard_weight_mean": float(sample_weights.detach().mean().cpu()),
    }
    return total, logs


def _avg(values):
    return sum(values) / max(1, len(values))


@torch.no_grad()
def evaluate(model, loader: DataLoader, cfg: Dict, device: torch.device, max_batches: int | None = None) -> Dict[str, float]:
    model.eval()
    rows = []
    f_threshold = float(cfg.get("eval", {}).get("fscore_threshold", 0.03))
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = to_device(batch, device)
        if "candidates" in batch:
            out = model(batch["partial"], batch["coarse"], candidates=batch["candidates"], source_ids=batch.get("source_ids"))
        else:
            out = model(batch["partial"], batch["coarse"])
        refined = out["refined"]
        coarse = batch["coarse"]
        gt = batch["gt"]
        batch_n = int(gt.shape[0])
        coarse_components = chamfer_components(coarse, gt)
        refined_components = chamfer_components(refined, gt)
        coarse_cd = coarse_components["cd"]
        refined_cd = refined_components["cd"]
        coarse_f = fscore(coarse, gt, threshold=f_threshold)["fscore"]
        refined_f = fscore(refined, gt, threshold=f_threshold)["fscore"]
        rows.append({
            "coarse_cd_pred_to_gt": coarse_components["cd_pred_to_gt"],
            "coarse_cd_gt_to_pred": coarse_components["cd_gt_to_pred"],
            "coarse_cd": coarse_cd,
            "refined_cd_pred_to_gt": refined_components["cd_pred_to_gt"],
            "refined_cd_gt_to_pred": refined_components["cd_gt_to_pred"],
            "refined_cd": refined_cd,
            "coarse_fscore": coarse_f,
            "refined_fscore": refined_f,
            "n_samples": batch_n,
        })
    if not rows:
        return {
            "coarse_cd_pred_to_gt": math.nan,
            "coarse_cd_gt_to_pred": math.nan,
            "coarse_cd": math.nan,
            "refined_cd_pred_to_gt": math.nan,
            "refined_cd_gt_to_pred": math.nan,
            "refined_cd": math.nan,
            "coarse_fscore": math.nan,
            "refined_fscore": math.nan,
            "cd_improvement_pct": math.nan,
            "fscore_gain": math.nan,
            "n_samples": 0,
        }
    total_samples = sum(int(r["n_samples"]) for r in rows)
    metrics = {
        k: sum(float(r[k]) * int(r["n_samples"]) for r in rows) / max(1, total_samples)
        for k in rows[0].keys()
        if k != "n_samples"
    }
    metrics["cd_improvement_pct"] = 100.0 * (metrics["coarse_cd"] - metrics["refined_cd"]) / max(metrics["coarse_cd"], 1e-9)
    metrics["fscore_gain"] = metrics["refined_fscore"] - metrics["coarse_fscore"]
    metrics["n_samples"] = total_samples
    return metrics


def train(cfg: Dict, output_dir: str | Path) -> Dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg.get("train", {})
    num_threads = int(train_cfg.get("num_threads", 4))
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    set_seed(int(cfg.get("seed", 0)))
    device_str = cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        if not bool(train_cfg.get("allow_cpu", False)):
            raise RuntimeError("CUDA was requested for training, but torch.cuda.is_available() is False.")
        device_str = "cpu"
    device = torch.device(device_str)
    save_config(cfg, output_dir / "resolved_config.yaml")

    train_loader, val_loader = make_loaders(cfg)
    model = build_model_from_config(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(train_cfg.get("epochs", 50))))
    start_epoch = 0
    best_cd = float("inf")

    metrics_path = output_dir / "metrics.csv"
    metric_fields = [
        "epoch", "split", "loss", "loss_cd", "loss_occ", "loss_delta", "loss_partial", "loss_repulsion",
        "loss_gate_bce", "loss_gate_sparsity", "loss_delta_l2", "gate_mean", "gate_positive_target",
        "loss_candidate_conf", "loss_source_entropy", "loss_fscore_proxy", "hard_weight_mean",
        "coarse_cd_pred_to_gt", "coarse_cd_gt_to_pred", "coarse_cd",
        "refined_cd_pred_to_gt", "refined_cd_gt_to_pred", "refined_cd",
        "cd_improvement_pct", "coarse_fscore", "refined_fscore", "fscore_gain", "n_samples", "lr"
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()

    epochs = int(train_cfg.get("epochs", 50))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 20))
    val_every = int(train_cfg.get("val_every", 1))
    max_val_batches = train_cfg.get("max_val_batches", None)
    max_val_batches = None if max_val_batches in (None, "null") else int(max_val_batches)
    max_train_batches = train_cfg.get("max_train_batches", None)
    max_train_batches = None if max_train_batches in (None, "null") else int(max_train_batches)

    last_val = {}
    for epoch in range(start_epoch, epochs):
        model.train()
        agg = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for step, batch in enumerate(pbar):
            if max_train_batches is not None and step >= max_train_batches:
                break
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, logs = compute_losses(model, batch, cfg)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            agg.append(logs)
            if (step + 1) % log_every == 0 or step == 0:
                pbar.set_postfix({"loss": f"{logs['loss']:.4f}", "cd": f"{logs['loss_cd']:.4f}", "occ": f"{logs['loss_occ']:.4f}"})
        scheduler.step()
        train_logs = {k: _avg([r[k] for r in agg]) for k in agg[0].keys()}
        train_logs.update({"epoch": epoch + 1, "split": "train", "lr": scheduler.get_last_lr()[0]})

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields, extrasaction="ignore")
            writer.writerow(train_logs)

        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            last_val = evaluate(model, val_loader, cfg, device=device, max_batches=max_val_batches)
            val_row = {"epoch": epoch + 1, "split": "val", "lr": scheduler.get_last_lr()[0], **last_val}
            with metrics_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=metric_fields, extrasaction="ignore")
                writer.writerow(val_row)
            print(f"epoch {epoch+1}: val refined_cd={last_val['refined_cd']:.6f} coarse_cd={last_val['coarse_cd']:.6f} improvement={last_val['cd_improvement_pct']:.2f}%")
            if math.isfinite(last_val["refined_cd"]) and last_val["refined_cd"] < best_cd:
                best_cd = last_val["refined_cd"]
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch + 1, "metrics": last_val}, output_dir / "best_model.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch + 1, "metrics": last_val}, output_dir / "last_model.pt")

    summary = {"best_refined_cd": best_cd, "last_val": last_val, "output_dir": str(output_dir)}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def apply_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    cfg = dict(cfg)
    cfg.setdefault("train", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("model", {})
    cfg.setdefault("dataset", {})
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
        cfg["train"]["val_batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.no_local:
        cfg["model"]["use_local"] = False
    if args.lambda_occ is not None:
        cfg["loss"]["lambda_occ"] = args.lambda_occ
    if args.dataset_type is not None:
        cfg["dataset"]["type"] = args.dataset_type
    if args.data_root is not None:
        cfg["dataset"]["root"] = args.data_root
    if args.manifest is not None:
        cfg["dataset"]["manifest"] = args.manifest
        cfg["dataset"]["type"] = "manifest"
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PoinTr-IF implicit surface refiner")
    parser.add_argument("--config", default="configs/smoke_synthetic.yaml")
    parser.add_argument("--output-dir", default="outputs/train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-local", action="store_true", help="Ablation: global feature only, no local KNN features")
    parser.add_argument("--lambda-occ", type=float, default=None, help="Override occupancy loss weight")
    parser.add_argument("--dataset-type", default=None, choices=["synthetic", "npz", "h5", "manifest", "candidate_manifest"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    summary = train(cfg, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
