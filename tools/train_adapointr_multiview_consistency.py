#!/usr/bin/env python
"""Fine-tune AdaPoinTr with train-only multi-view consistency."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
POINTR_ROOT = ROOT / "external" / "PoinTr"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(POINTR_ROOT))

from pointr_if.io import load_point_cloud, resolve_project_path
from pointr_if.point_ops import chamfer_components, fscore, resample_points_np, set_seed


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:180]


def _normalize_like_gt(points: np.ndarray, gt: np.ndarray) -> np.ndarray:
    center = gt.mean(axis=0, keepdims=True)
    scale = np.sqrt(((gt - center) ** 2).sum(axis=1)).max().clip(1e-8)
    return ((points - center) / scale).astype(np.float32)


def _load_pointr_config(config: str | Path):
    from utils.config import cfg_from_yaml_file

    config_path = Path(config).resolve()
    cwd = Path.cwd()
    try:
        os.chdir(POINTR_ROOT)
        rel = config_path.relative_to(POINTR_ROOT) if config_path.is_relative_to(POINTR_ROOT) else config_path
        return cfg_from_yaml_file(str(rel))
    finally:
        os.chdir(cwd)


def load_adapointr(config: str | Path, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    from models import build_model_from_cfg

    cfg = _load_pointr_config(config)
    cwd = Path.cwd()
    try:
        os.chdir(POINTR_ROOT)
        model = build_model_from_cfg(cfg.model)
    finally:
        os.chdir(cwd)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("base_model") or ckpt.get("model")
    if state is None:
        raise RuntimeError(f"Checkpoint {checkpoint} does not contain base_model/model weights")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def _name_is_decoder_trainable(name: str) -> bool:
    prefixes = (
        "base_model.increase_dim",
        "base_model.coarse_pred",
        "base_model.mlp_query",
        "base_model.decoder",
        "base_model.query_ranking",
        "increase_dim",
        "reduce_map",
        "decode_head",
    )
    return name.startswith(prefixes)


def _encoder_block_prefixes(n_blocks: int) -> list[str]:
    return [f"base_model.encoder.blocks.blocks.{idx}." for idx in range(max(0, n_blocks))]


def set_trainable_modules(
    model: torch.nn.Module,
    *,
    unfreeze: str = "decoder_only",
    unfreeze_last_encoder_blocks: int = 0,
) -> dict[str, int]:
    """Apply MVC fine-tuning freeze policy and return parameter counts."""
    for param in model.parameters():
        param.requires_grad = False

    total_encoder_blocks = 0
    blocks = getattr(getattr(getattr(model, "base_model", None), "encoder", None), "blocks", None)
    if blocks is not None and hasattr(blocks, "blocks"):
        total_encoder_blocks = len(blocks.blocks)
    start_encoder = max(0, total_encoder_blocks - int(unfreeze_last_encoder_blocks))
    encoder_prefixes = _encoder_block_prefixes(total_encoder_blocks)[start_encoder:]

    counts = {"trainable": 0, "frozen": 0}
    for name, param in model.named_parameters():
        trainable = _name_is_decoder_trainable(name)
        if unfreeze == "decoder_plus_last_encoder" and any(name.startswith(prefix) for prefix in encoder_prefixes):
            trainable = True
        param.requires_grad = trainable
        counts["trainable" if trainable else "frozen"] += int(param.numel())
    return counts


def build_optimizer(model: torch.nn.Module, *, lr_decoder: float, lr_encoder: float, weight_decay: float) -> torch.optim.Optimizer:
    decoder_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("base_model.encoder"):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    groups = []
    if decoder_params:
        groups.append({"params": decoder_params, "lr": float(lr_decoder), "weight_decay": float(weight_decay)})
    if encoder_params and lr_encoder <= 0:
        raise RuntimeError("Encoder parameters were unfrozen, but lr_encoder is <= 0")
    if encoder_params and lr_encoder > 0:
        groups.append({"params": encoder_params, "lr": float(lr_encoder), "weight_decay": float(weight_decay)})
    if not groups:
        raise RuntimeError("No trainable parameters selected for MVC fine-tuning")
    return torch.optim.AdamW(groups)


class MultiViewGroupDataset(Dataset):
    def __init__(
        self,
        groups_path: str | Path,
        *,
        views_per_object: int,
        n_partial: int,
        n_gt: int,
        seed: int,
    ):
        self.groups = json.loads(resolve_project_path(groups_path).read_text(encoding="utf-8"))
        if not self.groups:
            raise ValueError(f"No groups found in {groups_path}")
        self.views_per_object = int(views_per_object)
        self.n_partial = int(n_partial)
        self.n_gt = int(n_gt)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        group = self.groups[idx]
        rng = np.random.default_rng(self.seed + idx)
        members = list(group["members"])
        replace = len(members) < self.views_per_object
        chosen_idx = rng.choice(len(members), size=self.views_per_object, replace=replace)
        partials = []
        sample_ids = []
        for member_index in chosen_idx:
            member = members[int(member_index)]
            partial = load_point_cloud(member["partial_path"])
            partial = resample_points_np(partial, self.n_partial, rng, mode="random")
            partials.append(partial)
            sample_ids.append(member["sample_id"])
        gt_full = load_point_cloud(group["gt_path"])
        gt_eval = resample_points_np(gt_full, self.n_gt, rng, mode="fps")
        return {
            "partials": torch.from_numpy(np.stack(partials, axis=0).astype(np.float32)),
            "gt_full": torch.from_numpy(gt_full.astype(np.float32)),
            "gt_eval": torch.from_numpy(gt_eval.astype(np.float32)),
            "group_id": group["group_id"],
            "category": group.get("category", ""),
            "sample_ids": sample_ids,
        }


class ManifestEvalDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        n_partial: int,
        n_gt: int,
        n_output: int,
        eval_seed: int,
        input_seed: int,
    ):
        from pointr_if.io import read_manifest

        self.rows = read_manifest(resolve_project_path(manifest))
        self.n_partial = int(n_partial)
        self.n_gt = int(n_gt)
        self.n_output = int(n_output)
        self.eval_seed = int(eval_seed)
        self.input_seed = int(input_seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        input_rng = np.random.default_rng(self.input_seed + idx)
        eval_rng = np.random.default_rng(self.eval_seed + idx)
        row = self.rows[idx]
        partial_raw = load_point_cloud(row["partial"])
        coarse_raw = load_point_cloud(row["coarse"])
        gt_raw = load_point_cloud(row["gt"])
        partial_input = resample_points_np(partial_raw, self.n_partial, input_rng, mode="random")
        partial_eval = resample_points_np(_normalize_like_gt(partial_raw, gt_raw), self.n_partial, eval_rng, mode="fps")
        coarse_eval = resample_points_np(_normalize_like_gt(coarse_raw, gt_raw), self.n_output, eval_rng, mode="fps")
        gt_eval = resample_points_np(_normalize_like_gt(gt_raw, gt_raw), self.n_gt, eval_rng, mode="fps")
        return {
            "partial_input": torch.from_numpy(partial_input.astype(np.float32)),
            "partial_eval": torch.from_numpy(partial_eval.astype(np.float32)),
            "coarse_eval": torch.from_numpy(coarse_eval.astype(np.float32)),
            "gt_eval": torch.from_numpy(gt_eval.astype(np.float32)),
            "gt_raw": torch.from_numpy(gt_raw.astype(np.float32)),
            "sample_id": row.get("sample_id") or row.get("id"),
            "category": row.get("category", ""),
            "split": row.get("split", ""),
        }


def _as_list(value: Any, n: int) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] * n


def _fps(points: torch.Tensor, n: int) -> torch.Tensor:
    from utils import misc

    return misc.fps(points, int(n))


def _sample_points(points: torch.Tensor, n: int, mode: str) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "fps":
        return _fps(points, n)
    if mode != "random":
        raise ValueError(f"Unsupported point sample mode: {mode}")
    n = int(n)
    total = int(points.shape[1])
    if total == n:
        return points
    if total < n:
        repeats = (n + total - 1) // total
        return points.repeat(1, repeats, 1)[:, :n, :]
    idx = torch.stack([torch.randperm(total, device=points.device)[:n] for _ in range(points.shape[0])], dim=0)
    return points.gather(1, idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))


def _metric_row(sample_id: str, category: str, split: str, method: str, pred: torch.Tensor, gt: torch.Tensor) -> dict[str, Any]:
    comps = chamfer_components(pred, gt)
    fs = fscore(pred, gt, threshold=0.02)
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


def _mean_rows(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(rows)
    keys = ["cd_pred_to_gt", "cd_gt_to_pred", "chamfer", "precision", "recall", "fscore"]
    if not rows:
        return {key: math.nan for key in keys} | {"n_samples": 0}
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys} | {"n_samples": len(rows)}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    *,
    manifest: str | Path,
    out_dir: str | Path | None,
    device: torch.device,
    n_partial: int,
    n_output: int,
    n_gt: int,
    eval_seed: int,
    input_seed: int = 570,
    batch_size: int = 1,
    max_samples: int | None = None,
    save_predictions: bool = False,
) -> dict[str, Any]:
    from pointr_if.io import save_point_cloud

    model.eval()
    dataset = ManifestEvalDataset(
        manifest,
        n_partial=n_partial,
        n_gt=n_gt,
        n_output=n_output,
        eval_seed=eval_seed,
        input_seed=input_seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows: list[dict[str, Any]] = []
    out_path = None if out_dir is None else Path(out_dir)
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
    seen = 0
    for batch in tqdm(loader, desc=f"eval {Path(manifest).stem}", leave=False):
        if max_samples is not None and seen >= max_samples:
            break
        partial = batch["partial_input"].to(device=device, dtype=torch.float32)
        ret = model(partial)
        refined_raw = ret[-1]
        refined_raw = _fps(refined_raw, n_output)
        sample_ids = _as_list(batch["sample_id"], refined_raw.shape[0])
        categories = _as_list(batch["category"], refined_raw.shape[0])
        splits = _as_list(batch["split"], refined_raw.shape[0])
        for i in range(refined_raw.shape[0]):
            if max_samples is not None and seen >= max_samples:
                break
            gt_raw_np = batch["gt_raw"][i].cpu().numpy()
            refined_eval_np = _normalize_like_gt(refined_raw[i].detach().cpu().numpy(), gt_raw_np)
            refined_eval = torch.from_numpy(refined_eval_np).to(device=device, dtype=torch.float32).unsqueeze(0)
            coarse_eval = batch["coarse_eval"][i : i + 1].to(device=device, dtype=torch.float32)
            partial_eval = batch["partial_eval"][i : i + 1].to(device=device, dtype=torch.float32)
            gt_eval = batch["gt_eval"][i : i + 1].to(device=device, dtype=torch.float32)
            rows.extend(
                [
                    _metric_row(sample_ids[i], categories[i], splits[i], "partial", partial_eval, gt_eval),
                    _metric_row(sample_ids[i], categories[i], splits[i], "coarse", coarse_eval, gt_eval),
                    _metric_row(sample_ids[i], categories[i], splits[i], "refined", refined_eval, gt_eval),
                ]
            )
            if save_predictions and out_path is not None:
                pred_dir = out_path / "predictions"
                save_point_cloud(pred_dir / f"{_safe_name(sample_ids[i])}_refined.npy", refined_eval_np)
            seen += 1

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
        by_category_method[(str(row["category"]), str(row["method"]))].append(row)
    methods = {method: _mean_rows(group) for method, group in sorted(by_method.items())}
    coarse_cd = methods.get("coarse", {}).get("chamfer", math.nan)
    improvements = {
        f"{method}_vs_coarse_cd_percent": 100.0 * (coarse_cd - metrics["chamfer"]) / max(coarse_cd, 1e-9)
        for method, metrics in methods.items()
        if method != "coarse"
    }
    summary = {"n_samples": int(methods.get("coarse", {}).get("n_samples", 0)), "methods": methods, "improvements": improvements}

    if out_path is not None:
        fields = ["sample_id", "category", "split", "method", "cd_pred_to_gt", "cd_gt_to_pred", "chamfer", "precision", "recall", "fscore"]
        _write_csv(out_path / "per_sample_metrics.csv", rows, fields)
        category_rows = [{"category": cat, "method": method, **_mean_rows(group)} for (cat, method), group in sorted(by_category_method.items())]
        _write_csv(out_path / "category_metrics.csv", category_rows, ["category", "method", "n_samples", "cd_pred_to_gt", "cd_gt_to_pred", "chamfer", "precision", "recall", "fscore"])
        (out_path / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        lines = ["| method | n | chamfer | fscore |", "|---|---:|---:|---:|"]
        for method, metrics in sorted(methods.items()):
            lines.append(f"| {method} | {int(metrics['n_samples'])} | {metrics['chamfer']:.6f} | {metrics['fscore']:.6f} |")
        lines.extend(["", "| comparison | CD improvement vs coarse |", "|---|---:|"])
        for key, value in sorted(improvements.items()):
            lines.append(f"| {key} | {value:.4f}% |")
        (out_path / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _load_chamfer_l1():
    from extensions.chamfer_dist import ChamferDistanceL1

    return ChamferDistanceL1()


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for MVC training but unavailable")
    if int(args.num_threads) > 0:
        torch.set_num_threads(int(args.num_threads))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    set_seed(int(args.seed))
    random.seed(int(args.seed))
    started_at = time.monotonic()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    (out_dir / "command_log.txt").write_text(" ".join(shlex.quote(part) for part in sys.argv) + "\n", encoding="utf-8")

    model = load_adapointr(args.config, args.checkpoint, device)
    trainable_counts = set_trainable_modules(
        model,
        unfreeze=args.unfreeze,
        unfreeze_last_encoder_blocks=int(args.unfreeze_last_encoder_blocks),
    )
    optimizer = build_optimizer(model, lr_decoder=args.lr_decoder, lr_encoder=args.lr_encoder, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    chamfer_l1 = _load_chamfer_l1()

    train_dataset = MultiViewGroupDataset(
        args.train_groups,
        views_per_object=args.views_per_object,
        n_partial=args.n_partial,
        n_gt=args.n_gt,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    metrics_path = out_dir / "metrics.csv"
    metric_fields = [
        "epoch",
        "split",
        "loss",
        "loss_rec",
        "loss_target",
        "loss_self",
        "loss_missing",
        "val_coarse_cd",
        "val_refined_cd",
        "val_cd_improvement_pct",
        "val_coarse_fscore",
        "val_refined_fscore",
        "lr_decoder",
        "lr_encoder",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=metric_fields).writeheader()

    best_val_cd = math.inf
    best_summary: dict[str, Any] | None = None
    worsen_streak = 0
    baseline_val_cd = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        accum = {"loss": 0.0, "loss_rec": 0.0, "loss_target": 0.0, "loss_self": 0.0, "loss_missing": 0.0}
        batches = 0
        optimizer_stepped = False
        time_budget_reached = False
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}")):
            if args.max_train_batches is not None and step >= int(args.max_train_batches):
                break
            partials = batch["partials"].to(device=device, dtype=torch.float32)
            gt_full = batch["gt_full"].to(device=device, dtype=torch.float32)
            gt_eval = batch["gt_eval"].to(device=device, dtype=torch.float32)
            b, k, n, _ = partials.shape
            flat_partial = partials.reshape(b * k, n, 3)
            flat_gt_full = gt_full.unsqueeze(1).expand(-1, k, -1, -1).reshape(b * k, gt_full.shape[1], 3)
            flat_gt_eval = gt_eval.unsqueeze(1).expand(-1, k, -1, -1).reshape(b * k, gt_eval.shape[1], 3)

            with torch.amp.autocast("cuda", enabled=bool(args.amp and device.type == "cuda")):
                ret = model(flat_partial)
                loss_denoised, loss_recon = model.get_loss(ret, flat_gt_full, epoch)
                loss_rec = loss_denoised + loss_recon
                pred_fine = ret[-1]
                pred_eval = _sample_points(pred_fine, int(args.n_output), args.target_sample_mode)
                loss_target = chamfer_l1(pred_eval, flat_gt_eval)
                pred_self = _sample_points(pred_fine, int(args.self_points), args.target_sample_mode).reshape(b, k, int(args.self_points), 3)
                if k > 1 and float(args.lambda_self) != 0.0:
                    pair_losses = []
                    for i in range(k):
                        for j in range(i + 1, k):
                            pair_losses.append(chamfer_l1(pred_self[:, i], pred_self[:, j].detach()))
                            pair_losses.append(chamfer_l1(pred_self[:, j], pred_self[:, i].detach()))
                    loss_self = torch.stack(pair_losses).mean()
                else:
                    loss_self = pred_fine.new_tensor(0.0)
                loss_missing = pred_fine.new_tensor(0.0)
                loss = (
                    loss_rec
                    + float(args.lambda_target) * loss_target
                    + float(args.lambda_self) * loss_self
                    + float(args.lambda_missing) * loss_missing
                )
                loss = loss / int(args.grad_accum_steps)
            scaler.scale(loss).backward()
            if (step + 1) % int(args.grad_accum_steps) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = optimizer_stepped or (not bool(args.amp and device.type == "cuda") or scaler.get_scale() >= scale_before_step)
                optimizer.zero_grad(set_to_none=True)
            accum["loss"] += float(loss.detach().cpu()) * int(args.grad_accum_steps)
            accum["loss_rec"] += float(loss_rec.detach().cpu())
            accum["loss_target"] += float(loss_target.detach().cpu())
            accum["loss_self"] += float(loss_self.detach().cpu())
            accum["loss_missing"] += float(loss_missing.detach().cpu())
            batches += 1
            if args.time_budget_hours is not None:
                elapsed_hours = (time.monotonic() - started_at) / 3600.0
                if elapsed_hours >= float(args.time_budget_hours):
                    time_budget_reached = True
                    break
        if batches and batches % int(args.grad_accum_steps) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = optimizer_stepped or (not bool(args.amp and device.type == "cuda") or scaler.get_scale() >= scale_before_step)
            optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and optimizer_stepped:
            scheduler.step()

        train_row = {
            "epoch": epoch,
            "split": "train",
            **{key: value / max(1, batches) for key, value in accum.items()},
            "val_coarse_cd": "",
            "val_refined_cd": "",
            "val_cd_improvement_pct": "",
            "val_coarse_fscore": "",
            "val_refined_fscore": "",
            "lr_decoder": optimizer.param_groups[0]["lr"],
            "lr_encoder": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
        }

        val_summary = evaluate_model(
            model,
            manifest=args.val_manifest,
            out_dir=out_dir / f"val_epoch_{epoch:03d}" if args.save_val_outputs else None,
            device=device,
            n_partial=args.n_partial,
            n_output=args.n_output,
            n_gt=args.n_gt,
            eval_seed=args.eval_seed,
            input_seed=args.input_seed,
            batch_size=1,
            max_samples=args.max_val_samples,
        )
        val_coarse = val_summary["methods"]["coarse"]
        val_refined = val_summary["methods"]["refined"]
        if baseline_val_cd is None:
            baseline_val_cd = float(val_coarse["chamfer"])
        val_gain = 100.0 * (float(val_coarse["chamfer"]) - float(val_refined["chamfer"])) / max(float(val_coarse["chamfer"]), 1e-9)
        val_row = {
            "epoch": epoch,
            "split": "val",
            "loss": "",
            "loss_rec": "",
            "loss_target": "",
            "loss_self": "",
            "loss_missing": "",
            "val_coarse_cd": float(val_coarse["chamfer"]),
            "val_refined_cd": float(val_refined["chamfer"]),
            "val_cd_improvement_pct": val_gain,
            "val_coarse_fscore": float(val_coarse["fscore"]),
            "val_refined_fscore": float(val_refined["fscore"]),
            "lr_decoder": optimizer.param_groups[0]["lr"],
            "lr_encoder": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writerow(train_row)
            writer.writerow(val_row)

        ckpt_payload = {
            "base_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": val_summary,
            "best_metrics": best_summary or {},
            "args": vars(args),
            "trainable_counts": trainable_counts,
        }
        torch.save(ckpt_payload, out_dir / "ckpt-last.pth")
        if float(val_refined["chamfer"]) < best_val_cd:
            best_val_cd = float(val_refined["chamfer"])
            best_summary = val_summary
            ckpt_payload["best_metrics"] = best_summary
            torch.save(ckpt_payload, out_dir / "ckpt-best.pth")
            worsen_streak = 0
        elif float(val_refined["chamfer"]) > best_val_cd * 1.03:
            worsen_streak += 1
        else:
            worsen_streak = 0
        print(
            f"epoch {epoch}: val refined_cd={float(val_refined['chamfer']):.6f} "
            f"coarse_cd={float(val_coarse['chamfer']):.6f} gain={val_gain:.4f}%"
        )
        if worsen_streak >= 3:
            print("Early stop: validation CD worsened by >3% for 3 consecutive epochs.")
            break
        if args.time_budget_hours is not None:
            elapsed_hours = (time.monotonic() - started_at) / 3600.0
            if time_budget_reached or elapsed_hours >= float(args.time_budget_hours):
                print(f"Time budget reached after {elapsed_hours:.2f} hours.")
                break

    summary = {
        "best_val_cd": best_val_cd,
        "baseline_val_cd": baseline_val_cd,
        "best_val_improvement_percent": None
        if baseline_val_cd is None or not math.isfinite(best_val_cd)
        else 100.0 * (baseline_val_cd - best_val_cd) / max(baseline_val_cd, 1e-9),
        "best_checkpoint": str(out_dir / "ckpt-best.pth"),
        "metrics_csv": str(metrics_path),
        "trainable_counts": trainable_counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-groups", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--views-per-object", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-decoder", type=float, default=3e-5)
    parser.add_argument("--lr-encoder", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--unfreeze", choices=["decoder_only", "decoder_plus_last_encoder"], default="decoder_only")
    parser.add_argument("--unfreeze-last-encoder-blocks", type=int, default=0)
    parser.add_argument("--lambda-target", type=float, default=1.0)
    parser.add_argument("--lambda-self", type=float, default=0.1)
    parser.add_argument("--lambda-missing", type=float, default=0.0)
    parser.add_argument("--n-partial", type=int, default=2048)
    parser.add_argument("--n-output", type=int, default=4096)
    parser.add_argument("--n-gt", type=int, default=4096)
    parser.add_argument("--self-points", type=int, default=2048)
    parser.add_argument("--eval-seed", type=int, default=200570)
    parser.add_argument("--input-seed", type=int, default=570)
    parser.add_argument("--seed", type=int, default=570)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--target-sample-mode", choices=["fps", "random"], default="fps")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--save-val-outputs", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
