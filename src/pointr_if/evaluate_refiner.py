from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from .datasets import make_dataset
from .models import build_model_from_config
from .point_ops import chamfer_components, fscore, to_device


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str | Path, output_dir: str | Path, split: str = "val", device: str | None = None) -> Dict[str, float]:
    ckpt_path = Path(ckpt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    num_threads = int(cfg.get("train", {}).get("num_threads", 4))
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    dev = torch.device(device or cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    model = build_model_from_config(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(dev).eval()

    dset = make_dataset(cfg, split=split)
    loader = DataLoader(
        dset,
        batch_size=int(cfg.get("train", {}).get("val_batch_size", cfg.get("train", {}).get("batch_size", 4))),
        shuffle=False,
        num_workers=int(cfg.get("train", {}).get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    f_threshold = float(cfg.get("eval", {}).get("fscore_threshold", 0.03))
    rows = []
    for batch in tqdm(loader, desc=f"evaluating {split}"):
        ids = batch.get("id", [""] * batch["gt"].shape[0])
        batch = to_device(batch, dev)
        out = model(batch["partial"], batch["coarse"])
        refined = out["refined"]
        for b in range(refined.shape[0]):
            coarse_b = batch["coarse"][b:b+1]
            refined_b = refined[b:b+1]
            gt_b = batch["gt"][b:b+1]
            coarse_components = chamfer_components(coarse_b, gt_b)
            refined_components = chamfer_components(refined_b, gt_b)
            ccd = coarse_components["cd"]
            rcd = refined_components["cd"]
            cf = fscore(coarse_b, gt_b, threshold=f_threshold)["fscore"]
            rf = fscore(refined_b, gt_b, threshold=f_threshold)["fscore"]
            rows.append({
                "id": ids[b] if isinstance(ids, list) else str(ids[b]),
                "coarse_cd_pred_to_gt": coarse_components["cd_pred_to_gt"],
                "coarse_cd_gt_to_pred": coarse_components["cd_gt_to_pred"],
                "coarse_cd": ccd,
                "refined_cd_pred_to_gt": refined_components["cd_pred_to_gt"],
                "refined_cd_gt_to_pred": refined_components["cd_gt_to_pred"],
                "refined_cd": rcd,
                "cd_improvement_pct": 100.0 * (ccd - rcd) / max(ccd, 1e-9),
                "coarse_fscore": cf,
                "refined_fscore": rf,
                "fscore_gain": rf - cf,
            })

    csv_path = output_dir / f"{split}_per_sample_metrics.csv"
    fieldnames = [
        "id",
        "coarse_cd_pred_to_gt",
        "coarse_cd_gt_to_pred",
        "coarse_cd",
        "refined_cd_pred_to_gt",
        "refined_cd_gt_to_pred",
        "refined_cd",
        "cd_improvement_pct",
        "coarse_fscore",
        "refined_fscore",
        "fscore_gain",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        avg = {k: sum(float(r[k]) for r in rows) / len(rows) for k in fieldnames if k != "id"}
    else:
        avg = {k: None for k in fieldnames if k != "id"}
    avg["n_samples"] = len(rows)
    avg["checkpoint"] = str(ckpt_path)
    with (output_dir / f"{split}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(avg, f, indent=2)
    print(json.dumps(avg, indent=2))
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PoinTr-IF refiner checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate_checkpoint(args.ckpt, args.output_dir, split=args.split, device=args.device)


if __name__ == "__main__":
    main()
