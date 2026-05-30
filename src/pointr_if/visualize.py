from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import make_dataset
from .io import save_point_cloud
from .models import build_model_from_config
from .point_ops import to_device


def _scatter(ax, points, title):
    pts = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2)
    ax.set_title(title)
    ax.set_axis_off()
    lim = max(1.0, float(np.abs(pts).max()))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def visualize_checkpoint(ckpt_path: str | Path, output_dir: str | Path, n: int = 8, device: str | None = None) -> None:
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
    dset = make_dataset(cfg, split="val")
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n:
                break
            sample_id = batch.get("id", [f"sample_{i:04d}"])[0]
            batch = to_device(batch, dev)
            out = model(batch["partial"], batch["coarse"])
            fig = plt.figure(figsize=(16, 4))
            panels = [
                (batch["partial"][0], "partial"),
                (batch["coarse"][0], "coarse"),
                (out["refined"][0], "PoinTr-IF refined"),
                (batch["gt"][0], "ground truth"),
            ]
            for j, (pts, title) in enumerate(panels, start=1):
                ax = fig.add_subplot(1, 4, j, projection="3d")
                _scatter(ax, pts, title)
            fig.suptitle(sample_id)
            fig.tight_layout()
            fig.savefig(output_dir / f"{i:03d}_{sample_id}.png", dpi=180)
            plt.close(fig)
            save_point_cloud(output_dir / f"{i:03d}_{sample_id}_refined.pcd", out["refined"][0].detach().cpu().numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PoinTr-IF predictions")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-dir", default="outputs/visuals")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    visualize_checkpoint(args.ckpt, args.output_dir, n=args.n, device=args.device)


if __name__ == "__main__":
    main()
