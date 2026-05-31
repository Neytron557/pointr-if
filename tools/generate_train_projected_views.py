#!/usr/bin/env python
"""Generate deterministic train-only projected partial views from GT point clouds."""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pointr_if.io import load_point_cloud, save_point_cloud
from pointr_if.point_ops import resample_points_np


VIEW_DIRECTIONS = np.asarray(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 0.5],
        [-1, 1, 0.5],
        [1, -1, 0.5],
        [1, 1, -0.5],
        [0.5, 1, 1],
        [1, 0.5, 1],
    ],
    dtype=np.float32,
)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)[:180]


def _basis_from_direction(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(direction, dtype=np.float32)
    w = w / max(float(np.linalg.norm(w)), 1e-8)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(up, w))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(up, w)
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    v = np.cross(w, u)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    return u, v, w


def project_visible_points(
    points: np.ndarray,
    *,
    view_direction: np.ndarray,
    image_size: int = 64,
    n_points: int = 2048,
) -> np.ndarray:
    """Project points to an orthographic depth map and keep nearest visible cells."""
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    u, v, w = _basis_from_direction(view_direction)
    x = pts @ u
    y = pts @ v
    depth = pts @ w
    x_norm = (x - x.min()) / max(float(x.max() - x.min()), 1e-8)
    y_norm = (y - y.min()) / max(float(y.max() - y.min()), 1e-8)
    xi = np.clip(np.floor(x_norm * (image_size - 1)).astype(np.int64), 0, image_size - 1)
    yi = np.clip(np.floor(y_norm * (image_size - 1)).astype(np.int64), 0, image_size - 1)
    cell = yi * image_size + xi

    order = np.lexsort((-depth, cell))
    sorted_cell = cell[order]
    keep_mask = np.ones(order.shape[0], dtype=bool)
    keep_mask[1:] = sorted_cell[1:] != sorted_cell[:-1]
    visible = pts[order[keep_mask]]
    if visible.shape[0] == 0:
        visible = pts
    return resample_points_np(visible, int(n_points), np.random.default_rng(0), mode="fps")


def _model_id_from_group(group: dict[str, Any]) -> str:
    group_id = str(group["group_id"])
    category = str(group.get("category", ""))
    prefix = f"{category}-"
    return group_id[len(prefix) :] if category and group_id.startswith(prefix) else _safe_name(group_id)


def generate_views(args: argparse.Namespace) -> dict[str, Any]:
    groups = json.loads(Path(args.groups).read_text(encoding="utf-8"))
    out_root = Path(args.out_root)
    partial_root = out_root / "projected_partial"
    augmented = []
    generated = 0
    for group in groups:
        category = str(group.get("category", "unknown"))
        model_id = _model_id_from_group(group)
        gt = load_point_cloud(group["gt_path"])
        new_group = {**group, "members": list(group["members"])}
        for view_index in range(int(args.views_per_object)):
            direction = VIEW_DIRECTIONS[view_index % len(VIEW_DIRECTIONS)]
            if view_index >= len(VIEW_DIRECTIONS):
                angle = 2.0 * math.pi * (view_index / max(1, int(args.views_per_object)))
                direction = np.asarray([math.cos(angle), math.sin(angle), 0.6], dtype=np.float32)
            partial = project_visible_points(
                gt,
                view_direction=direction,
                image_size=int(args.image_size),
                n_points=int(args.n_points),
            )
            path = partial_root / category / model_id / "models" / f"{view_index + 1}.pcd"
            save_point_cloud(path, partial)
            generated += 1
            new_group["members"].append(
                {
                    "partial_path": str(path),
                    "coarse_path": "",
                    "sample_id": f"{group['group_id']}_generated_view{view_index + 1}",
                    "generated": True,
                }
            )
        augmented.append(new_group)

    out_groups = Path(args.out_groups)
    out_groups.parent.mkdir(parents=True, exist_ok=True)
    out_groups.write_text(json.dumps(augmented, indent=2) + "\n", encoding="utf-8")
    summary = {
        "input_groups": str(args.groups),
        "out_groups": str(out_groups),
        "out_root": str(out_root),
        "n_groups": len(groups),
        "generated_views_per_object": int(args.views_per_object),
        "n_generated_views": generated,
        "image_size": int(args.image_size),
        "n_points": int(args.n_points),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    summary_path = out_groups.parent / "generated_train_views_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="data/real_projected_shapenet55_groups/train_groups.json")
    parser.add_argument("--out-root", default="data/generated_train_views")
    parser.add_argument("--out-groups", default="data/real_projected_shapenet55_groups/train_groups_augmented.json")
    parser.add_argument("--views-per-object", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--n-points", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    generate_views(parse_args())


if __name__ == "__main__":
    main()
