from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_points_np(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Center and scale a point cloud to a unit sphere."""
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    scale = np.sqrt((points**2).sum(axis=1)).max()
    return points / max(float(scale), eps)


def normalize_points_torch(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Center and scale a batched point cloud to a unit sphere."""
    center = points.mean(dim=-2, keepdim=True)
    centered = points - center
    scale = torch.linalg.norm(centered, dim=-1).amax(dim=-1, keepdim=True).clamp_min(eps)
    return centered / scale.unsqueeze(-1)


def resample_points_np(points: np.ndarray, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Randomly resample a point cloud to exactly n points."""
    if rng is None:
        rng = np.random.default_rng()
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected point cloud of shape [N, >=3], got {points.shape}")
    points = points[:, :3]
    if points.shape[0] == n:
        return points.astype(np.float32)
    replace = points.shape[0] < n
    idx = rng.choice(points.shape[0], size=n, replace=replace)
    return points[idx].astype(np.float32)


def farthest_point_sample(points: torch.Tensor, n: int) -> torch.Tensor:
    """Deterministically sample ``n`` points from each batched cloud with FPS."""
    if points.ndim != 3 or points.shape[-1] < 3:
        raise ValueError("points must be a batched point cloud [B,N,>=3].")
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive.")
    points = points[..., :3]
    b, total, c = points.shape
    if total == n:
        return points
    if total < n:
        repeats = math.ceil(n / total)
        return points.repeat(1, repeats, 1)[:, :n, :]

    centroids = torch.zeros(b, n, dtype=torch.long, device=points.device)
    distance = torch.full((b, total), 1e10, dtype=points.dtype, device=points.device)
    farthest = torch.zeros(b, dtype=torch.long, device=points.device)
    batch_indices = torch.arange(b, dtype=torch.long, device=points.device)
    for i in range(n):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest].view(b, 1, c)
        dist = ((points - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=-1).indices
    return points[batch_indices[:, None], centroids]


def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance between BxNx3 and BxMx3 tensors."""
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("x and y must be batched point clouds [B,N,3] and [B,M,3].")
    return torch.cdist(x, y, p=2)


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """Symmetric Chamfer distance.

    Args:
        pred: [B, N, 3]
        target: [B, M, 3]
        squared: if True, use squared Euclidean distances.
    """
    d = pairwise_dist(pred, target)
    mins_pred = d.min(dim=2).values
    mins_target = d.min(dim=1).values
    if squared:
        mins_pred = mins_pred.square()
        mins_target = mins_target.square()
    return mins_pred.mean(dim=1).mean() + mins_target.mean(dim=1).mean()


@torch.no_grad()
def chamfer_components(pred: torch.Tensor, target: torch.Tensor, squared: bool = False) -> Dict[str, float]:
    d = pairwise_dist(pred, target)
    mins_pred = d.min(dim=2).values
    mins_target = d.min(dim=1).values
    if squared:
        mins_pred = mins_pred.square()
        mins_target = mins_target.square()
    return {
        "cd_pred_to_gt": float(mins_pred.mean().detach().cpu()),
        "cd_gt_to_pred": float(mins_target.mean().detach().cpu()),
        "cd": float((mins_pred.mean() + mins_target.mean()).detach().cpu()),
    }


@torch.no_grad()
def fscore(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.02) -> Dict[str, float]:
    """F-score at a distance threshold."""
    d = pairwise_dist(pred, target)
    pred_to_gt = d.min(dim=2).values
    gt_to_pred = d.min(dim=1).values
    precision = (pred_to_gt < threshold).float().mean()
    recall = (gt_to_pred < threshold).float().mean()
    denom = (precision + recall).clamp_min(1e-8)
    f = 2 * precision * recall / denom
    return {
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "fscore": float(f.detach().cpu()),
    }


def nearest_delta(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return nearest-neighbor residual from source to target: nearest(target)-source."""
    d = pairwise_dist(source, target)
    idx = d.argmin(dim=2)  # [B, N]
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, target.shape[-1])
    nearest = target.gather(dim=1, index=idx_expanded)
    return nearest - source


def sample_occupancy_queries(
    surface: torch.Tensor,
    n_query: int,
    threshold: float = 0.03,
    near_ratio: float = 0.65,
    near_std: float = 0.035,
    box_scale: float = 1.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create pseudo occupancy labels from complete point clouds.

    This avoids needing watertight meshes in a 2-day project. A query is treated as
    occupied / near-surface when its distance to the complete point cloud is below
    `threshold`. This is not a physically exact inside/outside label, but it gives
    the implicit head a continuous surface-aware supervision signal from PCN-like
    datasets that only provide complete point clouds.
    """
    if surface.ndim != 3 or surface.shape[-1] != 3:
        raise ValueError("surface must be [B,N,3]")
    b, n, _ = surface.shape
    device = surface.device
    n_near = int(n_query * near_ratio)
    n_uniform = n_query - n_near

    idx = torch.randint(0, n, (b, n_near), device=device)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, 3)
    near = surface.gather(dim=1, index=idx_exp) + near_std * torch.randn(b, n_near, 3, device=device)
    uniform = (torch.rand(b, n_uniform, 3, device=device) * 2 - 1) * box_scale
    query = torch.cat([near, uniform], dim=1)

    with torch.no_grad():
        dist = pairwise_dist(query, surface).min(dim=2).values
        labels = (dist < threshold).float()
    return query, labels


def partial_preservation_loss(refined: torch.Tensor, partial: torch.Tensor) -> torch.Tensor:
    """Encourage refined completion to retain observed partial geometry."""
    d = pairwise_dist(partial, refined).min(dim=2).values
    return d.mean()


def repulsion_loss(points: torch.Tensor, k: int = 5, radius: float = 0.05) -> torch.Tensor:
    """Simple uniformity regularizer that discourages point collapse."""
    d = pairwise_dist(points, points)
    # ignore self-distance
    eye = torch.eye(points.shape[1], device=points.device).unsqueeze(0) * 1e6
    d = d + eye
    knn = d.topk(k=min(k, points.shape[1] - 1), dim=2, largest=False).values
    return F.relu(radius - knn).mean()


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
