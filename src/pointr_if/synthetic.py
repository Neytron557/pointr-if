from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np

from .point_ops import normalize_points_np, resample_points_np


def _sample_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(1e-8)
    scales = rng.uniform(0.75, 1.15, size=(1, 3)).astype(np.float32)
    return v * scales


def _sample_cube(n: int, rng: np.random.Generator, scale=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    pts = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)
    faces = rng.integers(0, 6, size=n)
    axis = faces // 2
    sign = (faces % 2) * 2 - 1
    pts[np.arange(n), axis] = sign * 0.5
    return pts * scale + center


def _sample_cylinder(n: int, rng: np.random.Generator) -> np.ndarray:
    n_side = int(n * 0.7)
    n_cap = n - n_side
    theta = rng.uniform(0, 2 * math.pi, size=n_side)
    z = rng.uniform(-0.5, 0.5, size=n_side)
    r = rng.uniform(0.35, 0.55)
    side = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1).astype(np.float32)
    cap_theta = rng.uniform(0, 2 * math.pi, size=n_cap)
    cap_r = np.sqrt(rng.uniform(0, r**2, size=n_cap))
    cap_z = rng.choice([-0.5, 0.5], size=n_cap)
    cap = np.stack([cap_r * np.cos(cap_theta), cap_r * np.sin(cap_theta), cap_z], axis=1).astype(np.float32)
    return np.concatenate([side, cap], axis=0)


def _sample_chair(n: int, rng: np.random.Generator) -> np.ndarray:
    # Five cuboids: seat, back, four legs. This creates a simple hard-case shape
    # with thin structures and missing regions similar to chair completion examples.
    counts = np.array([0.35, 0.25, 0.10, 0.10, 0.10, 0.10])
    counts = np.maximum(8, (counts * n).astype(int))
    counts[-1] += n - counts.sum()
    parts = [
        _sample_cube(counts[0], rng, scale=(1.0, 0.9, 0.12), center=(0.0, 0.0, 0.05)),
        _sample_cube(counts[1], rng, scale=(1.0, 0.12, 0.9), center=(0.0, 0.42, 0.50)),
        _sample_cube(counts[2], rng, scale=(0.10, 0.10, 0.65), center=(-0.42, -0.32, -0.30)),
        _sample_cube(counts[3], rng, scale=(0.10, 0.10, 0.65), center=(0.42, -0.32, -0.30)),
        _sample_cube(counts[4], rng, scale=(0.10, 0.10, 0.65), center=(-0.42, 0.32, -0.30)),
        _sample_cube(counts[5], rng, scale=(0.10, 0.10, 0.65), center=(0.42, 0.32, -0.30)),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


def _sample_table(n: int, rng: np.random.Generator) -> np.ndarray:
    counts = np.array([0.50, 0.125, 0.125, 0.125, 0.125])
    counts = np.maximum(8, (counts * n).astype(int))
    counts[-1] += n - counts.sum()
    parts = [
        _sample_cube(counts[0], rng, scale=(1.15, 0.8, 0.12), center=(0.0, 0.0, 0.35)),
        _sample_cube(counts[1], rng, scale=(0.10, 0.10, 0.7), center=(-0.48, -0.32, -0.05)),
        _sample_cube(counts[2], rng, scale=(0.10, 0.10, 0.7), center=(0.48, -0.32, -0.05)),
        _sample_cube(counts[3], rng, scale=(0.10, 0.10, 0.7), center=(-0.48, 0.32, -0.05)),
        _sample_cube(counts[4], rng, scale=(0.10, 0.10, 0.7), center=(0.48, 0.32, -0.05)),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


SHAPE_SAMPLERS = {
    "sphere": _sample_sphere,
    "cube": _sample_cube,
    "cylinder": _sample_cylinder,
    "chair": _sample_chair,
    "table": _sample_table,
}


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(theta), math.sin(theta)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    theta2 = rng.uniform(-0.35, 0.35)
    c2, s2 = math.cos(theta2), math.sin(theta2)
    rx = np.array([[1, 0, 0], [0, c2, -s2], [0, s2, c2]], dtype=np.float32)
    return rz @ rx


def make_partial(gt: np.ndarray, n_partial: int, keep_ratio: float, rng: np.random.Generator) -> np.ndarray:
    view = rng.normal(size=(3,)).astype(np.float32)
    view /= np.linalg.norm(view).clip(1e-8)
    score = gt @ view
    q = np.quantile(score, 1.0 - keep_ratio)
    visible = gt[score >= q]
    visible = visible + rng.normal(scale=0.003, size=visible.shape).astype(np.float32)
    return resample_points_np(visible, n_partial, rng)


def make_coarse(gt: np.ndarray, n_coarse: int, noise_std: float, missing_bias: float, rng: np.random.Generator) -> np.ndarray:
    coarse = resample_points_np(gt, n_coarse, rng).copy()
    # Simulate a completion network that is mostly correct but smooth/noisy in hard regions.
    coarse += rng.normal(scale=noise_std, size=coarse.shape).astype(np.float32)
    # Global mild shrinkage causes thin structures to move toward the center.
    coarse *= rng.uniform(1.0 - missing_bias, 1.0 + missing_bias * 0.25)
    # Add a low-frequency bend to mimic structured completion artifacts.
    coarse[:, 2] += (0.03 * missing_bias) * np.sin(3.0 * coarse[:, 0] + rng.uniform(-math.pi, math.pi))
    return coarse.astype(np.float32)


def generate_completion_sample(
    idx: int,
    n_gt: int = 1024,
    n_partial: int = 512,
    n_coarse: int = 1024,
    shapes: Iterable[str] = ("sphere", "cube", "cylinder", "chair", "table"),
    severity: str = "mixed",
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed + idx * 9973)
    shape_names = list(shapes)
    name = shape_names[idx % len(shape_names)]
    sampler = SHAPE_SAMPLERS[name]
    gt = sampler(max(n_gt * 2, n_gt + 64), rng)
    rot = random_rotation(rng)
    gt = gt @ rot.T
    gt = normalize_points_np(gt)
    gt = resample_points_np(gt, n_gt, rng)

    if severity == "easy":
        keep_ratio, noise, bias = 0.75, 0.025, 0.04
    elif severity == "medium":
        keep_ratio, noise, bias = 0.50, 0.045, 0.08
    elif severity == "hard":
        keep_ratio, noise, bias = 0.25, 0.065, 0.12
    else:
        sev = rng.choice(["easy", "medium", "hard"], p=[0.25, 0.45, 0.30])
        return generate_completion_sample(idx, n_gt, n_partial, n_coarse, shapes, sev, seed)

    partial = make_partial(gt, n_partial, keep_ratio, rng)
    coarse = make_coarse(gt, n_coarse, noise, bias, rng)
    return {
        "partial": partial.astype(np.float32),
        "coarse": coarse.astype(np.float32),
        "gt": gt.astype(np.float32),
        "shape": name,
        "severity": severity,
        "id": f"synthetic_{idx:06d}_{name}_{severity}",
    }
