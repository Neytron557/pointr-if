from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_point_cloud, load_triplet_npz, read_manifest
from .point_ops import resample_points_np
from .synthetic import generate_completion_sample


class SyntheticCompletionDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 512,
        n_partial: int = 512,
        n_coarse: int = 1024,
        n_gt: int = 1024,
        shapes: str = "sphere,cube,cylinder,chair,table",
        severity: str = "mixed",
        seed: int = 0,
    ):
        self.num_samples = int(num_samples)
        self.n_partial = int(n_partial)
        self.n_coarse = int(n_coarse)
        self.n_gt = int(n_gt)
        self.shapes = [s.strip() for s in shapes.split(",") if s.strip()]
        self.severity = severity
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = generate_completion_sample(
            idx=idx,
            n_gt=self.n_gt,
            n_partial=self.n_partial,
            n_coarse=self.n_coarse,
            shapes=self.shapes,
            severity=self.severity,
            seed=self.seed,
        )
        return {
            "partial": torch.from_numpy(sample["partial"]),
            "coarse": torch.from_numpy(sample["coarse"]),
            "gt": torch.from_numpy(sample["gt"]),
            "id": sample["id"],
            "shape": sample["shape"],
            "severity": sample["severity"],
        }


class TripletNPZDataset(Dataset):
    """Dataset of .npz files with partial/coarse/gt arrays."""

    def __init__(
        self,
        root: str | Path,
        n_partial: int = 2048,
        n_coarse: int = 2048,
        n_gt: int = 2048,
        normalize: bool = True,
        seed: int = 0,
        resample_mode: str = "random",
    ):
        self.root = Path(root)
        self.files = sorted(self.root.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under {root}")
        self.n_partial = int(n_partial)
        self.n_coarse = int(n_coarse)
        self.n_gt = int(n_gt)
        self.normalize = normalize
        self.seed = int(seed)
        self.resample_mode = str(resample_mode)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(self.seed + idx)
        item = load_triplet_npz(self.files[idx])
        partial = item["partial"]
        coarse = item["coarse"]
        gt = item["gt"]
        if self.normalize:
            # Normalize all clouds with the GT transform to avoid mismatched scaling.
            center = gt.mean(axis=0, keepdims=True)
            scale = np.sqrt(((gt - center) ** 2).sum(axis=1)).max().clip(1e-8)
            partial, coarse, gt = (partial - center) / scale, (coarse - center) / scale, (gt - center) / scale
        return {
            "partial": torch.from_numpy(resample_points_np(partial, self.n_partial, rng, mode=self.resample_mode)),
            "coarse": torch.from_numpy(resample_points_np(coarse, self.n_coarse, rng, mode=self.resample_mode)),
            "gt": torch.from_numpy(resample_points_np(gt, self.n_gt, rng, mode=self.resample_mode)),
            "id": self.files[idx].stem,
        }


class ManifestTripletDataset(Dataset):
    """Dataset described by a CSV with columns: id,partial,coarse,gt."""

    def __init__(
        self,
        manifest: str | Path,
        n_partial: Optional[int] = 2048,
        n_coarse: Optional[int] = 2048,
        n_gt: Optional[int] = 2048,
        normalize: bool = True,
        seed: int = 0,
        resample_mode: str = "random",
    ):
        self.rows = read_manifest(manifest)
        if not self.rows:
            raise ValueError(f"Manifest has no rows: {manifest}")
        self.n_partial = None if n_partial is None else int(n_partial)
        self.n_coarse = None if n_coarse is None else int(n_coarse)
        self.n_gt = None if n_gt is None else int(n_gt)
        self.normalize = normalize
        self.seed = int(seed)
        self.resample_mode = str(resample_mode)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(self.seed + idx)
        row = self.rows[idx]
        partial = load_point_cloud(row["partial"])
        coarse = load_point_cloud(row["coarse"])
        gt = load_point_cloud(row["gt"])
        if self.normalize:
            center = gt.mean(axis=0, keepdims=True)
            scale = np.sqrt(((gt - center) ** 2).sum(axis=1)).max().clip(1e-8)
            partial, coarse, gt = (partial - center) / scale, (coarse - center) / scale, (gt - center) / scale
        return {
            "partial": torch.from_numpy(resample_points_np(partial, self.n_partial, rng, mode=self.resample_mode)),
            "coarse": torch.from_numpy(resample_points_np(coarse, self.n_coarse, rng, mode=self.resample_mode)),
            "gt": torch.from_numpy(resample_points_np(gt, self.n_gt, rng, mode=self.resample_mode)),
            "id": row.get("id") or Path(row["gt"]).stem,
            "sample_id": row.get("sample_id") or row.get("id") or Path(row["gt"]).stem,
            "category": row.get("category", ""),
            "split": row.get("split", ""),
        }


class H5TripletDataset(Dataset):
    """HDF5 dataset with arrays for partial/input, coarse/pred and gt/complete.

    Supported keys are flexible. Examples:
      partial: partial, input, inputs
      coarse: coarse, pred, prediction
      gt: gt, complete, target
    """

    def __init__(
        self,
        path: str | Path,
        n_partial: int = 2048,
        n_coarse: int = 2048,
        n_gt: int = 2048,
        normalize: bool = True,
        seed: int = 0,
        resample_mode: str = "random",
    ):
        self.path = Path(path)
        self.n_partial = int(n_partial)
        self.n_coarse = int(n_coarse)
        self.n_gt = int(n_gt)
        self.normalize = normalize
        self.seed = int(seed)
        self.resample_mode = str(resample_mode)
        with h5py.File(self.path, "r") as f:
            self.partial_key = self._pick_key(f, ["partial", "input", "inputs"])
            self.coarse_key = self._pick_key(f, ["coarse", "pred", "prediction"])
            self.gt_key = self._pick_key(f, ["gt", "complete", "target"])
            self.length = int(f[self.gt_key].shape[0])

    @staticmethod
    def _pick_key(f: h5py.File, names: List[str]) -> str:
        for n in names:
            if n in f:
                return n
        raise KeyError(f"None of keys {names} found in HDF5. Available keys: {list(f.keys())}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(self.seed + idx)
        with h5py.File(self.path, "r") as f:
            partial = np.asarray(f[self.partial_key][idx], dtype=np.float32)[:, :3]
            coarse = np.asarray(f[self.coarse_key][idx], dtype=np.float32)[:, :3]
            gt = np.asarray(f[self.gt_key][idx], dtype=np.float32)[:, :3]
        if self.normalize:
            center = gt.mean(axis=0, keepdims=True)
            scale = np.sqrt(((gt - center) ** 2).sum(axis=1)).max().clip(1e-8)
            partial, coarse, gt = (partial - center) / scale, (coarse - center) / scale, (gt - center) / scale
        return {
            "partial": torch.from_numpy(resample_points_np(partial, self.n_partial, rng, mode=self.resample_mode)),
            "coarse": torch.from_numpy(resample_points_np(coarse, self.n_coarse, rng, mode=self.resample_mode)),
            "gt": torch.from_numpy(resample_points_np(gt, self.n_gt, rng, mode=self.resample_mode)),
            "id": f"{self.path.stem}_{idx:06d}",
        }


class CandidateBankDataset(Dataset):
    """Triplet dataset with a precomputed candidate bank per sample."""

    def __init__(
        self,
        manifest: str | Path,
        n_partial: Optional[int] = 2048,
        n_coarse: Optional[int] = 4096,
        n_gt: Optional[int] = 4096,
        normalize: bool = True,
        seed: int = 0,
        resample_mode: str = "fps",
        candidate_points: Optional[int] = None,
        max_sources: Optional[int] = None,
    ):
        self.rows = read_manifest(manifest)
        if not self.rows:
            raise ValueError(f"Candidate manifest has no rows: {manifest}")
        self.n_partial = None if n_partial is None else int(n_partial)
        self.n_coarse = None if n_coarse is None else int(n_coarse)
        self.n_gt = None if n_gt is None else int(n_gt)
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self.resample_mode = str(resample_mode)
        self.candidate_points = None if candidate_points is None else int(candidate_points)
        self.max_sources = None if max_sources is None else int(max_sources)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rng = np.random.default_rng(self.seed + idx)
        row = self.rows[idx]
        partial = load_point_cloud(row["partial"])
        coarse = load_point_cloud(row["coarse"])
        gt = load_point_cloud(row["gt"])
        if self.normalize:
            center = gt.mean(axis=0, keepdims=True)
            scale = np.sqrt(((gt - center) ** 2).sum(axis=1)).max().clip(1e-8)
            partial, coarse, gt = (partial - center) / scale, (coarse - center) / scale, (gt - center) / scale
        candidate_npz = row.get("candidate_npz") or row.get("candidates")
        if not candidate_npz:
            raise ValueError(f"Candidate manifest row {row.get('sample_id', idx)} is missing candidate_npz")
        payload = np.load(candidate_npz, allow_pickle=True)
        candidates = np.asarray(payload["candidates"], dtype=np.float32)
        if candidates.ndim != 3 or candidates.shape[-1] < 3:
            raise ValueError(f"Expected candidates [S,N,3] in {candidate_npz}, got {candidates.shape}")
        candidates = candidates[:, :, :3]
        if self.candidate_points is not None:
            candidates = np.stack(
                [resample_points_np(points, self.candidate_points, rng, mode=self.resample_mode) for points in candidates],
                axis=0,
            )
        if self.max_sources is not None and candidates.shape[0] > self.max_sources:
            candidates = candidates[: self.max_sources]
        source_ids = np.arange(candidates.shape[0], dtype=np.int64)
        return {
            "partial": torch.from_numpy(resample_points_np(partial, self.n_partial, rng, mode=self.resample_mode)),
            "coarse": torch.from_numpy(resample_points_np(coarse, self.n_coarse, rng, mode=self.resample_mode)),
            "gt": torch.from_numpy(resample_points_np(gt, self.n_gt, rng, mode=self.resample_mode)),
            "candidates": torch.from_numpy(candidates.astype(np.float32)),
            "source_ids": torch.from_numpy(source_ids),
            "id": row.get("id") or Path(row["gt"]).stem,
            "sample_id": row.get("sample_id") or row.get("id") or Path(row["gt"]).stem,
            "category": row.get("category", ""),
            "split": row.get("split", ""),
        }


def make_dataset(cfg: Dict, split: str = "train") -> Dataset:
    dcfg = cfg.get("dataset", cfg)
    dtype = dcfg.get("type", "synthetic")
    seed = int(cfg.get("seed", 0)) + (0 if split == "train" else 100000)
    n_partial = int(dcfg.get("n_partial", 512))
    n_coarse = int(dcfg.get("n_coarse", 1024))
    n_gt = int(dcfg.get("n_gt", 1024))
    resample_mode = str(dcfg.get("resample_mode", "random"))

    if dtype == "synthetic":
        num_key = "num_samples" if split == "train" else "val_samples"
        return SyntheticCompletionDataset(
            num_samples=int(dcfg.get(num_key, dcfg.get("num_samples", 512 if split == "train" else 128))),
            n_partial=n_partial,
            n_coarse=n_coarse,
            n_gt=n_gt,
            shapes=dcfg.get("shapes", "sphere,cube,cylinder,chair,table"),
            severity=dcfg.get("severity", "mixed"),
            seed=seed,
        )
    if dtype == "npz":
        root = dcfg.get("root") if split == "train" else dcfg.get("val_root", dcfg.get("root"))
        return TripletNPZDataset(root, n_partial=n_partial, n_coarse=n_coarse, n_gt=n_gt, normalize=dcfg.get("normalize", True), seed=seed, resample_mode=resample_mode)
    if dtype == "manifest":
        manifest = dcfg.get("manifest") if split == "train" else dcfg.get("val_manifest", dcfg.get("manifest"))
        return ManifestTripletDataset(manifest, n_partial=n_partial, n_coarse=n_coarse, n_gt=n_gt, normalize=dcfg.get("normalize", True), seed=seed, resample_mode=resample_mode)
    if dtype == "candidate_manifest":
        manifest = dcfg.get("manifest") if split == "train" else dcfg.get("val_manifest", dcfg.get("manifest"))
        return CandidateBankDataset(
            manifest,
            n_partial=n_partial,
            n_coarse=n_coarse,
            n_gt=n_gt,
            normalize=dcfg.get("normalize", True),
            seed=seed,
            resample_mode=resample_mode,
            candidate_points=dcfg.get("candidate_points"),
            max_sources=dcfg.get("max_sources"),
        )
    if dtype == "h5":
        path = dcfg.get("path") if split == "train" else dcfg.get("val_path", dcfg.get("path"))
        return H5TripletDataset(path, n_partial=n_partial, n_coarse=n_coarse, n_gt=n_gt, normalize=dcfg.get("normalize", True), seed=seed, resample_mode=resample_mode)
    raise ValueError(f"Unsupported dataset type: {dtype}")
