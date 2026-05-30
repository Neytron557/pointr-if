from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

from .point_ops import normalize_points_np, resample_points_np


def _as_point_array(arr: np.ndarray, source: str, sample_index: int = 0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        if sample_index < 0 or sample_index >= arr.shape[0]:
            raise IndexError(f"sample_index {sample_index} out of range for {source} with shape {arr.shape}")
        arr = arr[sample_index]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected [N, >=3] point cloud in {source}, got {arr.shape}")
    return arr[:, :3].astype(np.float32)


def read_ascii_pcd(path: str | Path) -> np.ndarray:
    """Read an ASCII or simple binary PCD file with x y z fields.

    The parser handles the common ASCII PCN/PoinTr format directly. For binary PCD,
    it attempts a structured numpy read for float x/y/z fields. If the file uses an
    uncommon PCD layout, convert it to .npy/.npz with Open3D first.
    """
    path = Path(path)
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PCD file has no DATA line: {path}")
            decoded = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded.upper().startswith("DATA"):
                data_type = decoded.split()[-1].lower()
                break
        header = {line.split()[0].upper(): line.split()[1:] for line in header_lines if line and not line.startswith("#")}
        fields = header.get("FIELDS", [])
        if not fields:
            fields = header.get("FIELD", [])
        try:
            x_i, y_i, z_i = fields.index("x"), fields.index("y"), fields.index("z")
        except ValueError as exc:
            raise ValueError(f"PCD file must contain x y z fields: {path}") from exc

        if data_type == "ascii":
            arr = np.loadtxt(f, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            return arr[:, [x_i, y_i, z_i]].astype(np.float32)

        if data_type == "binary":
            sizes = list(map(int, header.get("SIZE", [])))
            types = header.get("TYPE", [])
            counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
            if not (len(sizes) == len(types) == len(counts) == len(fields)):
                raise ValueError(f"Unsupported binary PCD header: {path}")
            dtype_fields = []
            for name, typ, size, count in zip(fields, types, sizes, counts):
                if typ.upper() == "F" and size == 4:
                    dt = np.float32
                elif typ.upper() == "F" and size == 8:
                    dt = np.float64
                elif typ.upper() == "I" and size == 4:
                    dt = np.int32
                elif typ.upper() == "U" and size == 4:
                    dt = np.uint32
                else:
                    raise ValueError(f"Unsupported binary PCD field type {typ}{size} in {path}")
                dtype_fields.append((name, dt, (count,) if count > 1 else ()))
            n_points = int(header.get("POINTS", [0])[0])
            data = np.frombuffer(f.read(), dtype=np.dtype(dtype_fields), count=n_points)
            return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)

        raise ValueError(f"Unsupported PCD DATA mode {data_type!r} in {path}")


def write_ascii_pcd(path: str | Path, points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)[:, :3]
    header = "\n".join([
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {points.shape[0]}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {points.shape[0]}",
        "DATA ascii",
    ])
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        np.savetxt(f, points, fmt="%.8f %.8f %.8f")


def load_point_cloud(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        return _as_point_array(arr, str(path))
    if suffix == ".npz":
        data = np.load(path)
        for key in ("points", "pc", "xyz", "partial", "coarse", "gt", "complete"):
            if key in data:
                return _as_point_array(data[key], f"{path}:{key}")
        first = data.files[0]
        return _as_point_array(data[first], f"{path}:{first}")
    if suffix == ".pcd":
        return read_ascii_pcd(path)
    if suffix in {".txt", ".xyz"}:
        arr = np.loadtxt(path, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr[:, :3]
    if suffix in {".h5", ".hdf5"}:
        return load_point_cloud_h5(path)
    raise ValueError(f"Unsupported point-cloud format: {path}")


def _iter_h5_datasets(group: h5py.Group):
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            yield key, item
        elif isinstance(item, h5py.Group):
            for child_key, child in _iter_h5_datasets(item):
                yield f"{key}/{child_key}", child


def load_point_cloud_h5(path: str | Path, dataset_key: Optional[str] = None, sample_index: int = 0) -> np.ndarray:
    """Load a point cloud from a small/simple HDF5 file.

    If no dataset key is provided, common point-cloud keys are preferred, then
    the first dataset with shape [N, >=3] or [B, N, >=3] is used. For batched
    datasets, ``sample_index`` selects the sample.
    """
    path = Path(path)
    preferred = ("points", "pc", "xyz", "partial", "coarse", "pred", "prediction", "gt", "complete", "target", "data")
    with h5py.File(path, "r") as f:
        candidates = dict(_iter_h5_datasets(f))
        if dataset_key is None:
            key = next((name for name in preferred if name in candidates), None)
            if key is None:
                for name, dataset in candidates.items():
                    if dataset.ndim in (2, 3) and dataset.shape[-1] >= 3:
                        key = name
                        break
            if key is None:
                raise ValueError(f"No point-cloud-like dataset found in {path}; available keys: {list(candidates)}")
        else:
            key = dataset_key
            if key not in candidates:
                raise KeyError(f"Dataset {key!r} not found in {path}; available keys: {list(candidates)}")
        arr = np.asarray(candidates[key], dtype=np.float32)
    return _as_point_array(arr, f"{path}:{key}", sample_index=sample_index)


def save_point_cloud(path: str | Path, points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, np.asarray(points, dtype=np.float32)[:, :3])
    elif suffix == ".npz":
        np.savez_compressed(path, points=np.asarray(points, dtype=np.float32)[:, :3])
    elif suffix == ".pcd":
        write_ascii_pcd(path, points)
    elif suffix in {".txt", ".xyz"}:
        np.savetxt(path, np.asarray(points, dtype=np.float32)[:, :3], fmt="%.8f")
    else:
        raise ValueError(f"Unsupported output format: {path}")


def load_triplet_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    def pick(*names: str) -> np.ndarray:
        for name in names:
            if name in data:
                return _as_point_array(data[name], f"{path}:{name}")
        raise KeyError(f"None of {names} present in {path}. Available keys: {data.files}")
    return {
        "partial": pick("partial", "input"),
        "coarse": pick("coarse", "pred", "prediction"),
        "gt": pick("gt", "complete", "target"),
    }


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        legacy_required = {"partial", "coarse", "gt"}
        real_required = {"partial_path", "coarse_path", "gt_path"}
        if not (legacy_required.issubset(fieldnames) or real_required.issubset(fieldnames)):
            raise ValueError(
                "Manifest must contain either columns "
                f"{sorted(legacy_required)} or {sorted(real_required)}; got {reader.fieldnames}"
            )
        rows = list(reader)
    base = path.parent
    for row in rows:
        if "id" not in row or not row.get("id"):
            row["id"] = row.get("sample_id") or row.get("model_id") or Path(row.get("gt_path", row.get("gt", ""))).stem
        if "sample_id" not in row or not row.get("sample_id"):
            row["sample_id"] = row["id"]
        for canonical, path_key in (("partial", "partial_path"), ("coarse", "coarse_path"), ("gt", "gt_path")):
            value = row.get(canonical) or row.get(path_key)
            if not value:
                raise ValueError(f"Manifest row {row.get('id', '<unknown>')} is missing {canonical}/{path_key}")
            p = Path(value)
            if not p.is_absolute():
                value = str((base / p).resolve())
            else:
                value = str(p)
            row[canonical] = value
            row[path_key] = value
    return rows


def write_manifest(path: str | Path, rows: Iterable[Dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "partial", "coarse", "gt"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def find_pc_files(root: str | Path) -> List[Path]:
    root = Path(root)
    exts = {".npy", ".npz", ".pcd", ".txt", ".xyz", ".h5", ".hdf5"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
