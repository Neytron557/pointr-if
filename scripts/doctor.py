#!/usr/bin/env python
"""Environment diagnostics for the PoinTr-IF project."""
from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_IMPORTS = ["torch", "numpy", "h5py", "yaml", "tqdm", "matplotlib"]
EXPECTED_PATHS = [
    "configs/smoke_synthetic.yaml",
    "configs/titanx_fast_subset.yaml",
    "configs/titanx_pcn_refiner.yaml",
    "scripts/run_smoke.sh",
    "scripts/run_ablation_synthetic.sh",
    "tools/build_triplet_dataset.py",
    "tools/validate_triplet_manifest.py",
    "tools/convert_pointr_predictions.py",
    "src/pointr_if/models.py",
]


def _safe_import(name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - exact import failures are environment-specific.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "version": getattr(module, "__version__", "unknown")}


def _nvidia_smi() -> Dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "error": "nvidia-smi not found on PATH"}
    try:
        result = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - depends on host GPU/driver state.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            name, mem_total, mem_free, driver = parts
            gpus.append(
                {
                    "name": name,
                    "memory_total_mb": mem_total,
                    "memory_free_mb": mem_free,
                    "driver_version": driver,
                }
            )
    return {"available": bool(gpus), "gpus": gpus}


def collect() -> Dict[str, Any]:
    imports = {name: _safe_import(name) for name in REQUIRED_IMPORTS}
    torch_info: Dict[str, Any] = {"import_ok": imports.get("torch", {}).get("ok", False)}
    if torch_info["import_ok"]:
        import torch

        torch_info.update(
            {
                "version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "device_count": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available():
            torch_info["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_mb": int(torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)),
                }
                for i in range(torch.cuda.device_count())
            ]

    return {
        "root": str(ROOT),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "imports": imports,
        "torch": torch_info,
        "nvidia_smi": _nvidia_smi(),
        "expected_paths": {path: (ROOT / path).exists() for path in EXPECTED_PATHS},
    }


def main() -> None:
    info = collect()
    print(json.dumps(info, indent=2))

    missing_imports = [name for name, result in info["imports"].items() if not result["ok"]]
    missing_paths = [path for path, ok in info["expected_paths"].items() if not ok]
    if missing_imports or missing_paths:
        if missing_imports:
            print(f"Missing imports: {', '.join(missing_imports)}", file=sys.stderr)
        if missing_paths:
            print(f"Missing expected paths: {', '.join(missing_paths)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
