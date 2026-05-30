from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

from .train_refiner import load_config, save_config, train


def apply_cli_overrides(
    cfg: Dict[str, Any],
    *,
    train_manifest: str | None = None,
    val_manifest: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    amp: bool | None = None,
    seed: int | None = None,
    lr: float | None = None,
    device: str | None = None,
    no_local: bool = False,
    lambda_occ: float | None = None,
) -> Dict[str, Any]:
    updated = copy.deepcopy(cfg)
    updated.setdefault("dataset", {})
    updated.setdefault("train", {})
    updated.setdefault("model", {})
    updated.setdefault("loss", {})

    if seed is not None:
        updated["seed"] = int(seed)
    if train_manifest is not None:
        updated["dataset"]["type"] = "manifest"
        updated["dataset"]["manifest"] = train_manifest
    if val_manifest is not None:
        updated["dataset"]["type"] = "manifest"
        updated["dataset"]["val_manifest"] = val_manifest
    if epochs is not None:
        updated["train"]["epochs"] = int(epochs)
    if batch_size is not None:
        updated["train"]["batch_size"] = int(batch_size)
        updated["train"]["val_batch_size"] = int(batch_size)
    if num_workers is not None:
        updated["train"]["num_workers"] = int(num_workers)
    if amp is not None:
        updated["train"]["amp"] = bool(amp)
    if lr is not None:
        updated["train"]["lr"] = float(lr)
    if device is not None:
        updated["train"]["device"] = device
    if no_local:
        updated["model"]["use_local"] = False
    if lambda_occ is not None:
        updated["loss"]["lambda_occ"] = float(lambda_occ)
    return updated


def _write_command_log(out_dir: Path, args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(part) for part in [sys.executable, "-m", "pointr_if.train", *sys.argv[1:]])
    payload = {
        "command": command,
        "args": vars(args),
        "train_manifest": cfg.get("dataset", {}).get("manifest"),
        "val_manifest": cfg.get("dataset", {}).get("val_manifest"),
    }
    (out_dir / "command_log.txt").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_config(cfg, out_dir / "cli_resolved_config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PoinTr-IF on real PoinTr/AdaPoinTr triplets")
    parser.add_argument("--config", default="configs/smoke_synthetic.yaml")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", default="outputs/train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--lambda-occ", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(
        cfg,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        seed=args.seed,
        lr=args.lr,
        device=args.device,
        no_local=args.no_local,
        lambda_occ=args.lambda_occ,
    )
    out_dir = Path(args.out_dir)
    _write_command_log(out_dir, args, cfg)
    summary = train(cfg, out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
