#!/usr/bin/env python
"""Evaluate a base or MVC fine-tuned AdaPoinTr checkpoint on a manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_adapointr_multiview_consistency import evaluate_model, load_adapointr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-partial", type=int, default=2048)
    parser.add_argument("--n-output", type=int, default=4096)
    parser.add_argument("--n-gt", type=int, default=4096)
    parser.add_argument("--eval-seed", type=int, default=200570)
    parser.add_argument("--input-seed", type=int, default=570)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_adapointr(args.config, args.checkpoint, device)
    summary = evaluate_model(
        model,
        manifest=args.manifest,
        out_dir=args.out_dir,
        device=device,
        n_partial=args.n_partial,
        n_output=args.n_output,
        n_gt=args.n_gt,
        eval_seed=args.eval_seed,
        input_seed=args.input_seed,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        save_predictions=args.save_predictions,
    )
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "resolved_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
