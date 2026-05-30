#!/usr/bin/env python
"""Generate synthetic triplets to test the full npz dataset path."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pointr_if.synthetic import generate_completion_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/synthetic_npz")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--n-partial", type=int, default=256)
    parser.add_argument("--n-coarse", type=int, default=512)
    parser.add_argument("--n-gt", type=int, default=512)
    parser.add_argument("--severity", default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_samples):
        s = generate_completion_sample(i, n_gt=args.n_gt, n_partial=args.n_partial, n_coarse=args.n_coarse, severity=args.severity, seed=args.seed)
        np.savez_compressed(out / f"{s['id']}.npz", partial=s["partial"], coarse=s["coarse"], gt=s["gt"])
    print(f"Wrote {args.num_samples} synthetic triplets to {out}")


if __name__ == "__main__":
    main()
