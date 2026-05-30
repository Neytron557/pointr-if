from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_grid(image_dir: Path, out: Path, limit: int = 6, columns: int = 2) -> None:
    images = sorted(image_dir.glob("*.png"))[: int(limit)]
    if not images:
        raise FileNotFoundError(f"No PNG visualizations found in {image_dir}")
    columns = max(1, int(columns))
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(8 * columns, 4.8 * rows))
    if rows == 1 and columns == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif columns == 1:
        axes = [[ax] for ax in axes]
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")
    for idx, image_path in enumerate(images):
        ax = axes[idx // columns][idx % columns]
        ax.imshow(plt.imread(image_path))
        ax.set_title(image_path.stem, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact grid from per-sample qualitative PNGs")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--columns", type=int, default=2)
    args = parser.parse_args()
    make_grid(args.image_dir, args.out, limit=args.limit, columns=args.columns)
    print(args.out)


if __name__ == "__main__":
    main()
