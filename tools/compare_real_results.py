from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_metrics(eval_dir: Path) -> dict:
    path = eval_dir / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics.json under {eval_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_table(eval_dirs: list[Path]) -> str:
    lines = [
        "| eval | method | n | chamfer | fscore | CD improvement vs coarse |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for eval_dir in eval_dirs:
        metrics = _load_metrics(eval_dir)
        improvements = metrics.get("improvements", {})
        for method, values in sorted(metrics.get("methods", {}).items()):
            key = f"{method}_vs_coarse_cd_percent"
            improvement = improvements.get(key, 0.0 if method == "coarse" else float("nan"))
            lines.append(
                f"| {eval_dir.name} | {method} | {int(values['n_samples'])} | "
                f"{float(values['chamfer']):.6f} | {float(values['fscore']):.6f} | {float(improvement):.2f}% |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare real PoinTr-IF evaluation result directories")
    parser.add_argument("eval_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/real_results_summary_table.md"))
    args = parser.parse_args()
    table = build_table(args.eval_dirs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
