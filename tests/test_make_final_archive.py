from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from make_final_archive import build_archive


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_archive_includes_reproducible_artifacts_and_excludes_heavy_files(tmp_path: Path) -> None:
    _write(tmp_path / "README.md")
    _write(tmp_path / "requirements.txt")
    _write(tmp_path / "src/pointr_if/models.py")
    _write(tmp_path / "tools/analyze_real_results_stats.py")
    _write(tmp_path / "configs/run.yaml")
    _write(tmp_path / "docs/report.md")
    _write(tmp_path / "scripts/run.sh")
    _write(tmp_path / "outputs/run/test_eval/metrics.json")
    _write(tmp_path / "outputs/run/test_eval/stats/paired_stats.md")
    _write(tmp_path / "outputs/run/test_eval/qualitative_grid.png")
    _write(tmp_path / "data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv")
    _write(tmp_path / ".venv/pyvenv.cfg")
    _write(tmp_path / "external/PoinTr/data/sample.pcd")
    _write(tmp_path / "outputs/run/best_model.pt")
    _write(tmp_path / "outputs/run/test_eval/predictions/sample_refined.npy")
    _write(tmp_path / "deliverables/old.zip")

    out = tmp_path / "deliverables/final.zip"
    summary = build_archive(tmp_path, out)

    assert summary.n_files > 0
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "README.md" in names
    assert "src/pointr_if/models.py" in names
    assert "outputs/run/test_eval/metrics.json" in names
    assert "outputs/run/test_eval/stats/paired_stats.md" in names
    assert "data/real_projected_shapenet55_adapointr_predictions/manifests/test_triplets.csv" in names
    assert ".venv/pyvenv.cfg" not in names
    assert "external/PoinTr/data/sample.pcd" not in names
    assert "outputs/run/best_model.pt" not in names
    assert "outputs/run/test_eval/predictions/sample_refined.npy" not in names
    assert "deliverables/old.zip" not in names
