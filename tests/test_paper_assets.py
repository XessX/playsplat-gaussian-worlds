from __future__ import annotations

import csv
from pathlib import Path

from scripts.generate_paper_assets import (
    clean_ablation_rows,
    generate_figures,
    generate_paper_assets,
    generate_summary_markdown,
    latex_table,
    markdown_table,
    read_csv_rows,
)


def test_clean_table_generation_from_tiny_csv(tmp_path: Path) -> None:
    csv_path = _write_ablation_csv(tmp_path / "ablation_summary.csv")

    rows = read_csv_rows(csv_path)
    clean_rows = clean_ablation_rows(rows)

    assert clean_rows[0]["run_id"] == "scene_vx0p1_faces100"
    assert clean_rows[0]["voxel_size"] == "0.1"
    assert clean_rows[0]["target_face_count"] == "100"
    assert clean_rows[0]["collision_face_reduction_ratio"] == "0.9012"
    assert clean_rows[0]["walkable_area"] == "12.346"
    assert clean_rows[0]["warning_count"] == "0"


def test_markdown_table_generation() -> None:
    rows = [{"run_id": "a", "voxel_size": "0.1", "status": "ready_prototype"}]

    table = markdown_table(rows, ("run_id", "voxel_size", "status"))

    assert "| run_id | voxel_size | status |" in table
    assert "| a | 0.1 | ready_prototype |" in table


def test_latex_table_generation() -> None:
    rows = [{"run_id": "scene_1", "semantic_status": "geometry_semantic_layer"}]

    table = latex_table(rows, ("run_id", "semantic_status"), caption="Tiny table")

    assert "\\begin{tabular}{ll}" in table
    assert "scene\\_1" in table
    assert "geometry\\_semantic\\_layer" in table


def test_summary_markdown_generation(tmp_path: Path) -> None:
    rows = read_csv_rows(_write_ablation_csv(tmp_path / "ablation_summary.csv"))
    best_runs = {
        "best_lowest_collision_face_count": {
            "scene_id": "scene_vx0p1_faces100",
            "voxel_size": 0.1,
            "target_face_count": 100,
            "collision_face_count": 80,
            "walkable_area_ratio": 0.12,
        }
    }

    summary = generate_summary_markdown(rows=rows, best_runs=best_runs, title="Tiny Study")

    assert "# Tiny Study" in summary
    assert "Number of runs: 4" in summary
    assert "Best by collision complexity" in summary
    assert "Some target face counts converged" in summary


def test_figure_generation_creates_png_files(tmp_path: Path) -> None:
    rows = read_csv_rows(_write_ablation_csv(tmp_path / "ablation_summary.csv"))
    figures_dir = tmp_path / "figures"

    figures = generate_figures(rows, figures_dir)

    assert set(figures) == {
        "collision_faces_vs_voxel_size",
        "reduction_ratio_vs_target_faces",
        "walkable_ratio_vs_voxel_size",
        "playability_score_grid",
    }
    for path in figures.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_missing_optional_best_runs_json_does_not_crash(tmp_path: Path) -> None:
    csv_path = _write_ablation_csv(tmp_path / "ablation_summary.csv")
    output_dir = tmp_path / "paper_assets"

    outputs = generate_paper_assets(
        ablation_summary=csv_path,
        best_runs=tmp_path / "missing_best_runs.json",
        output_dir=output_dir,
        title="Tiny Study",
    )

    assert outputs["summary_markdown"].exists()
    assert outputs["latex_tables"].exists()
    assert (output_dir / "tables" / "ablation_summary_clean.csv").exists()
    assert (output_dir / "tables" / "best_runs.md").read_text(encoding="utf-8") == (
        "No best runs available.\n"
    )


def _write_ablation_csv(path: Path) -> Path:
    rows = [
        _row("scene_vx0p1_faces100", 0.1, 100, 1000, 80, 0.901234, 12.34567, 0.12),
        _row("scene_vx0p1_faces200", 0.1, 200, 1000, 81, 0.8999, 12.34567, 0.12),
        _row("scene_vx0p2_faces100", 0.2, 100, 600, 60, 0.9, 8.0, 0.08),
        _row("scene_vx0p2_faces200", 0.2, 200, 600, 200, 0.6667, 8.0, 0.08),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    scene_id: str,
    voxel_size: float,
    target_face_count: int,
    proxy_face_count: int,
    collision_face_count: int,
    reduction_ratio: float,
    walkable_area: float,
    walkable_ratio: float,
) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "input_path": "scene.ply",
        "voxel_size": voxel_size,
        "target_face_count": target_face_count,
        "status": "ready_prototype",
        "error": "",
        "gaussian_count": 10,
        "proxy_vertex_count": 500,
        "proxy_face_count": proxy_face_count,
        "collision_face_count": collision_face_count,
        "collision_face_reduction_ratio": reduction_ratio,
        "collision_face_to_proxy_face_ratio": collision_face_count / proxy_face_count,
        "simplification_status": "target_reached",
        "floor_area": walkable_area,
        "wall_area": 3.0,
        "obstacle_area": 4.0,
        "walkable_area": walkable_area,
        "walkable_area_ratio": walkable_ratio,
        "semantic_status": "geometry_semantic_layer",
        "affordance_status": "geometry_affordance_layer",
        "semantic_label_count": 4,
        "affordance_label_count": 4,
        "export_readiness_score": 1.0,
        "overall_playability_score": 1.0,
        "warning_count": 0,
    }
