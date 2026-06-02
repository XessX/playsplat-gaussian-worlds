from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_ablation_experiment import (
    format_ablation_scene_id,
    main,
    select_best_runs,
    write_ablation_summary_csv,
    write_ablation_summary_json,
)


def test_ablation_scene_id_formatting() -> None:
    scene_id = format_ablation_scene_id(
        "scene 1/demo",
        voxel_size=0.1,
        target_face_count=25_000,
    )

    assert scene_id == "scene_1_demo_vx0p1_faces25000"


def test_ablation_summary_writers_create_csv_and_json(tmp_path: Path) -> None:
    rows = [
        {
            "scene_id": "scene_vx0p1_faces100",
            "input_path": "scene.ply",
            "voxel_size": 0.1,
            "target_face_count": 100,
            "status": "ready_prototype",
            "collision_face_count": 80,
            "semantic_status": "geometry_semantic_layer",
            "affordance_status": "geometry_affordance_layer",
            "overall_playability_score": 1.0,
        }
    ]

    csv_path = write_ablation_summary_csv(rows, tmp_path / "ablation_summary.csv")
    json_path = write_ablation_summary_json(rows, tmp_path / "ablation_summary.json")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert csv_rows[0]["scene_id"] == "scene_vx0p1_faces100"
    assert csv_rows[0]["voxel_size"] == "0.1"
    assert json_data["runs"][0]["target_face_count"] == 100


def test_ablation_failed_scene_records_error(tmp_path: Path) -> None:
    config_path = _write_ablation_config(tmp_path)
    output_root = tmp_path / "ablations"

    assert main(
        [
            "--input",
            str(tmp_path / "missing.ply"),
            "--scene-id",
            "missing_scene",
            "--base-config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--voxel-sizes",
            "0.1",
            "--target-face-counts",
            "100",
        ]
    ) == 0

    summary = json.loads((output_root / "ablation_summary.json").read_text(encoding="utf-8"))
    row = summary["runs"][0]
    assert row["status"] == "failed"
    assert "does not exist" in row["error"]
    assert (output_root / "missing_scene_vx0p1_faces100").exists()


def test_best_run_selector_logic() -> None:
    rows = [
        _row("low_collision", collision_faces=10, walkable_ratio=0.2, warnings=1),
        _row("high_walkable", collision_faces=40, walkable_ratio=0.8, warnings=0),
        _row("warning_heavy", collision_faces=12, walkable_ratio=0.8, warnings=4),
        {
            "scene_id": "failed",
            "status": "failed",
            "collision_face_count": 1,
            "walkable_area_ratio": 1.0,
            "warning_count": 0,
        },
    ]

    best = select_best_runs(rows)

    assert best["eligible_run_count"] == 3
    assert best["best_lowest_collision_face_count"]["scene_id"] == "low_collision"
    assert best["best_highest_walkable_area_ratio"]["scene_id"] in {
        "high_walkable",
        "warning_heavy",
    }
    assert best["best_balanced"]["scene_id"] == "high_walkable"
    assert "selection_policy" in best


def test_ablation_runner_with_small_ascii_ply_fixture(tmp_path: Path) -> None:
    ply_path = _write_gaussian_fixture(tmp_path, "scene_one.ply")
    config_path = _write_ablation_config(tmp_path)
    output_root = tmp_path / "ablations"

    assert main(
        [
            "--input",
            str(ply_path),
            "--scene-id",
            "scene_one",
            "--base-config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--voxel-sizes",
            "0.1",
            "--target-face-counts",
            "10",
        ]
    ) == 0

    scene_dir = output_root / "scene_one_vx0p1_faces10"
    assert (scene_dir / "ablation_config.yaml").exists()
    assert (scene_dir / "playability_report.json").exists()
    assert (output_root / "ablation_summary.csv").exists()
    assert (output_root / "ablation_summary.json").exists()
    assert (output_root / "best_runs.json").exists()

    summary = json.loads((output_root / "ablation_summary.json").read_text(encoding="utf-8"))
    row = summary["runs"][0]
    assert row["scene_id"] == "scene_one_vx0p1_faces10"
    assert row["voxel_size"] == 0.1
    assert row["target_face_count"] == 10
    assert row["gaussian_count"] == 2


def _row(
    scene_id: str,
    *,
    collision_faces: int,
    walkable_ratio: float,
    warnings: int,
) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "status": "ready_prototype",
        "collision_face_count": collision_faces,
        "walkable_area_ratio": walkable_ratio,
        "warning_count": warnings,
    }


def _write_ablation_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  scene_id: ablation_scene",
                "input:",
                "  path: null",
                "output:",
                f"  directory: \"{(tmp_path / 'unused_outputs').as_posix()}\"",
                "geometry:",
                "  proxy:",
                "    max_grid_voxels: 1000000",
                "  simplification:",
                "    enabled: true",
                "    target_face_count: 10",
                "semantics:",
                "  vocabulary: []",
                "affordance:",
                "  labels: []",
                "export:",
                "  targets: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_gaussian_fixture(tmp_path: Path, filename: str) -> Path:
    ply_path = tmp_path / filename
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property float opacity",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.0 0.0 0.0 0.8 0.01 0.01 0.01 255 255 255",
                "0.1 0.0 0.1 0.9 0.01 0.01 0.01 255 255 255",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return ply_path
