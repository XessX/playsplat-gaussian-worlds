from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.generate_benchmark_assets import generate_benchmark_assets
from scripts.run_benchmark import (
    generate_benchmark_report,
    main,
    write_benchmark_report,
    write_benchmark_summary_csv,
    write_benchmark_summary_json,
)


def test_benchmark_summary_writing(tmp_path: Path) -> None:
    rows = [_summary_row("scene1", "ready_prototype")]

    csv_path = write_benchmark_summary_csv(rows, tmp_path / "benchmark_summary.csv")
    json_path = write_benchmark_summary_json(rows, tmp_path / "benchmark_summary.json")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert csv_rows[0]["scene_id"] == "scene1"
    assert csv_rows[0]["category"] == "indoor_room"
    assert json_data["scenes"][0]["collision_face_count"] == 80


def test_benchmark_report_generation(tmp_path: Path) -> None:
    rows = [
        _summary_row("scene1", "ready_prototype"),
        _summary_row("scene2", "failed", error="missing file"),
    ]

    report = generate_benchmark_report(rows)
    report_path = write_benchmark_report(rows, tmp_path / "benchmark_report.md")

    assert "Total scenes: 2" in report
    assert "Successful scenes: 1" in report
    assert "Failed scenes: 1" in report
    assert "| category | scenes | successful |" in report
    assert report_path.exists()


def test_benchmark_failure_handling_for_missing_scene(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path, tmp_path / "missing.ply")
    config_path = _write_benchmark_config(tmp_path)
    output_root = tmp_path / "benchmark"

    assert main(
        [
            "--scene-registry",
            str(registry_path),
            "--base-config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--split",
            "benchmark",
            "--voxel-size",
            "0.1",
            "--target-face-count",
            "10",
        ]
    ) == 0

    summary = json.loads((output_root / "benchmark_summary.json").read_text(encoding="utf-8"))
    row = summary["scenes"][0]
    assert row["status"] == "failed"
    assert "does not exist" in row["error"]
    assert (output_root / "scene1").exists()


def test_tiny_benchmark_run_with_generated_ascii_ply(tmp_path: Path) -> None:
    ply_path = _write_gaussian_fixture(tmp_path / "scene.ply")
    registry_path = _write_registry(tmp_path, ply_path)
    config_path = _write_benchmark_config(tmp_path)
    output_root = tmp_path / "benchmark"

    assert main(
        [
            "--scene-registry",
            str(registry_path),
            "--base-config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--split",
            "benchmark",
            "--voxel-size",
            "0.1",
            "--target-face-count",
            "10",
        ]
    ) == 0

    scene_dir = output_root / "scene1"
    assert (scene_dir / "benchmark_config.yaml").exists()
    assert (scene_dir / "playability_report.json").exists()
    assert (output_root / "benchmark_summary.csv").exists()
    assert (output_root / "benchmark_report.md").exists()
    summary = json.loads((output_root / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["scenes"][0]["scene_id"] == "scene1"
    assert summary["scenes"][0]["category"] == "indoor_room"
    assert summary["scenes"][0]["gaussian_count"] == 2


def test_benchmark_asset_generation(tmp_path: Path) -> None:
    summary_path = tmp_path / "benchmark_summary.csv"
    write_benchmark_summary_csv([_summary_row("scene1", "ready_prototype")], summary_path)

    outputs = generate_benchmark_assets(
        benchmark_summary=summary_path,
        output_dir=tmp_path / "paper_assets" / "benchmark",
        title="Tiny Benchmark",
    )

    assert outputs["benchmark_summary_clean_csv"].exists()
    assert outputs["benchmark_summary_markdown"].exists()
    assert outputs["benchmark_summary_latex"].exists()
    assert outputs["benchmark_summary"].exists()
    assert outputs["collision_faces_by_scene"].exists()
    assert outputs["walkable_ratio_by_scene"].exists()
    assert outputs["reduction_ratio_by_scene"].exists()


def _summary_row(scene_id: str, status: str, *, error: str = "") -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "category": "indoor_room",
        "source": "independent",
        "split": "benchmark",
        "input_path": "scene.ply",
        "status": status,
        "error": error,
        "gaussian_count": 2,
        "proxy_face_count": 200,
        "collision_face_count": 80,
        "collision_face_reduction_ratio": 0.6,
        "simplification_status": "target_reached",
        "floor_area": 1.0,
        "wall_area": 2.0,
        "obstacle_area": 3.0,
        "walkable_area": 1.0,
        "walkable_area_ratio": 0.25,
        "semantic_status": "geometry_semantic_layer",
        "affordance_status": "geometry_affordance_layer",
        "semantic_label_count": 4,
        "affordance_label_count": 4,
        "export_readiness_score": 1.0,
        "overall_playability_score": 1.0,
        "warning_count": 0,
    }


def _write_registry(tmp_path: Path, input_path: Path) -> Path:
    registry_path = tmp_path / "scenes.yaml"
    registry_path.write_text(
        "\n".join(
            [
                "scenes:",
                "  - scene_id: scene1",
                f"    input_path: \"{input_path.as_posix()}\"",
                "    category: indoor_room",
                "    source: independent",
                "    split: benchmark",
                "    notes: Tiny benchmark fixture.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return registry_path


def _write_benchmark_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  scene_id: benchmark_scene",
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


def _write_gaussian_fixture(path: Path) -> Path:
    path.write_text(
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
    return path
