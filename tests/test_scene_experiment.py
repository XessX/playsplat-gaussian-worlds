from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_scene_experiment import (
    main,
    write_experiment_summary_csv,
    write_experiment_summary_json,
)


def test_scene_experiment_creates_per_scene_outputs_and_summary(tmp_path: Path) -> None:
    ply_path = _write_gaussian_fixture(tmp_path, "scene_one.ply")
    config_path = _write_experiment_config(tmp_path)
    output_root = tmp_path / "experiments"

    assert main(
        [
            "--input",
            str(ply_path),
            "--scene-id",
            "scene_one",
            "--output-root",
            str(output_root),
            "--config",
            str(config_path),
        ]
    ) == 0

    scene_dir = output_root / "scene_one"
    assert scene_dir.exists()
    assert (scene_dir / "gaussian_stats.json").exists()
    assert (scene_dir / "proxy_mesh.obj").exists()
    assert (scene_dir / "playability_report.json").exists()
    assert (scene_dir / "playability_metrics.csv").exists()
    assert (output_root / "experiment_summary.csv").exists()
    assert (output_root / "experiment_summary.json").exists()

    summary = json.loads((output_root / "experiment_summary.json").read_text(encoding="utf-8"))
    assert summary["scenes"][0]["scene_id"] == "scene_one"
    assert summary["scenes"][0]["status"] in {"partial", "ready_prototype", "placeholder"}
    assert summary["scenes"][0]["gaussian_count"] == 2


def test_experiment_summary_writers_create_csv_and_json(tmp_path: Path) -> None:
    rows = [
        {
            "scene_id": "a",
            "input_path": "a.ply",
            "status": "ready_prototype",
            "gaussian_count": 2,
            "overall_playability_score": 0.75,
        }
    ]

    csv_path = write_experiment_summary_csv(rows, tmp_path / "summary.csv")
    json_path = write_experiment_summary_json(rows, tmp_path / "summary.json")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert csv_rows[0]["scene_id"] == "a"
    assert json_data["scenes"][0]["overall_playability_score"] == 0.75


def test_scene_experiment_failed_scene_records_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_scene.ply"
    config_path = _write_experiment_config(tmp_path)
    output_root = tmp_path / "experiments"

    assert main(
        [
            "--input",
            str(missing_path),
            "--output-root",
            str(output_root),
            "--config",
            str(config_path),
        ]
    ) == 0

    summary = json.loads((output_root / "experiment_summary.json").read_text(encoding="utf-8"))
    row = summary["scenes"][0]
    assert row["status"] == "failed"
    assert "does not exist" in row["error"]
    assert (output_root / "missing_scene").exists()


def _write_experiment_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  scene_id: experiment_scene",
                "input:",
                "  path: null",
                "output:",
                f"  directory: \"{(tmp_path / 'unused_outputs').as_posix()}\"",
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
