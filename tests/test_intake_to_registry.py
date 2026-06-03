from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from scripts.intake_to_registry import (
    INTAKE_COLUMNS,
    load_registry_mapping,
    read_intake_csv,
    update_registry_from_intake,
)


def test_intake_csv_parsing(tmp_path: Path) -> None:
    csv_path = _write_intake_csv(
        tmp_path,
        [
            {
                "scene_id": "room01",
                "category": "indoor_room",
                "capture_location_type": "indoor",
                "staged_path": "data/scenes/room01/point_cloud.ply",
                "image_count": "120",
                "training_iteration": "30000",
                "notes": "Independent room.",
                "privacy_checked": "yes",
                "ready_for_benchmark": "yes",
            }
        ],
    )

    result = read_intake_csv(csv_path)

    assert result.rows_read == 1
    assert result.malformed_rows == 0
    assert result.warnings == ()
    assert result.rows[0].scene_id == "room01"
    assert result.rows[0].category == "indoor_room"
    assert result.rows[0].input_path == "data/scenes/room01/point_cloud.ply"
    assert result.rows[0].is_ready is True


def test_ready_for_benchmark_filtering(tmp_path: Path) -> None:
    csv_path = _write_intake_csv(
        tmp_path,
        [
            {
                "scene_id": "room01",
                "category": "indoor_room",
                "staged_path": "data/scenes/room01/point_cloud.ply",
                "ready_for_benchmark": "true",
            },
            {
                "scene_id": "corridor01",
                "category": "corridor",
                "staged_path": "data/scenes/corridor01/point_cloud.ply",
                "ready_for_benchmark": "no",
            },
        ],
    )
    registry_path = tmp_path / "scenes.local.yaml"

    summary = update_registry_from_intake(csv_path, registry_path)
    registry = load_registry_mapping(registry_path)

    assert summary.rows_read == 2
    assert summary.ready_rows == 1
    assert summary.skipped_rows == 1
    assert summary.registry_records_written == 1
    assert [scene["scene_id"] for scene in registry["scenes"]] == ["room01"]


def test_existing_debug_scene_is_preserved_even_with_overwrite(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "scene_id": "scene1",
                "input_path": "data/scenes/scene1/point_cloud.ply",
                "category": "outdoor_complex",
                "source": "internal_debug",
                "split": "debug",
                "notes": "Debug scene.",
            }
        ],
    )
    csv_path = _write_intake_csv(
        tmp_path,
        [
            {
                "scene_id": "scene1",
                "category": "indoor_room",
                "staged_path": "data/scenes/scene1_replacement/point_cloud.ply",
                "notes": "Should not replace debug scene.",
                "ready_for_benchmark": "yes",
            }
        ],
    )

    summary = update_registry_from_intake(csv_path, registry_path, overwrite=True)
    registry = load_registry_mapping(registry_path)

    assert summary.registry_records_written == 0
    assert summary.results[0].status == "preserved_debug_scene"
    assert registry["scenes"][0]["source"] == "internal_debug"
    assert registry["scenes"][0]["split"] == "debug"
    assert registry["scenes"][0]["notes"] == "Debug scene."


def test_dry_run_does_not_write_registry(tmp_path: Path) -> None:
    csv_path = _write_intake_csv(
        tmp_path,
        [
            {
                "scene_id": "room01",
                "category": "indoor_room",
                "staged_path": "data/scenes/room01/point_cloud.ply",
                "ready_for_benchmark": "1",
            }
        ],
    )
    registry_path = tmp_path / "scenes.local.yaml"

    summary = update_registry_from_intake(csv_path, registry_path, dry_run=True)

    assert summary.registry_records_written == 0
    assert summary.results[0].status == "dry_run_add"
    assert not registry_path.exists()


def test_overwrite_updates_existing_benchmark_record(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "scene_id": "room01",
                "input_path": "data/scenes/room01/old_point_cloud.ply",
                "category": "indoor_room",
                "source": "independent",
                "split": "benchmark",
                "notes": "Old notes.",
            }
        ],
    )
    csv_path = _write_intake_csv(
        tmp_path,
        [
            {
                "scene_id": "room01",
                "category": "corridor",
                "staged_path": "data/scenes/room01/point_cloud.ply",
                "notes": "Replacement notes.",
                "ready_for_benchmark": "yes",
            }
        ],
    )

    rejected = update_registry_from_intake(csv_path, registry_path)
    updated = update_registry_from_intake(csv_path, registry_path, overwrite=True)
    registry = load_registry_mapping(registry_path)

    assert rejected.results[0].status == "duplicate_scene_id"
    assert updated.results[0].status == "updated"
    assert registry["scenes"][0]["category"] == "corridor"
    assert registry["scenes"][0]["input_path"] == "data/scenes/room01/point_cloud.ply"
    assert registry["scenes"][0]["notes"] == "Replacement notes."


def test_malformed_intake_rows_are_skipped_with_warnings(tmp_path: Path) -> None:
    csv_path = _write_intake_csv(
        tmp_path,
        [
            {
                "scene_id": "",
                "category": "indoor_room",
                "staged_path": "data/scenes/bad/point_cloud.ply",
                "ready_for_benchmark": "yes",
            },
            {
                "scene_id": "room01",
                "category": "indoor_room",
                "trained_3dgs_output_path": "D:/trained/room01/point_cloud.ply",
                "ready_for_benchmark": "yes",
            },
        ],
    )
    registry_path = tmp_path / "scenes.local.yaml"

    summary = update_registry_from_intake(csv_path, registry_path)
    registry = load_registry_mapping(registry_path)

    assert summary.rows_read == 2
    assert summary.ready_rows == 1
    assert summary.skipped_rows == 1
    assert any("missing scene_id" in warning for warning in summary.warnings)
    assert [scene["scene_id"] for scene in registry["scenes"]] == ["room01"]
    assert registry["scenes"][0]["input_path"] == "D:/trained/room01/point_cloud.ply"


def _write_intake_csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    csv_path = tmp_path / "scene_intake.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INTAKE_COLUMNS))
        writer.writeheader()
        for row in rows:
            full_row = {column: "" for column in INTAKE_COLUMNS}
            full_row.update(row)
            writer.writerow(full_row)
    return csv_path


def _write_registry(tmp_path: Path, scenes: list[dict[str, str]]) -> Path:
    registry_path = tmp_path / "scenes.local.yaml"
    registry: dict[str, Any] = {"scenes": scenes}
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return registry_path
