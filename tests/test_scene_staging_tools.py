from __future__ import annotations

from pathlib import Path

import yaml

from scripts.discover_candidate_plys import discover_candidate_plys
from scripts.stage_independent_scenes import (
    load_registry_mapping,
    parse_scene_spec,
    stage_scenes,
)
from scripts.validate_scene_registry import validate_scene_registry


def test_parse_scene_spec() -> None:
    spec = parse_scene_spec(
        'scene_id=room01,input="D:/Scenes/room/point_cloud.ply",'
        'category=indoor_room,notes="Independent indoor room scene"',
    )

    assert spec.scene_id == "room01"
    assert spec.input_path == Path("D:/Scenes/room/point_cloud.ply")
    assert spec.category == "indoor_room"
    assert spec.source == "independent"
    assert spec.split == "benchmark"
    assert spec.notes == "Independent indoor room scene"


def test_dry_run_staging_does_not_copy_or_modify_registry(tmp_path: Path) -> None:
    input_path = _write_tiny_ply(tmp_path / "source" / "point_cloud.ply")
    registry_path = tmp_path / "scenes.local.yaml"
    spec = parse_scene_spec(
        f"scene_id=room01,input={input_path.as_posix()},category=indoor_room",
    )

    results = stage_scenes(
        [spec],
        registry_path=registry_path,
        data_root=tmp_path / "data" / "scenes",
        dry_run=True,
    )

    assert results[0].status == "dry_run"
    assert not (tmp_path / "data" / "scenes" / "room01" / "point_cloud.ply").exists()
    assert not registry_path.exists()


def test_staging_copies_tiny_ply_fixture(tmp_path: Path) -> None:
    input_path = _write_tiny_ply(tmp_path / "source" / "point_cloud.ply")
    registry_path = tmp_path / "scenes.local.yaml"
    spec = parse_scene_spec(
        f"scene_id=room01,input={input_path.as_posix()},"
        "category=indoor_room,notes=Independent room",
    )

    results = stage_scenes(
        [spec],
        registry_path=registry_path,
        data_root=tmp_path / "data" / "scenes",
        mode="copy",
    )

    staged_path = tmp_path / "data" / "scenes" / "room01" / "point_cloud.ply"
    registry = load_registry_mapping(registry_path)
    assert results[0].status == "staged"
    assert staged_path.exists()
    assert staged_path.read_text(encoding="utf-8") == input_path.read_text(encoding="utf-8")
    assert registry["scenes"][0]["scene_id"] == "room01"
    assert registry["scenes"][0]["input_path"] == staged_path.as_posix()


def test_registry_update_preserves_existing_debug_scene(tmp_path: Path) -> None:
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
    input_path = _write_tiny_ply(tmp_path / "room" / "point_cloud.ply")
    spec = parse_scene_spec(
        f"scene_id=room01,input={input_path.as_posix()},category=indoor_room",
    )

    stage_scenes(
        [spec],
        registry_path=registry_path,
        data_root=tmp_path / "data" / "scenes",
    )

    registry = load_registry_mapping(registry_path)
    scene_ids = [scene["scene_id"] for scene in registry["scenes"]]
    assert scene_ids == ["scene1", "room01"]
    assert registry["scenes"][0]["split"] == "debug"


def test_duplicate_scene_id_rejected_unless_overwrite(tmp_path: Path) -> None:
    original_input = _write_tiny_ply(tmp_path / "original" / "point_cloud.ply")
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "scene_id": "room01",
                "input_path": original_input.as_posix(),
                "category": "indoor_room",
                "source": "independent",
                "split": "benchmark",
                "notes": "Original.",
            }
        ],
    )
    replacement_input = _write_tiny_ply(tmp_path / "replacement" / "point_cloud.ply")
    spec = parse_scene_spec(
        f"scene_id=room01,input={replacement_input.as_posix()},"
        "category=indoor_room,notes=Replacement",
    )

    rejected = stage_scenes(
        [spec],
        registry_path=registry_path,
        data_root=tmp_path / "data" / "scenes",
    )
    accepted = stage_scenes(
        [spec],
        registry_path=registry_path,
        data_root=tmp_path / "data" / "scenes",
        overwrite=True,
    )

    registry = load_registry_mapping(registry_path)
    assert rejected[0].status == "duplicate_scene_id"
    assert accepted[0].status == "staged"
    assert registry["scenes"][0]["notes"] == "Replacement"


def test_discovery_excludes_scientific_reports_and_playsplat_outputs(tmp_path: Path) -> None:
    independent = _write_tiny_ply(tmp_path / "Datasets" / "room01" / "point_cloud.ply")
    scientific = _write_tiny_ply(
        tmp_path / "sparse3d_scirep_starter" / "results" / "point_cloud.ply",
    )
    generated = _write_tiny_ply(
        tmp_path / "playsplat" / "outputs" / "scene1" / "point_cloud.ply",
    )

    candidates = discover_candidate_plys(roots=[tmp_path])
    by_path = {candidate.path: candidate for candidate in candidates}

    assert by_path[independent].excluded is False
    assert by_path[scientific].excluded is True
    assert by_path[generated].excluded is True
    assert by_path[scientific].suspected_source == "scientific_reports_sparse_view_project"
    assert by_path[generated].suspected_source == "playsplat_generated_output"


def test_registry_validator_flags_missing_benchmark_files(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "scene_id": "room01",
                "input_path": (tmp_path / "missing.ply").as_posix(),
                "category": "indoor_room",
                "source": "independent",
                "split": "benchmark",
                "notes": "Missing benchmark file.",
            }
        ],
    )

    report = validate_scene_registry(registry_path)

    assert report.is_valid is False
    assert report.benchmark_scenes == 1
    assert report.missing_benchmark_files


def test_registry_validator_warns_when_fewer_than_five_benchmark_scenes(tmp_path: Path) -> None:
    input_path = _write_tiny_ply(tmp_path / "room" / "point_cloud.ply")
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "scene_id": "room01",
                "input_path": input_path.as_posix(),
                "category": "indoor_room",
                "source": "independent",
                "split": "benchmark",
                "notes": "Only one benchmark scene.",
            }
        ],
    )

    report = validate_scene_registry(registry_path)

    assert report.is_valid is True
    assert report.benchmark_scenes == 1
    assert any("fewer than 5 benchmark scenes" in warning for warning in report.warnings)


def _write_tiny_ply(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0.0 0.0 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_registry(tmp_path: Path, scenes: list[dict[str, str]]) -> Path:
    registry_path = tmp_path / "scenes.local.yaml"
    registry_path.write_text(yaml.safe_dump({"scenes": scenes}, sort_keys=False), encoding="utf-8")
    return registry_path
