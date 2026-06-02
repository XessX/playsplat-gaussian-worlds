from __future__ import annotations

from pathlib import Path

import pytest

from playsplat.experiments import filter_scenes, load_scene_registry


def test_scene_registry_loading(tmp_path: Path) -> None:
    ply_path = tmp_path / "scene.ply"
    ply_path.write_text("ply\n", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "scene_id": "scene1",
                "input_path": str(ply_path),
                "category": "indoor_room",
                "source": "independent",
                "split": "benchmark",
                "notes": "Tiny scene.",
            }
        ],
    )

    records = load_scene_registry(registry_path)

    assert len(records) == 1
    assert records[0].scene_id == "scene1"
    assert records[0].input_path == ply_path
    assert records[0].metadata["input_exists"] is True
    assert records[0].warnings == ()


def test_scene_registry_allows_missing_files_with_warning(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.ply"
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "scene_id": "missing_scene",
                "input_path": str(missing_path),
                "category": "outdoor_open",
                "source": "debug",
                "split": "debug",
                "notes": "Missing file is allowed during planning.",
            }
        ],
    )

    records = load_scene_registry(registry_path)

    assert records[0].metadata["input_exists"] is False
    assert "input_path does not exist" in records[0].warnings[0]


def test_duplicate_scene_id_validation(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        [
            _scene("duplicate", "benchmark", "indoor_room", "independent"),
            _scene("duplicate", "debug", "corridor", "internal_debug"),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate scene_id"):
        load_scene_registry(registry_path)


def test_filter_scenes_by_split_category_and_source(tmp_path: Path) -> None:
    registry_path = _write_registry(
        tmp_path,
        [
            _scene("room01", "benchmark", "indoor_room", "independent"),
            _scene("corridor01", "benchmark", "corridor", "independent"),
            _scene("debug01", "debug", "indoor_room", "internal_debug"),
        ],
    )
    records = load_scene_registry(registry_path)

    filtered = filter_scenes(
        records,
        split="benchmark",
        category="indoor_room",
        source="independent",
    )

    assert [record.scene_id for record in filtered] == ["room01"]


def _write_registry(tmp_path: Path, scenes: list[dict[str, str]]) -> Path:
    registry_path = tmp_path / "scenes.yaml"
    lines = ["scenes:"]
    for scene in scenes:
        lines.append(f"  - scene_id: {scene['scene_id']}")
        lines.append(f"    input_path: \"{Path(scene['input_path']).as_posix()}\"")
        lines.append(f"    category: {scene['category']}")
        lines.append(f"    source: {scene['source']}")
        lines.append(f"    split: {scene['split']}")
        lines.append(f"    notes: \"{scene['notes']}\"")
    registry_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return registry_path


def _scene(scene_id: str, split: str, category: str, source: str) -> dict[str, str]:
    return {
        "scene_id": scene_id,
        "input_path": f"missing/{scene_id}.ply",
        "category": category,
        "source": source,
        "split": split,
        "notes": "Fixture scene.",
    }
