"""Scene registry loading and filtering for benchmark experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SCENE_FIELDS = ("scene_id", "input_path", "category", "source", "split", "notes")


@dataclass(frozen=True)
class SceneRecord:
    """One scene entry from a PlaySplat scene registry."""

    scene_id: str
    input_path: Path
    category: str
    source: str
    split: str
    notes: str
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def load_scene_registry(path: str | Path) -> list[SceneRecord]:
    """Load and validate a scene registry YAML file."""

    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Scene registry not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root of scene registry: {registry_path}")

    scenes_data = data.get("scenes")
    if not isinstance(scenes_data, list):
        raise ValueError("Scene registry must contain a 'scenes' list.")

    seen_scene_ids: set[str] = set()
    records: list[SceneRecord] = []
    for index, scene_data in enumerate(scenes_data):
        if not isinstance(scene_data, dict):
            raise ValueError(f"Scene entry at index {index} must be a mapping.")
        missing_fields = [
            field_name for field_name in REQUIRED_SCENE_FIELDS if field_name not in scene_data
        ]
        if missing_fields:
            raise ValueError(
                f"Scene entry at index {index} is missing required fields: "
                + ", ".join(missing_fields)
            )

        scene_id = str(scene_data["scene_id"])
        if scene_id in seen_scene_ids:
            raise ValueError(f"Duplicate scene_id in scene registry: {scene_id}")
        seen_scene_ids.add(scene_id)

        input_path = Path(str(scene_data["input_path"]))
        warnings = _warnings_for_scene(scene_id, input_path)
        metadata = {
            "input_exists": input_path.exists(),
            "registry_path": str(registry_path),
        }
        if warnings:
            metadata["warnings"] = list(warnings)

        records.append(
            SceneRecord(
                scene_id=scene_id,
                input_path=input_path,
                category=str(scene_data["category"]),
                source=str(scene_data["source"]),
                split=str(scene_data["split"]),
                notes=str(scene_data["notes"]),
                warnings=warnings,
                metadata=metadata,
            )
        )
    return records


def filter_scenes(
    records: Sequence[SceneRecord],
    *,
    split: str | None = None,
    category: str | None = None,
    source: str | None = None,
) -> list[SceneRecord]:
    """Filter scene records by optional split, category, and source."""

    return [
        record
        for record in records
        if (split is None or record.split == split)
        and (category is None or record.category == category)
        and (source is None or record.source == source)
    ]


def _warnings_for_scene(scene_id: str, input_path: Path) -> tuple[str, ...]:
    warnings: list[str] = []
    if not input_path.exists():
        warnings.append(f"input_path does not exist for scene '{scene_id}': {input_path}")
    return tuple(warnings)
