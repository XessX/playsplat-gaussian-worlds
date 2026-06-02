"""Configuration helpers for PlaySplat."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineSettings:
    """Runtime settings for a PlaySplat pipeline run."""

    scene_id: str
    input_path: Path | None
    output_dir: Path
    proxy_method: str = "placeholder_density_surface"
    collision_mode: str = "static"
    agent_radius: float = 0.4
    export_targets: tuple[str, ...] = ("unity", "playcanvas", "webgl")
    semantic_vocabulary: tuple[str, ...] = ()
    affordance_labels: tuple[str, ...] = ()
    raw_config: Mapping[str, Any] = field(default_factory=dict)

    def with_overrides(
        self,
        *,
        input_path: Path | None = None,
        output_dir: Path | None = None,
        scene_id: str | None = None,
    ) -> "PipelineSettings":
        """Return settings with optional CLI overrides applied."""

        return replace(
            self,
            input_path=input_path if input_path is not None else self.input_path,
            output_dir=output_dir if output_dir is not None else self.output_dir,
            scene_id=scene_id if scene_id is not None else self.scene_id,
        )


def load_pipeline_settings(config_path: Path) -> PipelineSettings:
    """Load pipeline settings from a YAML file."""

    config = _load_yaml_mapping(config_path)

    project_config = _mapping_at(config, "project")
    input_config = _mapping_at(config, "input")
    output_config = _mapping_at(config, "output")
    geometry_config = _mapping_at(config, "geometry")
    proxy_config = _mapping_at(geometry_config, "proxy")
    physics_config = _mapping_at(config, "physics")
    collision_config = _mapping_at(physics_config, "collision")
    navigation_config = _mapping_at(config, "navigation")
    walkable_config = _mapping_at(navigation_config, "walkable")
    export_config = _mapping_at(config, "export")
    semantics_config = _mapping_at(config, "semantics")
    affordance_config = _mapping_at(config, "affordance")

    scene_id = str(project_config.get("scene_id", "demo_scene"))
    input_path = _optional_path(input_config.get("path"))
    output_dir = _path_or_default(output_config.get("directory"), Path("outputs"))
    proxy_method = str(proxy_config.get("method", "placeholder_density_surface"))
    collision_mode = str(collision_config.get("mode", "static"))
    agent_radius = _float_or_default(walkable_config.get("agent_radius"), 0.4)
    export_targets = _string_tuple(export_config.get("targets"), ("unity", "playcanvas", "webgl"))
    semantic_vocabulary = _string_tuple(semantics_config.get("vocabulary"), ())
    affordance_labels = _string_tuple(affordance_config.get("labels"), ())

    return PipelineSettings(
        scene_id=scene_id,
        input_path=input_path,
        output_dir=output_dir,
        proxy_method=proxy_method,
        collision_mode=collision_mode,
        agent_radius=agent_radius,
        export_targets=export_targets,
        semantic_vocabulary=semantic_vocabulary,
        affordance_labels=affordance_labels,
        raw_config=config,
    )


def _load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root of config: {config_path}")

    return data


def _mapping_at(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected config section '{key}' to be a mapping.")
    return value


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def _path_or_default(value: Any, default: Path) -> Path:
    if value is None or value == "":
        return default
    return Path(str(value))


def _float_or_default(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _string_tuple(value: Any, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("Expected a list of strings.")
    return tuple(str(item) for item in value)
