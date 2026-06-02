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
    proxy_enabled: bool = True
    proxy_method: str = "placeholder_density_surface"
    opacity_threshold: float = 0.01
    bounds_quantile: float = 0.995
    max_gaussians: int | None = 300_000
    voxel_size: float = 0.05
    density_threshold: float = 1.0
    padding_voxels: int = 2
    smooth_sigma: float = 0.0
    max_grid_voxels: int = 20_000_000
    proxy_output_mesh: Path = Path("proxy_mesh.obj")
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
    proxy_enabled = _bool_or_default(proxy_config.get("enabled"), True)
    proxy_method = str(proxy_config.get("method", "placeholder_density_surface"))
    opacity_threshold = _float_or_default(proxy_config.get("opacity_threshold"), 0.01)
    bounds_quantile = _float_or_default(proxy_config.get("bounds_quantile"), 0.995)
    max_gaussians = _optional_int(proxy_config.get("max_gaussians"), 300_000)
    voxel_size = _float_or_default(proxy_config.get("voxel_size"), 0.05)
    density_threshold = _float_or_default(proxy_config.get("density_threshold"), 1.0)
    padding_voxels = _int_or_default(proxy_config.get("padding_voxels"), 2)
    smooth_sigma = _float_or_default(proxy_config.get("smooth_sigma"), 0.0)
    max_grid_voxels = _int_or_default(proxy_config.get("max_grid_voxels"), 20_000_000)
    proxy_output_mesh = _path_or_default(proxy_config.get("output_mesh"), Path("proxy_mesh.obj"))
    collision_mode = str(collision_config.get("mode", "static"))
    agent_radius = _float_or_default(walkable_config.get("agent_radius"), 0.4)
    export_targets = _string_tuple(export_config.get("targets"), ("unity", "playcanvas", "webgl"))
    semantic_vocabulary = _string_tuple(semantics_config.get("vocabulary"), ())
    affordance_labels = _string_tuple(affordance_config.get("labels"), ())

    return PipelineSettings(
        scene_id=scene_id,
        input_path=input_path,
        output_dir=output_dir,
        proxy_enabled=proxy_enabled,
        proxy_method=proxy_method,
        opacity_threshold=opacity_threshold,
        bounds_quantile=bounds_quantile,
        max_gaussians=max_gaussians,
        voxel_size=voxel_size,
        density_threshold=density_threshold,
        padding_voxels=padding_voxels,
        smooth_sigma=smooth_sigma,
        max_grid_voxels=max_grid_voxels,
        proxy_output_mesh=proxy_output_mesh,
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


def _int_or_default(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _optional_int(value: Any, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    return int(value)


def _bool_or_default(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected boolean value; got {value!r}.")


def _string_tuple(value: Any, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("Expected a list of strings.")
    return tuple(str(item) for item in value)
