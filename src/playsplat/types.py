"""Shared typed scene representations for PlaySplat."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SceneMetadata:
    """Metadata shared by every layer derived from an input scene."""

    scene_id: str
    source_path: Path | None = None
    coordinate_system: str = "right-handed-y-up"
    notes: str = ""


@dataclass
class GaussianSplatScene:
    """Input Gaussian splatting scene before playability layers are inferred."""

    metadata: SceneMetadata
    gaussian_count: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GaussianLayer:
    """Structured Gaussian splatting layer loaded from a source scene file."""

    positions: NDArray[np.float32]
    opacity: NDArray[np.float32] | None = None
    scales: NDArray[np.float32] | None = None
    rotations: NDArray[np.float32] | None = None
    colors: NDArray[np.float32] | None = None
    color_format: str | None = None
    features_dc: NDArray[np.float32] | None = None
    features_rest: NDArray[np.float32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array shapes early so downstream modules can trust them."""

        _validate_matrix("positions", self.positions, columns=3)
        _validate_vector("opacity", self.opacity, rows=self.gaussian_count)
        _validate_matrix("scales", self.scales, rows=self.gaussian_count, columns=3)
        _validate_matrix("rotations", self.rotations, rows=self.gaussian_count, columns=4)
        _validate_matrix("colors", self.colors, rows=self.gaussian_count, columns=3)
        _validate_matrix("features_dc", self.features_dc, rows=self.gaussian_count, columns=3)
        _validate_matrix("features_rest", self.features_rest, rows=self.gaussian_count)

    @property
    def gaussian_count(self) -> int:
        """Number of Gaussians represented by this layer."""

        return int(self.positions.shape[0])


@dataclass
class VisualSplatLayer:
    """Visual layer that preserves the source splat representation."""

    metadata: SceneMetadata
    gaussian_count: int
    representation: str = "3d-gaussian-splatting"
    gaussians: GaussianLayer | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProxyGeometryLayer:
    """Proxy geometry layer for approximate surfaces and object boundaries."""

    metadata: SceneMetadata
    mesh_count: int = 0
    vertex_count: int = 0
    face_count: int = 0
    method: str = "placeholder"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollisionPhysicsLayer:
    """Collision and physics layer derived from proxy geometry."""

    metadata: SceneMetadata
    collider_count: int = 0
    rigid_body_count: int = 0
    mode: str = "static"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationLayer:
    """Walkability and navigation layer for simulated agents."""

    metadata: SceneMetadata
    walkable_region_count: int = 0
    navmesh_polygon_count: int = 0
    agent_radius: float = 0.4
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticSceneLayer:
    """Semantic scene labels attached to regions, objects, or splat clusters."""

    metadata: SceneMetadata
    label_count: int = 0
    labels: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class AffordanceLayer:
    """Interaction affordances inferred from geometry, navigation, and semantics."""

    metadata: SceneMetadata
    affordance_count: int = 0
    labels: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportBundle:
    """Description of an exported representation for a target engine or runtime."""

    target: str
    output_path: Path
    status: str = "planned"


@dataclass
class PlaySplatScene:
    """Layered playability-aware scene representation."""

    metadata: SceneMetadata
    visual: VisualSplatLayer
    proxy_geometry: ProxyGeometryLayer
    collision_physics: CollisionPhysicsLayer
    navigation: NavigationLayer
    semantics: SemanticSceneLayer
    affordances: AffordanceLayer


def _validate_vector(
    name: str,
    array: NDArray[np.float32] | None,
    *,
    rows: int,
) -> None:
    if array is None:
        return
    if array.ndim != 1 or array.shape[0] != rows:
        raise ValueError(f"{name} must have shape ({rows},); got {array.shape}.")


def _validate_matrix(
    name: str,
    array: NDArray[np.float32] | None,
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> None:
    if array is None:
        return
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {array.shape}.")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows; got shape {array.shape}.")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns; got shape {array.shape}.")
