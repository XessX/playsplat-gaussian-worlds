"""Shared typed scene representations for PlaySplat."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
class VisualSplatLayer:
    """Visual layer that preserves the source splat representation."""

    metadata: SceneMetadata
    gaussian_count: int
    representation: str = "3d-gaussian-splatting"
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
