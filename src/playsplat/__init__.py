"""PlaySplat research prototype package."""

from playsplat.pipeline import PipelineResult, run_pipeline
from playsplat.types import (
    AffordanceLayer,
    CollisionPhysicsLayer,
    ExportBundle,
    FilteredGaussianLayer,
    GaussianLayer,
    GaussianSplatScene,
    NavigationLayer,
    PlaySplatScene,
    ProxyGeometryLayer,
    ProxyMesh,
    SceneStructure,
    SceneMetadata,
    SemanticSceneLayer,
    SurfaceRegion,
    VoxelOccupancyGrid,
    VisualSplatLayer,
)

__all__ = [
    "AffordanceLayer",
    "CollisionPhysicsLayer",
    "ExportBundle",
    "FilteredGaussianLayer",
    "GaussianLayer",
    "GaussianSplatScene",
    "NavigationLayer",
    "PipelineResult",
    "PlaySplatScene",
    "ProxyGeometryLayer",
    "ProxyMesh",
    "SceneStructure",
    "SceneMetadata",
    "SemanticSceneLayer",
    "SurfaceRegion",
    "VoxelOccupancyGrid",
    "VisualSplatLayer",
    "run_pipeline",
]
