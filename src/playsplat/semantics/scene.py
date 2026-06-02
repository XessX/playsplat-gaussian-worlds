"""Geometry-derived semantic scene layer inference."""

from __future__ import annotations

from collections.abc import Sequence

from playsplat.types import (
    ProxyGeometryLayer,
    SceneStructure,
    SemanticSceneLayer,
    SurfaceRegion,
    VisualSplatLayer,
)


def infer_semantic_layer(
    visual_layer: VisualSplatLayer,
    proxy_geometry: ProxyGeometryLayer,
    *,
    vocabulary: Sequence[str] | None = None,
) -> SemanticSceneLayer:
    """Infer semantic labels for the scene."""

    return infer_geometry_semantic_layer(
        visual_layer,
        proxy_geometry,
        vocabulary=tuple(vocabulary or ()),
    )


def infer_geometry_semantic_layer(
    visual_layer: VisualSplatLayer,
    proxy_geometry: ProxyGeometryLayer,
    vocabulary: Sequence[str] = (),
) -> SemanticSceneLayer:
    """Infer deterministic baseline semantic labels from detected scene structure."""

    scene_structure = proxy_geometry.attributes.get("scene_structure")
    if isinstance(scene_structure, SceneStructure):
        region_specs = (
            ("floor", scene_structure.floor),
            ("wall", scene_structure.walls),
            ("obstacle", scene_structure.obstacles),
            ("walkable_surface", scene_structure.walkable),
        )
        label_to_area: dict[str, float] = {}
        label_to_face_count: dict[str, int] = {}
        region_labels: list[str] = []
        for label, region in region_specs:
            if region is not None and _region_available(region):
                region_labels.append(label)
                label_to_area[label] = region.area
                label_to_face_count[label] = region.face_count

        if region_labels:
            labels = tuple(region_labels)
            return SemanticSceneLayer(
                metadata=visual_layer.metadata,
                label_count=len(labels),
                labels=labels,
                attributes={
                    "status": "geometry_semantic_layer",
                    "source": "scene_structure",
                    "proxy_method": proxy_geometry.method,
                    "configured_vocabulary": tuple(vocabulary),
                    "region_labels": labels,
                    "label_to_area": label_to_area,
                    "label_to_face_count": label_to_face_count,
                    "structure_metadata": dict(scene_structure.metadata),
                },
            )

    labels = tuple(vocabulary or ())
    return SemanticSceneLayer(
        metadata=visual_layer.metadata,
        label_count=len(labels),
        labels=labels,
        attributes={
            "proxy_method": proxy_geometry.method,
            "status": "placeholder_semantic_layer",
            "source": "configured_vocabulary",
            "configured_vocabulary": labels,
            "warning": "No scene_structure regions were available for geometry-derived semantics.",
        },
    )


def _region_available(region: SurfaceRegion | None) -> bool:
    return region is not None and region.face_count > 0 and region.area > 0.0
