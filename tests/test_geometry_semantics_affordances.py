from __future__ import annotations

import numpy as np

from playsplat.affordance import infer_geometry_affordance_layer
from playsplat.semantics import infer_geometry_semantic_layer, infer_semantic_layer
from playsplat.types import (
    NavigationLayer,
    ProxyGeometryLayer,
    SceneMetadata,
    SceneStructure,
    SurfaceRegion,
    VisualSplatLayer,
)


def test_structure_derived_regions_create_geometry_semantic_layer() -> None:
    visual_layer, proxy_geometry = _structured_layers()

    semantics = infer_geometry_semantic_layer(
        visual_layer,
        proxy_geometry,
        vocabulary=("floor", "wall"),
    )

    assert semantics.attributes["status"] == "geometry_semantic_layer"
    assert semantics.attributes["source"] == "scene_structure"
    assert semantics.labels == ("floor", "wall", "obstacle", "walkable_surface")
    assert semantics.label_count == 4
    assert semantics.attributes["region_labels"] == semantics.labels
    assert semantics.attributes["label_to_area"]["floor"] == 2.0
    assert semantics.attributes["label_to_face_count"]["obstacle"] == 3


def test_public_semantic_inference_uses_geometry_when_available() -> None:
    visual_layer, proxy_geometry = _structured_layers()

    semantics = infer_semantic_layer(visual_layer, proxy_geometry, vocabulary=("floor",))

    assert semantics.attributes["status"] == "geometry_semantic_layer"
    assert "walkable_surface" in semantics.labels


def test_missing_structure_returns_placeholder_semantic_layer() -> None:
    metadata = SceneMetadata(scene_id="placeholder")
    visual_layer = VisualSplatLayer(metadata=metadata, gaussian_count=0)
    proxy_geometry = ProxyGeometryLayer(metadata=metadata, method="none")

    semantics = infer_geometry_semantic_layer(
        visual_layer,
        proxy_geometry,
        vocabulary=("floor", "wall"),
    )

    assert semantics.attributes["status"] == "placeholder_semantic_layer"
    assert semantics.labels == ("floor", "wall")
    assert "warning" in semantics.attributes


def test_geometry_semantic_layer_produces_geometry_affordances() -> None:
    visual_layer, proxy_geometry = _structured_layers()
    semantics = infer_geometry_semantic_layer(visual_layer, proxy_geometry)
    navigation = NavigationLayer(
        metadata=visual_layer.metadata,
        walkable_region_count=1,
        navmesh_polygon_count=1,
        attributes={"status": "walkable_region_detected", "walkable_area": 2.0},
    )

    affordances = infer_geometry_affordance_layer(
        semantics,
        navigation,
        affordance_labels=("walkable",),
    )

    assert affordances.attributes["status"] == "geometry_affordance_layer"
    assert affordances.attributes["source"] == "geometry_semantics_and_navigation"
    assert affordances.labels == (
        "walkable",
        "support",
        "blocking",
        "interactable_candidate",
    )
    assert affordances.affordance_count == 4
    assert "floor" in affordances.attributes["affordance_to_source_labels"]["support"]
    assert "wall" in affordances.attributes["affordance_to_source_labels"]["blocking"]
    assert "obstacle" in affordances.attributes["affordance_to_source_labels"]["blocking"]
    assert affordances.attributes["walkable_area"] == 2.0


def test_placeholder_affordance_layer_when_no_usable_data_exists() -> None:
    metadata = SceneMetadata(scene_id="placeholder")
    navigation = NavigationLayer(metadata=metadata)
    visual_layer = VisualSplatLayer(metadata=metadata, gaussian_count=0)
    semantics = infer_geometry_semantic_layer(
        visual_layer,
        ProxyGeometryLayer(metadata=metadata),
        vocabulary=("floor",),
    )

    affordances = infer_geometry_affordance_layer(
        semantics,
        navigation,
        affordance_labels=("walkable",),
    )

    assert affordances.attributes["status"] == "placeholder_affordance_layer"
    assert affordances.labels == ("walkable",)
    assert "warning" in affordances.attributes


def _structured_layers() -> tuple[VisualSplatLayer, ProxyGeometryLayer]:
    metadata = SceneMetadata(scene_id="structured")
    structure = SceneStructure(
        floor=SurfaceRegion(
            label="floor",
            face_indices=np.asarray([0, 1], dtype=np.int64),
            area=2.0,
        ),
        walls=SurfaceRegion(
            label="wall",
            face_indices=np.asarray([2], dtype=np.int64),
            area=1.0,
        ),
        obstacles=SurfaceRegion(
            label="obstacle",
            face_indices=np.asarray([3, 4, 5], dtype=np.int64),
            area=3.0,
        ),
        walkable=SurfaceRegion(
            label="walkable",
            face_indices=np.asarray([0, 1], dtype=np.int64),
            area=2.0,
        ),
        metadata={
            "total_area": 6.0,
            "floor_area": 2.0,
            "wall_area": 1.0,
            "obstacle_area": 3.0,
            "walkable_area": 2.0,
        },
    )
    visual_layer = VisualSplatLayer(metadata=metadata, gaussian_count=0)
    proxy_geometry = ProxyGeometryLayer(
        metadata=metadata,
        method="synthetic",
        attributes={"scene_structure": structure, "structure_metadata": structure.metadata},
    )
    return visual_layer, proxy_geometry
