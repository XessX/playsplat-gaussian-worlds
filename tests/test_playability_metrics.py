from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from playsplat.evaluation import (
    compute_playability_metrics,
    evaluate_playability,
    write_playability_metrics_csv,
    write_playability_report,
)
from playsplat.types import (
    AffordanceLayer,
    CollisionPhysicsLayer,
    ExportBundle,
    GaussianLayer,
    NavigationLayer,
    PlaySplatScene,
    ProxyGeometryLayer,
    ProxyMesh,
    SceneMetadata,
    SceneStructure,
    SemanticSceneLayer,
    SurfaceRegion,
    VisualSplatLayer,
)


def test_metrics_computed_from_placeholder_scene() -> None:
    scene = _placeholder_scene()

    metrics = compute_playability_metrics(scene)
    report = evaluate_playability(scene)

    assert metrics["gaussian_count"] == 0
    assert metrics["has_visual_gaussian_layer"] is False
    assert metrics["proxy_mesh_available"] is False
    assert metrics["collision_mesh_available"] is False
    assert metrics["semantic_status"] == "placeholder_semantic_layer"
    assert metrics["affordance_status"] == "placeholder_affordance_layer"
    assert report.semantics_ready is False
    assert report.affordances_ready is False
    assert metrics["overall_playability_score"] < 0.5
    assert report.status == "placeholder"


def test_metrics_computed_from_synthetic_structured_scene() -> None:
    scene = _structured_scene()

    metrics = compute_playability_metrics(scene)
    report = evaluate_playability(scene)

    assert metrics["gaussian_count"] == 10
    assert metrics["proxy_mesh_available"] is True
    assert metrics["collision_mesh_available"] is True
    assert metrics["collision_face_count"] == 1
    assert metrics["collision_face_reduction_ratio"] == 0.5
    assert metrics["simplification_status"] == "target_reached"
    assert metrics["semantic_status"] == "geometry_semantic_layer"
    assert metrics["affordance_status"] == "geometry_affordance_layer"
    assert metrics["geometry_semantics_available"] is True
    assert metrics["geometry_affordances_available"] is True
    assert metrics["semantic_label_count"] == 4
    assert metrics["affordance_label_count"] == 4
    assert metrics["floor_area"] == 2.0
    assert metrics["walkable_area_ratio"] == 0.5
    assert metrics["engine_export_target_count"] == 2
    assert report.navigation_ready is True
    assert report.semantics_ready is True
    assert report.affordances_ready is True
    assert report.summary["overall_playability_score"] >= 0.75


def test_placeholder_semantics_and_affordances_do_not_count_as_ready() -> None:
    scene = _structured_scene_with_placeholder_meaning()

    metrics = compute_playability_metrics(scene)
    report = evaluate_playability(scene)

    assert metrics["semantic_label_count"] == 2
    assert metrics["affordance_label_count"] == 2
    assert metrics["semantic_status"] == "placeholder_semantic_layer"
    assert metrics["affordance_status"] == "placeholder_affordance_layer"
    assert report.semantics_ready is False
    assert report.affordances_ready is False
    assert report.summary["overall_playability_score"] < 1.0
    assert "semantics are placeholder" in report.warnings
    assert "affordances are placeholder" in report.warnings


def test_playability_score_improves_with_geometry_derived_layers() -> None:
    placeholder_report = evaluate_playability(_structured_scene_with_placeholder_meaning())
    geometry_report = evaluate_playability(_structured_scene())

    assert geometry_report.summary["overall_playability_score"] > placeholder_report.summary[
        "overall_playability_score"
    ]


def test_warnings_appear_when_key_layers_are_missing() -> None:
    report = evaluate_playability(_placeholder_scene())

    assert "missing visual Gaussian layer" in report.warnings
    assert "missing proxy mesh" in report.warnings
    assert "missing walkable region" in report.warnings
    assert "missing structure detection" in report.warnings


def test_write_playability_report_creates_valid_json(tmp_path: Path) -> None:
    report = evaluate_playability(_structured_scene())
    output_path = tmp_path / "playability_report.json"

    written = write_playability_report(report, output_path)

    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["summary"]["status"] == report.status
    assert data["metrics"]["proxy_face_count"] == 2
    assert data["metrics"]["semantic_status"] == "geometry_semantic_layer"
    assert data["metrics"]["affordance_status"] == "geometry_affordance_layer"
    assert data["summary"]["semantics_ready"] is True
    assert data["summary"]["affordances_ready"] is True
    assert isinstance(data["warnings"], list)


def test_write_playability_metrics_csv_creates_key_value_file(tmp_path: Path) -> None:
    report = evaluate_playability(_structured_scene())
    output_path = tmp_path / "playability_metrics.csv"

    written = write_playability_metrics_csv(report, output_path)

    with written.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["metric", "value"]
    assert ["gaussian_count", "10"] in rows
    assert any(row[0] == "summary.overall_playability_score" for row in rows)


def _placeholder_scene() -> PlaySplatScene:
    metadata = SceneMetadata(scene_id="placeholder")
    return PlaySplatScene(
        metadata=metadata,
        visual=VisualSplatLayer(metadata=metadata, gaussian_count=0),
        proxy_geometry=ProxyGeometryLayer(metadata=metadata),
        collision_physics=CollisionPhysicsLayer(metadata=metadata),
        navigation=NavigationLayer(metadata=metadata),
        semantics=SemanticSceneLayer(
            metadata=metadata,
            label_count=0,
            attributes={"status": "placeholder_semantic_layer"},
        ),
        affordances=AffordanceLayer(
            metadata=metadata,
            affordance_count=0,
            attributes={"status": "placeholder_affordance_layer"},
        ),
    )


def _structured_scene() -> PlaySplatScene:
    metadata = SceneMetadata(scene_id="structured", source_path=Path("scene.ply"))
    gaussian_layer = GaussianLayer(
        positions=np.zeros((10, 3), dtype=np.float32),
    )
    structure = SceneStructure(
        floor=SurfaceRegion(
            label="floor",
            face_indices=np.asarray([0], dtype=np.int64),
            area=2.0,
        ),
        walls=SurfaceRegion(
            label="wall",
            face_indices=np.asarray([1], dtype=np.int64),
            area=1.0,
        ),
        obstacles=SurfaceRegion(
            label="obstacle",
            face_indices=np.asarray([2], dtype=np.int64),
            area=1.0,
        ),
        walkable=SurfaceRegion(
            label="walkable",
            face_indices=np.asarray([0], dtype=np.int64),
            area=2.0,
        ),
        metadata={
            "total_area": 4.0,
            "floor_area": 2.0,
            "wall_area": 1.0,
            "obstacle_area": 1.0,
            "walkable_area": 2.0,
            "floor_face_count": 1,
            "wall_face_count": 1,
            "obstacle_face_count": 1,
            "walkable_face_count": 1,
        },
    )
    proxy_mesh = ProxyMesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.int32),
    )
    collision_mesh = ProxyMesh(
        vertices=proxy_mesh.vertices.copy(),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        metadata={
            "original_vertex_count": 3,
            "original_face_count": 2,
            "simplified_vertex_count": 3,
            "simplified_face_count": 1,
            "target_face_count": 1,
            "method": "vertex_clustering",
            "clustering_voxel_size": 0.5,
            "achieved_reduction_ratio": 0.5,
            "iterations": 1,
            "status": "target_reached",
        },
    )
    proxy_geometry = ProxyGeometryLayer(
        metadata=metadata,
        mesh_count=1,
        vertex_count=3,
        face_count=2,
        method="synthetic",
        attributes={
            "status": "proxy_mesh_extracted",
            "proxy_mesh": proxy_mesh,
            "collision_mesh": collision_mesh,
            "collision_mesh_metadata": collision_mesh.metadata,
            "scene_structure": structure,
            "structure_metadata": structure.metadata,
            "engine_exports": (
                ExportBundle(target="unity", output_path=Path("exports/unity"), status="created"),
                ExportBundle(target="webgl", output_path=Path("exports/webgl"), status="created"),
            ),
        },
    )
    return PlaySplatScene(
        metadata=metadata,
        visual=VisualSplatLayer(
            metadata=metadata,
            gaussian_count=10,
            gaussians=gaussian_layer,
        ),
        proxy_geometry=proxy_geometry,
        collision_physics=CollisionPhysicsLayer(
            metadata=metadata,
            collider_count=1,
            attributes={"status": "proxy_collision_ready"},
        ),
        navigation=NavigationLayer(
            metadata=metadata,
            walkable_region_count=1,
            navmesh_polygon_count=1,
            attributes={"status": "walkable_region_detected", "walkable_area": 2.0},
        ),
        semantics=SemanticSceneLayer(
            metadata=metadata,
            label_count=4,
            labels=("floor", "wall", "obstacle", "walkable_surface"),
            attributes={
                "status": "geometry_semantic_layer",
                "source": "scene_structure",
            },
        ),
        affordances=AffordanceLayer(
            metadata=metadata,
            affordance_count=4,
            labels=("walkable", "support", "blocking", "interactable_candidate"),
            attributes={
                "status": "geometry_affordance_layer",
                "source": "geometry_semantics_and_navigation",
            },
        ),
    )


def _structured_scene_with_placeholder_meaning() -> PlaySplatScene:
    scene = _structured_scene()
    scene.semantics = SemanticSceneLayer(
        metadata=scene.metadata,
        label_count=2,
        labels=("floor", "wall"),
        attributes={"status": "placeholder_semantic_layer"},
    )
    scene.affordances = AffordanceLayer(
        metadata=scene.metadata,
        affordance_count=2,
        labels=("walkable", "blocking"),
        attributes={"status": "placeholder_affordance_layer"},
    )
    return scene
