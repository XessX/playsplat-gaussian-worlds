from __future__ import annotations

import json
from pathlib import Path

import pytest

from playsplat.export import create_engine_export_bundle
from playsplat.export.targets import export_scene
from playsplat.types import (
    AffordanceLayer,
    CollisionPhysicsLayer,
    NavigationLayer,
    PlaySplatScene,
    ProxyGeometryLayer,
    SceneMetadata,
    SemanticSceneLayer,
    VisualSplatLayer,
)


@pytest.mark.parametrize(
    ("target", "readme_name"),
    [
        ("unity", "README_unity.md"),
        ("playcanvas", "README_playcanvas.md"),
        ("webgl", "README_webgl.md"),
    ],
)
def test_engine_export_bundle_creates_target_directory_and_manifest(
    tmp_path: Path,
    target: str,
    readme_name: str,
) -> None:
    output_dir = tmp_path / "outputs"
    _write_fake_assets(output_dir)
    scene = _fake_scene(source_path=output_dir / "source_scene.ply")

    bundle = create_engine_export_bundle(scene, output_dir, target)

    assert bundle.target == target
    assert bundle.status == "created"
    assert bundle.output_path == output_dir / "exports" / target
    assert (bundle.output_path / "manifest.json").exists()
    assert (bundle.output_path / readme_name).exists()

    manifest = json.loads((bundle.output_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["target"] == target
    assert manifest["scene_id"] == "bundle_scene"
    assert manifest["metadata"]["project_name"] == "PlaySplat"
    assert "layer_summary" in manifest["metadata"]
    assert "proxy_mesh" in manifest["files"]
    assert "collision_mesh" in manifest["files"]
    assert "collision_mesh.obj" in (
        bundle.output_path / readme_name
    ).read_text(encoding="utf-8")
    layer_summary = manifest["metadata"]["layer_summary"]
    assert layer_summary["semantics"]["status"] == "geometry_semantic_layer"
    assert layer_summary["semantics"]["is_geometry_derived"] is True
    assert layer_summary["semantics"]["is_placeholder"] is False
    assert layer_summary["affordances"]["status"] == "geometry_affordance_layer"
    assert layer_summary["affordances"]["is_geometry_derived"] is True
    assert layer_summary["affordances"]["is_placeholder"] is False


def test_engine_export_bundle_missing_assets_do_not_crash(tmp_path: Path) -> None:
    scene = _fake_scene(source_path=None)

    bundle = create_engine_export_bundle(scene, tmp_path / "outputs", "unity")

    manifest = json.loads((bundle.output_path / "manifest.json").read_text(encoding="utf-8"))
    assert bundle.status == "created"
    assert manifest["metadata"]["missing_assets"]
    assert "visual_gaussian_ply" in manifest["metadata"]["missing_assets"]


def test_export_scene_returns_bundles_for_configured_targets(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    _write_fake_assets(output_dir)
    scene = _fake_scene(source_path=output_dir / "source_scene.ply")

    bundles = export_scene(scene, output_dir, ("unity", "playcanvas", "webgl"))

    assert [bundle.target for bundle in bundles] == ["unity", "playcanvas", "webgl"]
    for bundle in bundles:
        assert bundle.status == "created"
        assert (bundle.output_path / "manifest.json").exists()
        assert (bundle.output_path / f"README_{bundle.target}.md").exists()


def _write_fake_assets(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "source_scene.ply",
        "proxy_mesh.obj",
        "collision_mesh.obj",
        "floor_mesh.obj",
        "wall_mesh.obj",
        "obstacle_mesh.obj",
        "walkable_mesh.obj",
        "gaussian_stats.json",
        "proxy_metadata.json",
        "scene_structure.json",
    ):
        path = output_dir / filename
        if path.suffix == ".json":
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text("# fake asset\n", encoding="utf-8")


def _fake_scene(source_path: Path | None) -> PlaySplatScene:
    metadata = SceneMetadata(scene_id="bundle_scene", source_path=source_path)
    return PlaySplatScene(
        metadata=metadata,
        visual=VisualSplatLayer(metadata=metadata, gaussian_count=4),
        proxy_geometry=ProxyGeometryLayer(
            metadata=metadata,
            mesh_count=1,
            vertex_count=8,
            face_count=12,
            method="test_proxy",
            attributes={"status": "proxy_mesh_extracted"},
        ),
        collision_physics=CollisionPhysicsLayer(metadata=metadata, collider_count=1),
        navigation=NavigationLayer(
            metadata=metadata,
            walkable_region_count=1,
            navmesh_polygon_count=2,
            attributes={"status": "walkable_region_detected", "walkable_area": 1.0},
        ),
        semantics=SemanticSceneLayer(
            metadata=metadata,
            label_count=1,
            labels=("floor",),
            attributes={"status": "geometry_semantic_layer"},
        ),
        affordances=AffordanceLayer(
            metadata=metadata,
            affordance_count=1,
            labels=("walkable",),
            attributes={"status": "geometry_affordance_layer"},
        ),
    )
