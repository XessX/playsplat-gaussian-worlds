"""Engine export bundle generation for PlaySplat research outputs."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any

from playsplat.types import EngineExportManifest, ExportBundle, PlaySplatScene, ProxyMesh


SUPPORTED_ENGINE_TARGETS = ("unity", "playcanvas", "webgl")


def create_engine_export_bundle(
    scene: PlaySplatScene,
    output_dir: str | Path,
    target: str,
) -> ExportBundle:
    """Create a target-specific engine export bundle with manifest and README."""

    normalized_target = target.strip().lower()
    bundle_dir = Path(output_dir) / "exports" / normalized_target
    bundle_dir.mkdir(parents=True, exist_ok=True)

    asset_sources = _asset_sources(scene, Path(output_dir))
    copied_files: dict[str, str] = {}
    missing_assets: list[str] = []
    source_references: dict[str, str] = {}

    for asset_name, source_path in asset_sources.items():
        if source_path is None:
            missing_assets.append(asset_name)
            continue
        if source_path.exists() and source_path.is_file():
            destination = _copy_asset(source_path, bundle_dir, asset_name)
            copied_files[asset_name] = destination.name
        else:
            missing_assets.append(asset_name)
            source_references[asset_name] = str(source_path)

    readme_path = bundle_dir / f"README_{normalized_target}.md"
    manifest_path = bundle_dir / "manifest.json"
    manifest = EngineExportManifest(
        target=normalized_target,
        scene_id=scene.metadata.scene_id,
        files={
            **copied_files,
            "manifest": manifest_path.name,
            "readme": readme_path.name,
        },
        metadata={
            "project_name": "PlaySplat",
            "coordinate_system": scene.metadata.coordinate_system,
            "source_path": str(scene.metadata.source_path)
            if scene.metadata.source_path is not None
            else None,
            "source_references": source_references,
            "missing_assets": missing_assets,
            "layer_summary": _layer_summary(scene),
            "notes": _notes_for_bundle(missing_assets),
        },
    )

    readme_path.write_text(
        _target_readme(normalized_target, manifest),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    return ExportBundle(target=normalized_target, output_path=bundle_dir, status="created")


def _asset_sources(scene: PlaySplatScene, output_dir: Path) -> dict[str, Path | None]:
    return {
        "visual_gaussian_ply": scene.metadata.source_path,
        "proxy_mesh": output_dir / "proxy_mesh.obj",
        "collision_mesh": output_dir / "collision_mesh.obj",
        "floor_mesh": output_dir / "floor_mesh.obj",
        "wall_mesh": output_dir / "wall_mesh.obj",
        "obstacle_mesh": output_dir / "obstacle_mesh.obj",
        "walkable_mesh": output_dir / "walkable_mesh.obj",
        "gaussian_stats": output_dir / "gaussian_stats.json",
        "proxy_metadata": output_dir / "proxy_metadata.json",
        "scene_structure": output_dir / "scene_structure.json",
    }


def _copy_asset(source_path: Path, bundle_dir: Path, asset_name: str) -> Path:
    destination = bundle_dir / _bundle_filename(asset_name, source_path)
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
    return destination


def _bundle_filename(asset_name: str, source_path: Path) -> str:
    if asset_name == "visual_gaussian_ply":
        return source_path.name
    return source_path.name


def _layer_summary(scene: PlaySplatScene) -> dict[str, Any]:
    collision_mesh = scene.proxy_geometry.attributes.get("collision_mesh")
    collision_metadata = scene.proxy_geometry.attributes.get("collision_mesh_metadata", {})
    if not isinstance(collision_metadata, dict):
        collision_metadata = {}
    return {
        "visual_layer": {
            "representation": scene.visual.representation,
            "gaussian_count": scene.visual.gaussian_count,
            "has_gaussian_layer": scene.visual.gaussians is not None,
        },
        "proxy_geometry": {
            "method": scene.proxy_geometry.method,
            "mesh_count": scene.proxy_geometry.mesh_count,
            "vertex_count": scene.proxy_geometry.vertex_count,
            "face_count": scene.proxy_geometry.face_count,
            "status": scene.proxy_geometry.attributes.get("status", "unknown"),
            "collision_mesh_available": isinstance(collision_mesh, ProxyMesh),
            "collision_vertex_count": int(collision_mesh.vertices.shape[0])
            if isinstance(collision_mesh, ProxyMesh)
            else 0,
            "collision_face_count": int(collision_mesh.faces.shape[0])
            if isinstance(collision_mesh, ProxyMesh)
            else 0,
            "simplification_status": collision_metadata.get("status", "missing"),
        },
        "collision_physics": {
            "mode": scene.collision_physics.mode,
            "collider_count": scene.collision_physics.collider_count,
            "rigid_body_count": scene.collision_physics.rigid_body_count,
        },
        "navigation": {
            "walkable_region_count": scene.navigation.walkable_region_count,
            "navmesh_polygon_count": scene.navigation.navmesh_polygon_count,
            "status": scene.navigation.attributes.get("status", "unknown"),
            "walkable_area": scene.navigation.attributes.get("walkable_area"),
        },
        "semantics": {
            "label_count": scene.semantics.label_count,
            "labels": list(scene.semantics.labels),
            "status": scene.semantics.attributes.get("status", "placeholder"),
        },
        "affordances": {
            "affordance_count": scene.affordances.affordance_count,
            "labels": list(scene.affordances.labels),
            "status": scene.affordances.attributes.get("status", "placeholder"),
        },
    }


def _notes_for_bundle(missing_assets: list[str]) -> list[str]:
    notes = [
        "This bundle is a research packaging artifact, not a complete engine integration.",
        "The visual splat, collision mesh, proxy geometry, structure meshes, and navigation metadata are generated or referenced when available.",
        "Semantic labels and affordances are still placeholder research layers unless produced by future modules.",
    ]
    if missing_assets:
        notes.append(
            "Some expected assets are missing; see metadata.missing_assets in manifest.json."
        )
    return notes


def _target_readme(target: str, manifest: EngineExportManifest) -> str:
    if target == "unity":
        body = _unity_readme_body()
    elif target == "playcanvas":
        body = _playcanvas_readme_body()
    elif target == "webgl":
        body = _webgl_readme_body()
    else:
        body = _generic_readme_body(target)

    available_assets = "\n".join(
        f"- `{filename}` ({name})" for name, filename in sorted(manifest.files.items())
    )
    missing_assets = manifest.metadata.get("missing_assets", [])
    missing = "\n".join(f"- `{asset}`" for asset in missing_assets) or "- none"
    return (
        f"# PlaySplat {target} Export\n\n"
        f"Scene: `{manifest.scene_id}`\n\n"
        "## Available Files\n\n"
        f"{available_assets}\n\n"
        "## Missing Or Referenced Assets\n\n"
        f"{missing}\n\n"
        f"{body}\n"
    )


def _unity_readme_body() -> str:
    return """## Unity Import Notes

Expected assets include a visual Gaussian splat `.ply`, `collision_mesh.obj`, `proxy_mesh.obj`, floor or walkable meshes, and wall or obstacle meshes.

Suggested setup:

- Import the splat with any compatible Gaussian Splatting Unity plugin or renderer.
- Treat the splat as a non-collider visual layer.
- Use `collision_mesh.obj` as the first MeshCollider candidate for physics.
- Keep `proxy_mesh.obj` as higher-detail debug geometry or a fallback collider.
- Use `wall_mesh.obj` and `obstacle_mesh.obj` as labeled blocker/debug meshes.
- Use `walkable_mesh.obj` or `floor_mesh.obj` with NavMeshSurface baking.
- Keep visual and collision layers separate so physics does not depend on splat rendering.
"""


def _playcanvas_readme_body() -> str:
    return """## PlayCanvas Import Notes

Suggested setup:

- Use a compatible splat viewer or runtime for the visual Gaussian layer.
- Load `collision_mesh.obj` as coarse collision geometry.
- Keep `proxy_mesh.obj` for higher-detail inspection or fallback collision.
- Use `walkable_mesh.obj` to constrain player or agent movement.
- Use `wall_mesh.obj` and `obstacle_mesh.obj` as blockers.
- Read `scene_structure.json` for floor, wall, obstacle, and walkable labels.
"""


def _webgl_readme_body() -> str:
    return """## Generic WebGL Usage Notes

Suggested setup:

- Render the visual splat with a compatible WebGL Gaussian splatting renderer.
- Use `collision_mesh.obj` as collision geometry for a physics engine.
- Keep `proxy_mesh.obj` as the higher-detail proxy for inspection and debugging.
- Use `scene_structure.json` to attach layer labels to floor, wall, obstacle, and walkable surfaces.
- Treat this bundle as import metadata and assets, not a full browser runtime.
"""


def _generic_readme_body(target: str) -> str:
    return (
        "## Import Notes\n\n"
        f"`{target}` is not a first-class PlaySplat target yet. "
        "Use the manifest and available assets as generic engine import metadata.\n"
    )
