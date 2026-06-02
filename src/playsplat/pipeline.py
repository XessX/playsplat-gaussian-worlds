"""High-level orchestration for the PlaySplat pipeline skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from playsplat.affordance import infer_affordance_layer
from playsplat.evaluation import PlayabilityReport, evaluate_playability
from playsplat.export import export_scene
from playsplat.gaussian import build_visual_splat_layer, filter_gaussians_for_geometry
from playsplat.geometry import (
    build_voxel_occupancy,
    classify_proxy_mesh_structure,
    extract_proxy_geometry,
    extract_proxy_mesh,
)
from playsplat.io import load_gaussian_scene
from playsplat.navigation import build_navigation_layer
from playsplat.physics import build_collision_layer
from playsplat.semantics import infer_semantic_layer
from playsplat.types import ExportBundle, PlaySplatScene, ProxyGeometryLayer
from playsplat.utils.config import PipelineSettings


@dataclass(frozen=True)
class PipelineResult:
    """Result returned by a pipeline run."""

    scene: PlaySplatScene
    exports: tuple[ExportBundle, ...]
    report: PlayabilityReport


def run_pipeline(settings: PipelineSettings) -> PipelineResult:
    """Run the current placeholder PlaySplat pipeline.

    The function wires together the intended research stages while returning
    typed placeholder layers. Future milestones should replace each stage with
    concrete algorithms without changing this high-level contract.
    """

    source_scene = load_gaussian_scene(settings.input_path, settings.scene_id)
    visual_layer = build_visual_splat_layer(source_scene)
    proxy_geometry = extract_proxy_geometry(visual_layer, method=settings.proxy_method)
    if visual_layer.gaussians is not None and settings.proxy_enabled:
        filtered = filter_gaussians_for_geometry(
            visual_layer.gaussians,
            opacity_threshold=settings.opacity_threshold,
            bounds_quantile=settings.bounds_quantile,
            max_gaussians=settings.max_gaussians,
        )
        occupancy_grid = build_voxel_occupancy(
            filtered,
            voxel_size=settings.voxel_size,
            density_threshold=settings.density_threshold,
            padding_voxels=settings.padding_voxels,
            max_grid_voxels=settings.max_grid_voxels,
        )
        proxy_mesh = extract_proxy_mesh(occupancy_grid, smooth_sigma=settings.smooth_sigma)
        scene_structure = None
        structure_metadata = None
        if settings.structure_enabled:
            scene_structure = classify_proxy_mesh_structure(
                proxy_mesh,
                up_axis=settings.structure_up_axis,
                max_floor_slope_degrees=settings.max_floor_slope_degrees,
                floor_height_quantile=settings.floor_height_quantile,
                floor_height_tolerance=settings.floor_height_tolerance,
                wall_normal_tolerance=settings.wall_normal_tolerance,
                min_region_area=settings.min_region_area,
            )
            structure_metadata = scene_structure.metadata
        proxy_metadata = {
            "filter": filtered.filter_metadata,
            "occupancy": occupancy_grid.metadata,
            "mesh": proxy_mesh.metadata,
        }
        if structure_metadata is not None:
            proxy_metadata["structure"] = structure_metadata
        proxy_geometry = ProxyGeometryLayer(
            metadata=visual_layer.metadata,
            mesh_count=1,
            vertex_count=int(proxy_mesh.vertices.shape[0]),
            face_count=int(proxy_mesh.faces.shape[0]),
            method=proxy_mesh.metadata["method"],
            attributes={
                "status": "proxy_mesh_extracted",
                "filtered_gaussians": filtered,
                "occupancy_grid": occupancy_grid,
                "proxy_mesh": proxy_mesh,
                "proxy_metadata": proxy_metadata,
                "scene_structure": scene_structure,
                "structure_metadata": structure_metadata,
            },
        )
    collision_physics = build_collision_layer(
        proxy_geometry,
        collision_mode=settings.collision_mode,
    )
    navigation = build_navigation_layer(
        proxy_geometry,
        collision_physics,
        agent_radius=settings.agent_radius,
    )
    semantics = infer_semantic_layer(
        visual_layer,
        proxy_geometry,
        vocabulary=settings.semantic_vocabulary,
    )
    affordances = infer_affordance_layer(
        semantics,
        navigation,
        affordance_labels=settings.affordance_labels,
    )

    scene = PlaySplatScene(
        metadata=source_scene.metadata,
        visual=visual_layer,
        proxy_geometry=proxy_geometry,
        collision_physics=collision_physics,
        navigation=navigation,
        semantics=semantics,
        affordances=affordances,
    )
    exports = tuple(export_scene(scene, settings.output_dir, settings.export_targets))
    scene.proxy_geometry.attributes["engine_exports"] = exports
    report = evaluate_playability(scene)
    return PipelineResult(scene=scene, exports=exports, report=report)
