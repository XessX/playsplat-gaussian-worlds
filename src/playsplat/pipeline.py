"""High-level orchestration for the PlaySplat pipeline skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from playsplat.affordance import infer_affordance_layer
from playsplat.evaluation import PlayabilityReport, evaluate_playability
from playsplat.export import export_scene
from playsplat.gaussian import build_visual_splat_layer
from playsplat.geometry import extract_proxy_geometry
from playsplat.io import load_gaussian_scene
from playsplat.navigation import build_navigation_layer
from playsplat.physics import build_collision_layer
from playsplat.semantics import infer_semantic_layer
from playsplat.types import ExportBundle, PlaySplatScene
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
    report = evaluate_playability(scene)
    return PipelineResult(scene=scene, exports=exports, report=report)
