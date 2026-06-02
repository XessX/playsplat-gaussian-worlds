"""Playability evaluation stubs."""

from __future__ import annotations

from dataclasses import dataclass

from playsplat.types import PlaySplatScene


@dataclass(frozen=True)
class PlayabilityReport:
    """High-level playability readiness summary."""

    status: str
    collision_ready: bool
    navigation_ready: bool
    semantics_ready: bool
    affordances_ready: bool
    notes: tuple[str, ...] = ()


def evaluate_playability(scene: PlaySplatScene) -> PlayabilityReport:
    """Evaluate whether a layered scene is ready for interaction."""

    return PlayabilityReport(
        status="placeholder",
        collision_ready=scene.collision_physics.collider_count > 0,
        navigation_ready=scene.navigation.navmesh_polygon_count > 0,
        semantics_ready=scene.semantics.label_count > 0,
        affordances_ready=scene.affordances.affordance_count > 0,
        notes=(
            "Collision and navigation algorithms are not implemented yet.",
            "Semantic and affordance labels are config-level placeholders.",
        ),
    )
