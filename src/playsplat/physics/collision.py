"""Collision and physics layer stubs."""

from __future__ import annotations

from playsplat.types import CollisionPhysicsLayer, ProxyGeometryLayer


def build_collision_layer(
    proxy_geometry: ProxyGeometryLayer,
    *,
    collision_mode: str = "static",
) -> CollisionPhysicsLayer:
    """Build a collision and physics layer from proxy geometry."""

    return CollisionPhysicsLayer(
        metadata=proxy_geometry.metadata,
        mode=collision_mode,
        attributes={"status": "placeholder_collision_layer"},
    )
