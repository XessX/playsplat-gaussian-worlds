"""Navigation layer stubs."""

from __future__ import annotations

from playsplat.types import CollisionPhysicsLayer, NavigationLayer, ProxyGeometryLayer


def build_navigation_layer(
    proxy_geometry: ProxyGeometryLayer,
    collision_layer: CollisionPhysicsLayer,
    *,
    agent_radius: float = 0.4,
) -> NavigationLayer:
    """Build a navigation and walkability layer."""

    return NavigationLayer(
        metadata=proxy_geometry.metadata,
        agent_radius=agent_radius,
        attributes={
            "collision_mode": collision_layer.mode,
            "status": "placeholder_navigation_layer",
        },
    )
