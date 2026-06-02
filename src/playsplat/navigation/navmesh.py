"""Navigation layer stubs."""

from __future__ import annotations

from playsplat.types import CollisionPhysicsLayer, NavigationLayer, ProxyGeometryLayer, SceneStructure


def build_navigation_layer(
    proxy_geometry: ProxyGeometryLayer,
    collision_layer: CollisionPhysicsLayer,
    *,
    agent_radius: float = 0.4,
) -> NavigationLayer:
    """Build a navigation and walkability layer."""

    scene_structure = proxy_geometry.attributes.get("scene_structure")
    if isinstance(scene_structure, SceneStructure) and scene_structure.walkable is not None:
        walkable = scene_structure.walkable
        return NavigationLayer(
            metadata=proxy_geometry.metadata,
            walkable_region_count=1,
            navmesh_polygon_count=walkable.face_count,
            agent_radius=agent_radius,
            attributes={
                "collision_mode": collision_layer.mode,
                "walkable_area": walkable.area,
                "status": "walkable_region_detected",
            },
        )

    return NavigationLayer(
        metadata=proxy_geometry.metadata,
        agent_radius=agent_radius,
        attributes={
            "collision_mode": collision_layer.mode,
            "status": "placeholder_navigation_layer",
        },
    )
