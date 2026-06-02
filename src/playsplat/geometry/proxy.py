"""Proxy geometry extraction stubs."""

from __future__ import annotations

from playsplat.types import ProxyGeometryLayer, VisualSplatLayer


def extract_proxy_geometry(
    visual_layer: VisualSplatLayer,
    *,
    method: str = "placeholder_density_surface",
) -> ProxyGeometryLayer:
    """Extract proxy geometry from a visual splat layer."""

    return ProxyGeometryLayer(
        metadata=visual_layer.metadata,
        method=method,
        attributes={"status": "placeholder_proxy_geometry"},
    )
