"""Semantic scene layer stubs."""

from __future__ import annotations

from collections.abc import Sequence

from playsplat.types import ProxyGeometryLayer, SemanticSceneLayer, VisualSplatLayer


def infer_semantic_layer(
    visual_layer: VisualSplatLayer,
    proxy_geometry: ProxyGeometryLayer,
    *,
    vocabulary: Sequence[str] | None = None,
) -> SemanticSceneLayer:
    """Infer semantic labels for the scene."""

    labels = tuple(vocabulary or ())
    return SemanticSceneLayer(
        metadata=visual_layer.metadata,
        label_count=len(labels),
        labels=labels,
        attributes={
            "proxy_method": proxy_geometry.method,
            "status": "placeholder_semantic_layer",
        },
    )
