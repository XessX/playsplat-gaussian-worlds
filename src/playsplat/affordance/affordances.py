"""Affordance inference stubs."""

from __future__ import annotations

from collections.abc import Sequence

from playsplat.types import AffordanceLayer, NavigationLayer, SemanticSceneLayer


def infer_affordance_layer(
    semantic_layer: SemanticSceneLayer,
    navigation_layer: NavigationLayer,
    *,
    affordance_labels: Sequence[str] | None = None,
) -> AffordanceLayer:
    """Infer interaction affordances from scene semantics and navigation."""

    labels = tuple(affordance_labels or ())
    return AffordanceLayer(
        metadata=semantic_layer.metadata,
        affordance_count=len(labels),
        labels=labels,
        attributes={
            "semantic_label_count": semantic_layer.label_count,
            "walkable_region_count": navigation_layer.walkable_region_count,
            "status": "placeholder_affordance_layer",
        },
    )
