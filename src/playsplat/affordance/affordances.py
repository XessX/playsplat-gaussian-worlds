"""Geometry-derived affordance inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from playsplat.types import AffordanceLayer, NavigationLayer, SemanticSceneLayer


def infer_affordance_layer(
    semantic_layer: SemanticSceneLayer,
    navigation_layer: NavigationLayer,
    *,
    affordance_labels: Sequence[str] | None = None,
) -> AffordanceLayer:
    """Infer interaction affordances from scene semantics and navigation."""

    return infer_geometry_affordance_layer(
        semantic_layer,
        navigation_layer,
        affordance_labels=tuple(affordance_labels or ()),
    )


def infer_geometry_affordance_layer(
    semantics: SemanticSceneLayer,
    navigation: NavigationLayer,
    affordance_labels: Sequence[str] = (),
) -> AffordanceLayer:
    """Infer deterministic baseline affordances from geometry semantics and navigation."""

    semantic_status = str(semantics.attributes.get("status", "unknown"))
    semantic_labels = set(semantics.labels)
    walkable_area = _float_or_none(navigation.attributes.get("walkable_area"))
    affordance_to_source_labels: dict[str, list[str]] = {}

    if semantic_status == "geometry_semantic_layer":
        if "floor" in semantic_labels:
            _add_source(affordance_to_source_labels, "walkable", "floor")
            _add_source(affordance_to_source_labels, "support", "floor")
        if "walkable_surface" in semantic_labels:
            _add_source(affordance_to_source_labels, "walkable", "walkable_surface")
            _add_source(affordance_to_source_labels, "support", "walkable_surface")
        if "wall" in semantic_labels:
            _add_source(affordance_to_source_labels, "blocking", "wall")
        if "obstacle" in semantic_labels:
            _add_source(affordance_to_source_labels, "blocking", "obstacle")
            _add_source(affordance_to_source_labels, "interactable_candidate", "obstacle")

    if walkable_area is not None and walkable_area > 0.0:
        _add_source(affordance_to_source_labels, "walkable", "navigation_walkable_area")

    if affordance_to_source_labels:
        labels = tuple(affordance_to_source_labels)
        return AffordanceLayer(
            metadata=semantics.metadata,
            affordance_count=len(labels),
            labels=labels,
            attributes={
                "status": "geometry_affordance_layer",
                "source": "geometry_semantics_and_navigation",
                "semantic_status": semantic_status,
                "semantic_labels": tuple(semantics.labels),
                "configured_affordance_labels": tuple(affordance_labels),
                "affordance_to_source_labels": affordance_to_source_labels,
                "walkable_area": walkable_area,
                "walkable_region_count": navigation.walkable_region_count,
                "navigation_status": navigation.attributes.get("status", "unknown"),
            },
        )

    labels = tuple(affordance_labels or ())
    return AffordanceLayer(
        metadata=semantics.metadata,
        affordance_count=len(labels),
        labels=labels,
        attributes={
            "semantic_label_count": semantics.label_count,
            "semantic_status": semantic_status,
            "walkable_region_count": navigation.walkable_region_count,
            "navigation_status": navigation.attributes.get("status", "unknown"),
            "status": "placeholder_affordance_layer",
            "source": "configured_affordance_labels",
            "configured_affordance_labels": labels,
            "warning": "No usable geometry semantic or navigation data was available.",
        },
    )


def _add_source(
    affordance_to_source_labels: dict[str, list[str]],
    affordance: str,
    source_label: str,
) -> None:
    sources = affordance_to_source_labels.setdefault(affordance, [])
    if source_label not in sources:
        sources.append(source_label)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
