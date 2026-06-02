"""Visual Gaussian splat layer stubs."""

from __future__ import annotations

from playsplat.types import GaussianSplatScene, VisualSplatLayer


def build_visual_splat_layer(scene: GaussianSplatScene) -> VisualSplatLayer:
    """Build the visual splat layer from an input Gaussian scene."""

    return VisualSplatLayer(
        metadata=scene.metadata,
        gaussian_count=scene.gaussian_count,
        attributes={"status": "placeholder_visual_layer"},
    )
