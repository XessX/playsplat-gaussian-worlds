"""Visual Gaussian splat layer stubs."""

from __future__ import annotations

from playsplat.types import GaussianLayer, GaussianSplatScene, VisualSplatLayer


def build_visual_splat_layer(scene: GaussianSplatScene) -> VisualSplatLayer:
    """Build the visual splat layer from an input Gaussian scene."""

    gaussian_layer = scene.attributes.get("gaussian_layer")
    if not isinstance(gaussian_layer, GaussianLayer):
        gaussian_layer = None

    return VisualSplatLayer(
        metadata=scene.metadata,
        gaussian_count=scene.gaussian_count,
        gaussians=gaussian_layer,
        attributes={"status": "placeholder_visual_layer"},
    )
