"""Scene loading stubs."""

from __future__ import annotations

from pathlib import Path

from playsplat.types import GaussianSplatScene, SceneMetadata


def load_gaussian_scene(input_path: Path | None, scene_id: str) -> GaussianSplatScene:
    """Load a Gaussian splatting scene.

    This placeholder returns an empty scene object. Future implementations can
    support formats such as PLY-based 3DGS, compressed splat files, or renderer-
    specific scene bundles.
    """

    metadata = SceneMetadata(
        scene_id=scene_id,
        source_path=input_path,
        notes="Placeholder Gaussian scene. No file has been parsed yet.",
    )
    return GaussianSplatScene(metadata=metadata)
