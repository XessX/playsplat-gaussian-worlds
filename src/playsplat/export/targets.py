"""Export target stubs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from playsplat.types import ExportBundle, PlaySplatScene


def export_scene(
    scene: PlaySplatScene,
    output_dir: Path,
    targets: Sequence[str],
) -> list[ExportBundle]:
    """Plan exports for engine targets.

    The current version does not write files. It returns typed export bundle
    descriptions that future adapters can use to generate artifacts.
    """

    return [
        ExportBundle(
            target=target,
            output_path=output_dir / scene.metadata.scene_id / target,
            status="planned",
        )
        for target in targets
    ]
