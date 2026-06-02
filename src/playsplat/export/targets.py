"""Export target dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from playsplat.types import ExportBundle, PlaySplatScene
from playsplat.export.engine_bundle import SUPPORTED_ENGINE_TARGETS, create_engine_export_bundle


def export_scene(
    scene: PlaySplatScene,
    output_dir: Path,
    targets: Sequence[str],
) -> list[ExportBundle]:
    """Create export bundles for supported engine targets."""

    bundles: list[ExportBundle] = []
    supported_targets = set(SUPPORTED_ENGINE_TARGETS)
    for target in targets:
        normalized = target.strip().lower()
        if normalized in supported_targets:
            bundles.append(create_engine_export_bundle(scene, output_dir, normalized))
        else:
            bundles.append(
                ExportBundle(
                    target=normalized,
                    output_path=output_dir / "exports" / normalized,
                    status="unsupported",
                )
            )
    return bundles
