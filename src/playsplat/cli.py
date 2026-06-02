"""Command-line interface for PlaySplat."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from playsplat.pipeline import run_pipeline
from playsplat.utils.config import load_pipeline_settings


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run the PlaySplat pipeline skeleton.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to a YAML pipeline config.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Optional Gaussian scene input.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--scene-id", type=str, default=None, help="Optional scene identifier.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the PlaySplat CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_pipeline_settings(args.config).with_overrides(
        input_path=args.input,
        output_dir=args.output_dir,
        scene_id=args.scene_id,
    )
    result = run_pipeline(settings)

    print(f"PlaySplat scene: {result.scene.metadata.scene_id}")
    print(f"Visual layer: {result.scene.visual.gaussian_count} Gaussians")
    print(f"Proxy geometry: {result.scene.proxy_geometry.method}")
    print(f"Physics mode: {result.scene.collision_physics.mode}")
    print(f"Semantic labels: {', '.join(result.scene.semantics.labels) or 'none'}")
    print(f"Affordances: {', '.join(result.scene.affordances.labels) or 'none'}")
    print(f"Exports planned: {', '.join(bundle.target for bundle in result.exports) or 'none'}")
    print(f"Playability status: {result.report.status}")
    return 0
