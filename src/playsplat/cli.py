"""Command-line interface for PlaySplat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from playsplat.gaussian import compute_gaussian_stats
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

    if result.scene.visual.gaussians is not None:
        stats = compute_gaussian_stats(result.scene.visual.gaussians)
        _print_stats_table(stats)
        stats_path = settings.output_dir / "gaussian_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
            handle.write("\n")
        print(f"Gaussian stats saved: {stats_path}")

    return 0


def _print_stats_table(stats: dict[str, Any]) -> None:
    rows = [
        ("Gaussians", str(stats["num_gaussians"])),
        ("BBox min", _format_value(stats["bounding_box"]["min"])),
        ("BBox max", _format_value(stats["bounding_box"]["max"])),
        ("Scene center", _format_value(stats["scene_center"])),
        ("Scene size", _format_value(stats["scene_size"])),
        ("Opacity", _format_summary(stats["opacity"])),
        ("Scales", _format_summary(stats["scales"])),
        ("Color fields", str(stats["color_field_count"])),
        ("Memory MB", f"{stats['estimated_memory_footprint']['megabytes']:.6f}"),
    ]
    width = max(len(label) for label, _ in rows)
    print("")
    print("Gaussian statistics")
    print("-" * (width + 3 + 32))
    for label, value in rows:
        print(f"{label:<{width}} | {value}")


def _format_summary(summary: Any) -> str:
    if summary is None:
        return "n/a"
    return (
        f"min={_format_value(summary['min'])}; "
        f"max={_format_value(summary['max'])}; "
        f"mean={_format_value(summary['mean'])}"
    )


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(f"{float(item):.6g}" for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
