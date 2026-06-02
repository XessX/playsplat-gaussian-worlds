"""Generate debug preview images for a PlaySplat scene output directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from playsplat.visualization import generate_scene_previews  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the preview generation CLI parser."""

    parser = argparse.ArgumentParser(description="Generate PlaySplat debug preview PNGs.")
    parser.add_argument(
        "--scene-output",
        type=Path,
        required=True,
        help="Path to one scene output directory, e.g. outputs/experiments/scene1.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run preview generation."""

    parser = build_parser()
    args = parser.parse_args(argv)
    previews = generate_scene_previews(args.scene_output)
    if previews:
        print(f"Generated {len(previews)} preview(s):")
        for name, path in sorted(previews.items()):
            print(f"  {name}: {path}")
    else:
        print(f"No previewable files found in {args.scene_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
