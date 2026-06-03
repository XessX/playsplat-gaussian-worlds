"""Create a raw capture scene checklist for independent PlaySplat scenes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the checklist CLI parser."""

    parser = argparse.ArgumentParser(
        description="Create a Markdown checklist for an independent raw capture scene.",
    )
    parser.add_argument("--scene-id", type=str, required=True, help="Scene identifier.")
    parser.add_argument("--category", type=str, required=True, help="Scene category.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown checklist path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run checklist generation."""

    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = create_scene_checklist(
        scene_id=args.scene_id,
        category=args.category,
        output_path=args.output,
    )
    print(f"scene checklist: {output_path}")
    return 0


def create_scene_checklist(
    *,
    scene_id: str,
    category: str,
    output_path: str | Path,
) -> Path:
    """Create a Markdown capture and benchmark readiness checklist."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_scene_checklist(scene_id=scene_id, category=category),
        encoding="utf-8",
    )
    return destination


def render_scene_checklist(*, scene_id: str, category: str) -> str:
    """Render a Markdown checklist for one scene."""

    return "\n".join(
        [
            f"# Scene Checklist: {scene_id}",
            "",
            f"- Scene category: `{category}`",
            "- Capture date:",
            "- Location type:",
            "- Capture operator:",
            "",
            "## Capture Notes",
            "",
            "- Lighting notes:",
            "- Image or frame count:",
            "- Floor or walkable surface visible:",
            "- Reflective or transparent objects excessive:",
            "- Privacy or sensitive-object notes:",
            "",
            "## Readiness Checklist",
            "",
            "- [ ] Privacy check completed.",
            "- [ ] Lighting is reasonably stable.",
            "- [ ] 80-200 usable images or extracted frames are available.",
            "- [ ] Floor or another walkable surface is visible where applicable.",
            "- [ ] Reflective or transparent objects are not excessive.",
            "- [ ] Capture image validation report generated.",
            "- [ ] External 3DGS training plan generated.",
            "- [ ] Final `point_cloud.ply` exists.",
            "- [ ] Final `point_cloud.ply` staged into PlaySplat.",
            "- [ ] Scene added to `configs/scenes.local.yaml`.",
            "- [ ] Registry validation passed.",
            "- [ ] Benchmark run completed.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
