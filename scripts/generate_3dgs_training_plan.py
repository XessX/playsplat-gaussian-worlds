"""Generate external 3DGS training instructions for a raw capture scene."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence


TRAINER_PROFILES = ("generic", "graphdeco_3dgs", "postshot_or_other_gui")


@dataclass(frozen=True)
class TrainingPlan:
    """External 3DGS training plan artifact."""

    scene_id: str
    images_dir: str
    output_dir: str
    trainer_name: str
    trainer_root: str | None
    expected_point_cloud: str
    staging_path: str
    commands: tuple[str, ...]
    manual_steps: tuple[str, ...]
    markdown_path: str
    json_path: str


def build_parser() -> argparse.ArgumentParser:
    """Build the training plan CLI parser."""

    parser = argparse.ArgumentParser(
        description="Generate reproducible external 3DGS training instructions.",
    )
    parser.add_argument("--scene-id", type=str, required=True, help="Scene identifier.")
    parser.add_argument("--images", type=Path, required=True, help="Capture image directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Raw capture scene directory where plan files will be written.",
    )
    parser.add_argument(
        "--trainer-name",
        choices=TRAINER_PROFILES,
        default="generic",
        help="Training profile to document.",
    )
    parser.add_argument(
        "--trainer-root",
        type=Path,
        default=None,
        help="Optional external trainer repository or tool root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run training plan generation."""

    parser = build_parser()
    args = parser.parse_args(argv)
    plan = generate_training_plan(
        scene_id=args.scene_id,
        images_dir=args.images,
        output_dir=args.output_dir,
        trainer_name=args.trainer_name,
        trainer_root=args.trainer_root,
    )
    print(f"training commands: {plan.markdown_path}")
    print(f"training plan: {plan.json_path}")
    print(f"expected point_cloud.ply: {plan.expected_point_cloud}")
    return 0


def generate_training_plan(
    *,
    scene_id: str,
    images_dir: str | Path,
    output_dir: str | Path,
    trainer_name: str = "generic",
    trainer_root: str | Path | None = None,
) -> TrainingPlan:
    """Generate Markdown and JSON training plan artifacts."""

    if trainer_name not in TRAINER_PROFILES:
        raise ValueError(
            f"Unsupported trainer profile {trainer_name!r}; "
            f"expected one of: {', '.join(TRAINER_PROFILES)}"
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    markdown_path = output_root / "training_commands.md"
    json_path = output_root / "training_plan.json"
    expected_point_cloud = (
        output_root / "training_output" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    )
    staging_path = Path("data") / "scenes" / scene_id / "point_cloud.ply"
    commands, manual_steps = _profile_steps(
        scene_id=scene_id,
        images_dir=Path(images_dir),
        output_root=output_root,
        trainer_name=trainer_name,
        trainer_root=Path(trainer_root) if trainer_root is not None else None,
    )
    plan = TrainingPlan(
        scene_id=scene_id,
        images_dir=Path(images_dir).as_posix(),
        output_dir=output_root.as_posix(),
        trainer_name=trainer_name,
        trainer_root=Path(trainer_root).as_posix() if trainer_root is not None else None,
        expected_point_cloud=expected_point_cloud.as_posix(),
        staging_path=staging_path.as_posix(),
        commands=commands,
        manual_steps=manual_steps,
        markdown_path=markdown_path.as_posix(),
        json_path=json_path.as_posix(),
    )
    markdown_path.write_text(render_training_commands_markdown(plan), encoding="utf-8")
    json_path.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
    return plan


def render_training_commands_markdown(plan: TrainingPlan) -> str:
    """Render a Markdown training plan."""

    command_block = "\n".join(plan.commands) if plan.commands else "# Manual GUI workflow"
    manual_steps = "\n".join(f"{index}. {step}" for index, step in enumerate(plan.manual_steps, 1))
    return "\n".join(
        [
            f"# 3DGS Training Plan: {plan.scene_id}",
            "",
            f"- Trainer profile: `{plan.trainer_name}`",
            f"- Images: `{plan.images_dir}`",
            f"- Training output root: `{plan.output_dir}`",
            f"- Expected point cloud: `{plan.expected_point_cloud}`",
            f"- PlaySplat staging path: `{plan.staging_path}`",
            "",
            "## Steps",
            "",
            manual_steps,
            "",
            "## Command Template",
            "",
            "```bash",
            command_block,
            "```",
            "",
            "## After Training",
            "",
            "Locate `point_cloud/iteration_xxxxx/point_cloud.ply`, then stage it with",
            "`scripts/stage_independent_scenes.py` before validating the scene registry.",
            "",
        ]
    )


def _profile_steps(
    *,
    scene_id: str,
    images_dir: Path,
    output_root: Path,
    trainer_name: str,
    trainer_root: Path | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    training_output = output_root / "training_output"
    common_steps = (
        "Place or extract capture images into the documented image directory.",
        "Run external photogrammetry or 3DGS preprocessing with reproducible paths.",
        "Run the external 3DGS training command.",
        "Locate the final point_cloud.ply from the highest usable iteration.",
        "Stage the selected point_cloud.ply into PlaySplat.",
    )
    if trainer_name == "generic":
        commands = (
            f'# Replace this with your selected 3DGS trainer command for "{scene_id}".',
            f'python PREPROCESS_COMMAND --images "{images_dir.as_posix()}" '
            f'--output "{training_output.as_posix()}"',
            f'python TRAIN_COMMAND --data "{training_output.as_posix()}" '
            f'--output "{training_output.as_posix()}"',
        )
        return commands, common_steps
    if trainer_name == "graphdeco_3dgs":
        root = trainer_root.as_posix() if trainer_root is not None else "PATH/TO/gaussian-splatting"
        commands = (
            f'cd "{root}"',
            f'python convert.py -s "{output_root.as_posix()}"',
            f'python train.py -s "{output_root.as_posix()}" '
            f'-m "{training_output.as_posix()}"',
        )
        return commands, common_steps

    gui_steps = (
        "Import the image folder into the GUI 3DGS tool.",
        "Run the tool's camera alignment, reconstruction, and Gaussian training workflow.",
        "Export or locate point_cloud/iteration_xxxxx/point_cloud.ply.",
        "Record the exact export path and training settings in this Markdown file.",
        "Stage the selected point_cloud.ply into PlaySplat.",
    )
    return (), gui_steps


if __name__ == "__main__":
    raise SystemExit(main())
