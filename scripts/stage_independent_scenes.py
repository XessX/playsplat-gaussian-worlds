"""Stage independent Gaussian Splat scenes and update a local scene registry."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class SceneSpec:
    """A requested independent scene staging operation."""

    scene_id: str
    input_path: Path
    category: str
    source: str
    split: str
    notes: str


@dataclass(frozen=True)
class StagingResult:
    """Result for one staged scene."""

    scene_id: str
    category: str
    source: str
    split: str
    input_path: Path
    staged_path: Path
    status: str
    message: str = ""


def build_parser() -> argparse.ArgumentParser:
    """Build the independent scene staging CLI parser."""

    parser = argparse.ArgumentParser(
        description="Stage independent Gaussian Splat .ply scenes for PlaySplat benchmarks.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        required=True,
        help=(
            "Scene spec such as "
            'scene_id=room01,input="D:/path/point_cloud.ply",category=indoor_room,'
            'notes="Independent indoor room scene"'
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/scenes.local.yaml"),
        help="Local scene registry to update.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/scenes"),
        help="Root directory where scenes are staged.",
    )
    parser.add_argument("--split", type=str, default="benchmark", help="Default scene split.")
    parser.add_argument("--source", type=str, default="independent", help="Default scene source.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print planned actions without copying/linking or updating registry.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing registry entry and staged file for a scene id.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--copy", action="store_true", help="Copy scene files when staging.")
    mode_group.add_argument("--link", action="store_true", help="Hard-link scene files when possible.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scene staging CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    mode = "link" if args.link else "copy"
    specs = [
        parse_scene_spec(scene_spec, default_source=args.source, default_split=args.split)
        for scene_spec in args.scene
    ]
    results = stage_scenes(
        specs,
        registry_path=args.registry,
        data_root=args.data_root,
        mode=mode,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print_summary_table(results)
    return 0 if all(result.status in {"staged", "dry_run"} for result in results) else 1


def parse_scene_spec(
    spec: str,
    *,
    default_source: str = "independent",
    default_split: str = "benchmark",
) -> SceneSpec:
    """Parse one comma-separated scene specification."""

    parsed: dict[str, str] = {}
    for token in next(csv.reader([spec], skipinitialspace=True)):
        if "=" not in token:
            raise ValueError(f"Scene spec token must be key=value; got {token!r}.")
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip().strip('"').strip("'")

    required = ("scene_id", "input", "category")
    missing = [key for key in required if not parsed.get(key)]
    if missing:
        raise ValueError("Scene spec is missing required keys: " + ", ".join(missing))
    return SceneSpec(
        scene_id=parsed["scene_id"],
        input_path=Path(parsed["input"]),
        category=parsed["category"],
        source=parsed.get("source", default_source),
        split=parsed.get("split", default_split),
        notes=parsed.get("notes", ""),
    )


def stage_scenes(
    specs: Sequence[SceneSpec],
    *,
    registry_path: str | Path,
    data_root: str | Path,
    mode: str = "copy",
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[StagingResult]:
    """Stage scene files and update the local registry."""

    registry = load_registry_mapping(registry_path)
    scenes = registry.setdefault("scenes", [])
    if not isinstance(scenes, list):
        raise ValueError("Registry 'scenes' field must be a list.")
    existing_indices = _scene_indices(scenes)
    data_root_path = Path(data_root)
    results: list[StagingResult] = []
    updated_scenes = list(scenes)

    for spec in specs:
        staged_path = data_root_path / spec.scene_id / "point_cloud.ply"
        if spec.scene_id in existing_indices and not overwrite:
            results.append(
                _result(
                    spec,
                    staged_path,
                    status="duplicate_scene_id",
                    message="Scene id already exists; use --overwrite to replace it.",
                )
            )
            continue
        if not spec.input_path.exists():
            results.append(
                _result(
                    spec,
                    staged_path,
                    status="missing_input",
                    message=f"Input file does not exist: {spec.input_path}",
                )
            )
            continue

        if dry_run:
            results.append(_result(spec, staged_path, status="dry_run"))
            continue

        staged_path.parent.mkdir(parents=True, exist_ok=True)
        if staged_path.exists() and not overwrite:
            results.append(
                _result(
                    spec,
                    staged_path,
                    status="staged_file_exists",
                    message="Staged file already exists; use --overwrite to replace it.",
                )
            )
            continue
        if staged_path.exists() and overwrite:
            staged_path.unlink()
        message = _stage_file(spec.input_path, staged_path, mode=mode)
        record = scene_spec_to_registry_record(spec, staged_path)
        if spec.scene_id in existing_indices:
            updated_scenes[existing_indices[spec.scene_id]] = record
        else:
            existing_indices[spec.scene_id] = len(updated_scenes)
            updated_scenes.append(record)
        results.append(_result(spec, staged_path, status="staged", message=message))

    if not dry_run and any(result.status == "staged" for result in results):
        registry["scenes"] = updated_scenes
        write_registry_mapping(registry, registry_path)
    return results


def scene_spec_to_registry_record(spec: SceneSpec, staged_path: Path) -> dict[str, str]:
    """Convert a staged scene spec into a registry record."""

    return {
        "scene_id": spec.scene_id,
        "input_path": staged_path.as_posix(),
        "category": spec.category,
        "source": spec.source,
        "split": spec.split,
        "notes": spec.notes,
    }


def load_registry_mapping(path: str | Path) -> dict[str, Any]:
    """Load a registry YAML mapping, or create an empty registry."""

    registry_path = Path(path)
    if not registry_path.exists():
        return {"scenes": []}
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected registry YAML mapping: {registry_path}")
    data.setdefault("scenes", [])
    return data


def write_registry_mapping(registry: dict[str, Any], path: str | Path) -> Path:
    """Write a registry YAML mapping."""

    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )
    return registry_path


def print_summary_table(results: Sequence[StagingResult]) -> None:
    """Print a concise staging summary table."""

    headers = ("scene_id", "category", "source", "split", "input path", "staged path", "status")
    rows = [
        (
            result.scene_id,
            result.category,
            result.source,
            result.split,
            str(result.input_path),
            str(result.staged_path),
            result.status if not result.message else f"{result.status}: {result.message}",
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _stage_file(input_path: Path, staged_path: Path, *, mode: str) -> str:
    if mode == "link":
        try:
            os.link(input_path, staged_path)
            return "hard linked"
        except OSError as exc:
            shutil.copy2(input_path, staged_path)
            return f"link failed ({exc}); copied instead"
    shutil.copy2(input_path, staged_path)
    return "copied"


def _scene_indices(scenes: Sequence[Any]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, scene in enumerate(scenes):
        if isinstance(scene, dict) and "scene_id" in scene:
            indices[str(scene["scene_id"])] = index
    return indices


def _result(
    spec: SceneSpec,
    staged_path: Path,
    *,
    status: str,
    message: str = "",
) -> StagingResult:
    return StagingResult(
        scene_id=spec.scene_id,
        category=spec.category,
        source=spec.source,
        split=spec.split,
        input_path=spec.input_path,
        staged_path=staged_path,
        status=status,
        message=message,
    )


if __name__ == "__main__":
    raise SystemExit(main())
