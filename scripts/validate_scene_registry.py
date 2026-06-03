"""Validate a PlaySplat local scene registry before benchmark runs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


REQUIRED_FIELDS = ("scene_id", "input_path", "category", "source", "split", "notes")
SUSPICIOUS_PATTERNS = (
    "sparse3d_scirep_starter",
    "scientific reports",
    "scientific_reports",
    "playsplat/outputs",
    "outputs/",
)


@dataclass(frozen=True)
class RegistryValidationReport:
    """Validation summary for a scene registry."""

    total_scenes: int
    debug_scenes: int
    benchmark_scenes: int
    duplicate_scene_ids: tuple[str, ...]
    missing_files: tuple[str, ...]
    missing_benchmark_files: tuple[str, ...]
    suspicious_benchmark_paths: tuple[str, ...]
    categories: tuple[str, ...]
    source_counts: dict[str, int]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Whether the registry is safe to use for benchmark execution."""

        return not (
            self.duplicate_scene_ids
            or self.missing_benchmark_files
            or self.suspicious_benchmark_paths
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the validator CLI parser."""

    parser = argparse.ArgumentParser(description="Validate a PlaySplat scene registry.")
    parser.add_argument(
        "--scene-registry",
        type=Path,
        required=True,
        help="Scene registry YAML file to validate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run registry validation."""

    parser = build_parser()
    args = parser.parse_args(argv)
    report = validate_scene_registry(args.scene_registry)
    print_validation_report(report)
    return 0 if report.is_valid else 1


def validate_scene_registry(path: str | Path) -> RegistryValidationReport:
    """Validate registry content and benchmark-readiness conditions."""

    scenes = _load_scene_entries(path)
    scene_ids = [str(scene.get("scene_id", "")) for scene in scenes]
    duplicate_scene_ids = tuple(
        sorted(scene_id for scene_id, count in Counter(scene_ids).items() if scene_id and count > 1)
    )

    missing_files: list[str] = []
    missing_benchmark_files: list[str] = []
    suspicious_benchmark_paths: list[str] = []
    categories: set[str] = set()
    source_counts: Counter[str] = Counter()
    debug_scenes = 0
    benchmark_scenes = 0

    for index, scene in enumerate(scenes):
        _validate_required_fields(scene, index)
        split = str(scene["split"])
        source = str(scene["source"])
        category = str(scene["category"])
        input_path = Path(str(scene["input_path"]))
        scene_id = str(scene["scene_id"])

        categories.add(category)
        source_counts[source] += 1
        if split == "debug":
            debug_scenes += 1
        if split == "benchmark":
            benchmark_scenes += 1

        if not input_path.exists():
            missing_message = f"{scene_id}: {input_path}"
            missing_files.append(missing_message)
            if split == "benchmark":
                missing_benchmark_files.append(missing_message)

        suspicious_reason = suspicious_reason_for_scene(scene)
        if split == "benchmark" and suspicious_reason:
            suspicious_benchmark_paths.append(f"{scene_id}: {suspicious_reason}")

    warnings = _registry_warnings(
        benchmark_scenes=benchmark_scenes,
        categories=categories,
        missing_files=missing_files,
    )
    return RegistryValidationReport(
        total_scenes=len(scenes),
        debug_scenes=debug_scenes,
        benchmark_scenes=benchmark_scenes,
        duplicate_scene_ids=duplicate_scene_ids,
        missing_files=tuple(missing_files),
        missing_benchmark_files=tuple(missing_benchmark_files),
        suspicious_benchmark_paths=tuple(suspicious_benchmark_paths),
        categories=tuple(sorted(categories)),
        source_counts=dict(sorted(source_counts.items())),
        warnings=warnings,
    )


def print_validation_report(report: RegistryValidationReport) -> None:
    """Print a concise registry validation report."""

    print(f"total scenes: {report.total_scenes}")
    print(f"debug scenes: {report.debug_scenes}")
    print(f"benchmark scenes: {report.benchmark_scenes}")
    print(f"missing files: {len(report.missing_files)}")
    print(f"duplicate scene IDs: {', '.join(report.duplicate_scene_ids) or 'none'}")
    print(f"categories represented: {', '.join(report.categories) or 'none'}")
    print("source counts:")
    for source, count in report.source_counts.items():
        print(f"  {source}: {count}")

    if report.missing_benchmark_files:
        print("missing benchmark files:")
        for item in report.missing_benchmark_files:
            print(f"  {item}")
    if report.suspicious_benchmark_paths:
        print("suspicious benchmark paths:")
        for item in report.suspicious_benchmark_paths:
            print(f"  {item}")
    if report.warnings:
        print("warnings:")
        for warning in report.warnings:
            print(f"  {warning}")
    print(f"registry valid: {'yes' if report.is_valid else 'no'}")


def suspicious_reason_for_scene(scene: dict[str, Any]) -> str:
    """Return a suspicious benchmark reason for a scene, if any."""

    scene_id = str(scene.get("scene_id", ""))
    source = str(scene.get("source", ""))
    input_path = str(scene.get("input_path", ""))
    normalized_path = input_path.lower().replace("\\", "/")
    for pattern in SUSPICIOUS_PATTERNS:
        if pattern in normalized_path:
            return f"path for {scene_id} matches suspicious pattern '{pattern}'"
    if source in {"internal_debug", "debug"}:
        return f"source for {scene_id} is {source}, not independent"
    return ""


def _load_scene_entries(path: str | Path) -> list[dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Scene registry not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root of registry: {registry_path}")
    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("Scene registry must contain a 'scenes' list.")
    entries: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene entry at index {index} must be a mapping.")
        entries.append(scene)
    return entries


def _validate_required_fields(scene: dict[str, Any], index: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in scene]
    if missing:
        raise ValueError(
            f"Scene entry at index {index} is missing required fields: " + ", ".join(missing)
        )


def _registry_warnings(
    *,
    benchmark_scenes: int,
    categories: set[str],
    missing_files: Sequence[str],
) -> tuple[str, ...]:
    warnings: list[str] = []
    if benchmark_scenes < 5:
        warnings.append(
            f"fewer than 5 benchmark scenes are registered ({benchmark_scenes}); "
            "paper evidence should use 5-8 independent scenes."
        )
    if len(categories) < 3:
        warnings.append(
            f"fewer than 3 categories are represented ({len(categories)}); "
            "add more geometry diversity before paper benchmarks."
        )
    if missing_files:
        warnings.append(f"{len(missing_files)} scene input file(s) are missing.")
    return tuple(warnings)


if __name__ == "__main__":
    raise SystemExit(main())
