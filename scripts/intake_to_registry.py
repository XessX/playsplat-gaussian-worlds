"""Convert a scene intake CSV into a local PlaySplat scene registry."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import yaml


DEFAULT_INPUT_CSV = Path("docs/scene_intake_template.csv")
DEFAULT_REGISTRY = Path("configs/scenes.local.yaml")
INTAKE_COLUMNS = (
    "scene_id",
    "category",
    "capture_location_type",
    "source",
    "split",
    "raw_capture_path",
    "trained_3dgs_output_path",
    "staged_path",
    "image_count",
    "training_iteration",
    "notes",
    "license_or_permission",
    "privacy_checked",
    "ready_for_benchmark",
)
TRUE_VALUES = frozenset({"1", "true", "yes", "y", "on"})
SUCCESS_STATUSES = frozenset({"written", "updated", "dry_run_add", "dry_run_update"})


@dataclass(frozen=True)
class IntakeRow:
    """One row from the independent scene intake sheet."""

    row_number: int
    scene_id: str
    category: str
    capture_location_type: str
    source: str
    split: str
    raw_capture_path: str
    trained_3dgs_output_path: str
    staged_path: str
    image_count: str
    training_iteration: str
    notes: str
    license_or_permission: str
    privacy_checked: str
    ready_for_benchmark: str

    @property
    def is_ready(self) -> bool:
        """Whether this row is marked ready for benchmark use."""

        return normalize_bool(self.ready_for_benchmark)

    @property
    def input_path(self) -> str:
        """Return the preferred registry input path for this row."""

        return self.staged_path or self.trained_3dgs_output_path


@dataclass(frozen=True)
class IntakeReadResult:
    """Parsed intake sheet rows and row-level warnings."""

    rows_read: int
    malformed_rows: int
    rows: tuple[IntakeRow, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class IntakeRegistryResult:
    """Registry conversion outcome for one intake row."""

    row_number: int
    scene_id: str
    status: str
    message: str = ""


@dataclass(frozen=True)
class IntakeRegistrySummary:
    """Summary of an intake-to-registry conversion."""

    rows_read: int
    ready_rows: int
    skipped_rows: int
    registry_records_written: int
    results: tuple[IntakeRegistryResult, ...]
    warnings: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Build the intake conversion CLI parser."""

    parser = argparse.ArgumentParser(
        description="Convert a PlaySplat scene intake CSV into configs/scenes.local.yaml.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Scene intake CSV to read.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Local scene registry YAML to write or update.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned registry updates without writing the registry.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Update existing non-debug registry records with matching scene IDs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the intake-to-registry CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    summary = update_registry_from_intake(
        input_csv=args.input_csv,
        registry_path=args.registry,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print_intake_summary(summary)
    return 0


def update_registry_from_intake(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> IntakeRegistrySummary:
    """Read an intake CSV and update a local scene registry."""

    read_result = read_intake_csv(input_csv)
    registry = load_registry_mapping(registry_path)
    scenes = _registry_scenes(registry)
    existing_indices = _scene_indices(scenes)
    updated_scenes = list(scenes)
    results: list[IntakeRegistryResult] = []
    records_written = 0

    for row in read_result.rows:
        if not row.is_ready:
            results.append(
                IntakeRegistryResult(
                    row_number=row.row_number,
                    scene_id=row.scene_id,
                    status="not_ready",
                    message="Row is not marked ready_for_benchmark.",
                )
            )
            continue

        input_path = normalize_registry_path(row.input_path)
        if not input_path:
            results.append(
                IntakeRegistryResult(
                    row_number=row.row_number,
                    scene_id=row.scene_id,
                    status="missing_input_path",
                    message="Ready row has no staged_path or trained_3dgs_output_path.",
                )
            )
            continue

        record = intake_row_to_registry_record(row, input_path=input_path)
        existing_index = existing_indices.get(row.scene_id)
        if existing_index is not None:
            existing_scene = updated_scenes[existing_index]
            if _is_debug_scene(existing_scene):
                results.append(
                    IntakeRegistryResult(
                        row_number=row.row_number,
                        scene_id=row.scene_id,
                        status="preserved_debug_scene",
                        message="Existing debug scene was preserved.",
                    )
                )
                continue
            if not overwrite:
                results.append(
                    IntakeRegistryResult(
                        row_number=row.row_number,
                        scene_id=row.scene_id,
                        status="duplicate_scene_id",
                        message="Scene id already exists; use --overwrite to update it.",
                    )
                )
                continue
            if dry_run:
                results.append(
                    IntakeRegistryResult(
                        row_number=row.row_number,
                        scene_id=row.scene_id,
                        status="dry_run_update",
                    )
                )
                continue
            updated_scenes[existing_index] = record
            records_written += 1
            results.append(
                IntakeRegistryResult(
                    row_number=row.row_number,
                    scene_id=row.scene_id,
                    status="updated",
                )
            )
            continue

        if dry_run:
            results.append(
                IntakeRegistryResult(
                    row_number=row.row_number,
                    scene_id=row.scene_id,
                    status="dry_run_add",
                )
            )
            continue

        existing_indices[row.scene_id] = len(updated_scenes)
        updated_scenes.append(record)
        records_written += 1
        results.append(
            IntakeRegistryResult(
                row_number=row.row_number,
                scene_id=row.scene_id,
                status="written",
            )
        )

    if records_written and not dry_run:
        registry["scenes"] = updated_scenes
        write_registry_mapping(registry, registry_path)

    skipped_rows = read_result.malformed_rows + sum(
        1 for result in results if result.status not in SUCCESS_STATUSES
    )
    return IntakeRegistrySummary(
        rows_read=read_result.rows_read,
        ready_rows=sum(1 for row in read_result.rows if row.is_ready),
        skipped_rows=skipped_rows,
        registry_records_written=records_written,
        results=tuple(results),
        warnings=read_result.warnings,
    )


def read_intake_csv(path: str | Path) -> IntakeReadResult:
    """Read scene intake rows from a CSV file."""

    input_path = Path(path)
    rows: list[IntakeRow] = []
    warnings: list[str] = []
    malformed_rows = 0
    rows_read = 0

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [column for column in INTAKE_COLUMNS if column not in fieldnames]
        if missing_columns:
            warnings.append(
                "CSV is missing expected columns: " + ", ".join(sorted(missing_columns))
            )

        for row_number, raw_row in enumerate(reader, start=2):
            rows_read += 1
            raw_mapping = cast(Mapping[str | None, object], raw_row)
            if None in raw_mapping:
                malformed_rows += 1
                warnings.append(f"row {row_number} skipped: unexpected extra CSV fields.")
                continue
            row, warning = _parse_intake_row(row_number, raw_mapping)
            if warning:
                malformed_rows += 1
                warnings.append(warning)
                continue
            if row is not None:
                rows.append(row)

    return IntakeReadResult(
        rows_read=rows_read,
        malformed_rows=malformed_rows,
        rows=tuple(rows),
        warnings=tuple(warnings),
    )


def intake_row_to_registry_record(row: IntakeRow, *, input_path: str | None = None) -> dict[str, str]:
    """Convert one ready intake row into a scene registry record."""

    registry_input_path = normalize_registry_path(input_path or row.input_path)
    return {
        "scene_id": row.scene_id,
        "input_path": registry_input_path,
        "category": row.category,
        "source": row.source or "independent",
        "split": row.split or "benchmark",
        "notes": row.notes,
    }


def normalize_bool(value: str) -> bool:
    """Normalize common truthy strings from intake sheets."""

    return value.strip().lower() in TRUE_VALUES


def normalize_registry_path(path: str) -> str:
    """Normalize a registry path while preserving local absolute paths."""

    return path.strip().replace("\\", "/")


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
    return cast(dict[str, Any], data)


def write_registry_mapping(registry: dict[str, Any], path: str | Path) -> Path:
    """Write a registry YAML mapping."""

    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )
    return registry_path


def print_intake_summary(summary: IntakeRegistrySummary) -> None:
    """Print a concise intake conversion summary."""

    print(f"rows read: {summary.rows_read}")
    print(f"ready rows: {summary.ready_rows}")
    print(f"skipped rows: {summary.skipped_rows}")
    print(f"registry records written: {summary.registry_records_written}")

    if summary.warnings:
        print("warnings:")
        for warning in summary.warnings:
            print(f"  {warning}")

    if not summary.results:
        return

    headers = ("row", "scene_id", "status", "message")
    rows = [
        (
            str(result.row_number),
            result.scene_id,
            result.status,
            result.message,
        )
        for result in summary.results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _parse_intake_row(
    row_number: int,
    mapping: Mapping[str | None, object],
) -> tuple[IntakeRow | None, str | None]:
    values = {
        column: _clean_text(mapping.get(column))
        for column in INTAKE_COLUMNS
    }
    scene_id = values["scene_id"]
    category = values["category"]
    if not scene_id:
        return None, f"row {row_number} skipped: missing scene_id."
    if not category:
        return None, f"row {row_number} skipped: missing category for scene_id {scene_id}."

    return (
        IntakeRow(
            row_number=row_number,
            scene_id=scene_id,
            category=category,
            capture_location_type=values["capture_location_type"],
            source=values["source"] or "independent",
            split=values["split"] or "benchmark",
            raw_capture_path=values["raw_capture_path"],
            trained_3dgs_output_path=values["trained_3dgs_output_path"],
            staged_path=values["staged_path"],
            image_count=values["image_count"],
            training_iteration=values["training_iteration"],
            notes=values["notes"],
            license_or_permission=values["license_or_permission"],
            privacy_checked=values["privacy_checked"],
            ready_for_benchmark=values["ready_for_benchmark"],
        ),
        None,
    )


def _registry_scenes(registry: dict[str, Any]) -> list[dict[str, Any]]:
    scenes_data = registry.setdefault("scenes", [])
    if not isinstance(scenes_data, list):
        raise ValueError("Registry 'scenes' field must be a list.")

    scenes: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes_data):
        if not isinstance(scene, dict):
            raise ValueError(f"Registry scene entry at index {index} must be a mapping.")
        scenes.append({str(key): value for key, value in scene.items()})
    return scenes


def _scene_indices(scenes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, scene in enumerate(scenes):
        scene_id = scene.get("scene_id")
        if scene_id is not None:
            indices[str(scene_id)] = index
    return indices


def _is_debug_scene(scene: Mapping[str, Any]) -> bool:
    return str(scene.get("split", "")) == "debug" or str(scene.get("source", "")) in {
        "debug",
        "internal_debug",
    }


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    raise SystemExit(main())
