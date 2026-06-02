"""Run PlaySplat over a filtered scene registry and aggregate benchmark results."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from playsplat.cli import main as run_pipeline_cli  # noqa: E402
from playsplat.experiments import SceneRecord, filter_scenes, load_scene_registry  # noqa: E402
from playsplat.utils.config import load_pipeline_settings  # noqa: E402
from playsplat.visualization import generate_scene_previews  # noqa: E402


SUMMARY_FIELDS = (
    "scene_id",
    "category",
    "source",
    "split",
    "input_path",
    "status",
    "error",
    "gaussian_count",
    "proxy_face_count",
    "collision_face_count",
    "collision_face_reduction_ratio",
    "simplification_status",
    "floor_area",
    "wall_area",
    "obstacle_area",
    "walkable_area",
    "walkable_area_ratio",
    "semantic_status",
    "affordance_status",
    "semantic_label_count",
    "affordance_label_count",
    "export_readiness_score",
    "overall_playability_score",
    "warning_count",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""

    parser = argparse.ArgumentParser(description="Run a PlaySplat multi-scene benchmark.")
    parser.add_argument(
        "--scene-registry",
        type=Path,
        required=True,
        help="Path to a PlaySplat scene registry YAML file.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Base PlaySplat YAML config to copy and override per scene.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/benchmark"),
        help="Directory where per-scene outputs and summaries are written.",
    )
    parser.add_argument("--split", type=str, default=None, help="Optional split filter.")
    parser.add_argument("--category", type=str, default=None, help="Optional category filter.")
    parser.add_argument("--source", type=str, default=None, help="Optional source filter.")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
        help="Voxel size used for all benchmark scenes.",
    )
    parser.add_argument(
        "--target-face-count",
        type=int,
        default=10_000,
        help="Collision simplification target face count for all benchmark scenes.",
    )
    parser.add_argument(
        "--generate-previews",
        action="store_true",
        help="Generate debug preview PNGs after each successful scene run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    records = filter_scenes(
        load_scene_registry(args.scene_registry),
        split=args.split,
        category=args.category,
        source=args.source,
    )
    load_pipeline_settings(args.base_config)
    base_config = load_yaml_config(args.base_config)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for record in records:
        scene_output_dir = output_root / record.scene_id
        config_path = scene_output_dir / "benchmark_config.yaml"
        print(f"Running benchmark scene '{record.scene_id}' -> {scene_output_dir}")
        row = run_single_benchmark_scene(
            record=record,
            base_config=base_config,
            output_dir=scene_output_dir,
            config_path=config_path,
            voxel_size=args.voxel_size,
            target_face_count=args.target_face_count,
            generate_previews=args.generate_previews,
        )
        rows.append(row)

    csv_path = write_benchmark_summary_csv(rows, output_root / "benchmark_summary.csv")
    json_path = write_benchmark_summary_json(rows, output_root / "benchmark_summary.json")
    report_path = write_benchmark_report(rows, output_root / "benchmark_report.md")
    print(f"Benchmark summary CSV saved: {csv_path}")
    print(f"Benchmark summary JSON saved: {json_path}")
    print(f"Benchmark report saved: {report_path}")
    return 0


def run_single_benchmark_scene(
    *,
    record: SceneRecord,
    base_config: dict[str, Any],
    output_dir: Path,
    config_path: Path,
    voxel_size: float,
    target_face_count: int,
    generate_previews: bool = False,
) -> dict[str, Any]:
    """Run one benchmark scene and return a summary row."""

    row = empty_summary_row(record)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        config = build_benchmark_config(
            base_config,
            record=record,
            output_dir=output_dir,
            voxel_size=voxel_size,
            target_face_count=target_face_count,
        )
        write_yaml_config(config, config_path)
        run_pipeline_cli(["--config", str(config_path)])
        report = load_report(output_dir / "playability_report.json")
        row.update(summary_from_report(report))
        row["status"] = str(report.get("status", "completed"))
        if generate_previews:
            previews = generate_scene_previews(output_dir)
            print(f"Generated {len(previews)} preview(s) for scene '{record.scene_id}'")
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        row["status"] = "failed"
        row["error"] = str(exc)
    return row


def build_benchmark_config(
    base_config: dict[str, Any],
    *,
    record: SceneRecord,
    output_dir: Path,
    voxel_size: float,
    target_face_count: int,
) -> dict[str, Any]:
    """Return a config copy with benchmark-specific overrides applied."""

    config = deepcopy(base_config)
    project_config = _ensure_mapping(config, "project")
    input_config = _ensure_mapping(config, "input")
    output_config = _ensure_mapping(config, "output")
    geometry_config = _ensure_mapping(config, "geometry")
    proxy_config = _ensure_mapping(geometry_config, "proxy")
    simplification_config = _ensure_mapping(geometry_config, "simplification")

    project_config["scene_id"] = record.scene_id
    input_config["path"] = str(record.input_path)
    output_config["directory"] = str(output_dir)
    proxy_config["voxel_size"] = voxel_size
    simplification_config["enabled"] = True
    simplification_config["target_face_count"] = target_face_count
    return config


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config mapping."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root of config: {path}")
    return data


def write_yaml_config(config: dict[str, Any], output_path: str | Path) -> Path:
    """Write a benchmark config YAML file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def write_benchmark_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write benchmark summary rows as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return path


def write_benchmark_summary_json(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write benchmark summary rows as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenes": rows}, indent=2) + "\n", encoding="utf-8")
    return path


def write_benchmark_report(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write an aggregate benchmark Markdown report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_benchmark_report(rows), encoding="utf-8")
    return path


def generate_benchmark_report(rows: Sequence[dict[str, Any]]) -> str:
    """Generate a benchmark aggregate report as Markdown."""

    successful_rows = [row for row in rows if row.get("status") == "ready_prototype"]
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    lines = [
        "# PlaySplat Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Total scenes: {len(rows)}",
        f"- Successful scenes: {len(successful_rows)}",
        f"- Failed scenes: {len(failed_rows)}",
        "- Average collision face count: "
        + _format_average(successful_rows, "collision_face_count", decimals=0),
        "- Average collision reduction ratio: "
        + _format_average(successful_rows, "collision_face_reduction_ratio", decimals=4),
        "- Average walkable area ratio: "
        + _format_average(successful_rows, "walkable_area_ratio", decimals=4),
        "- Scenes with geometry_semantic_layer: "
        + str(_count_status(successful_rows, "semantic_status", "geometry_semantic_layer")),
        "- Scenes with geometry_affordance_layer: "
        + str(_count_status(successful_rows, "affordance_status", "geometry_affordance_layer")),
        "",
    ]
    if rows:
        lines.extend(["## Category Summary", "", _category_summary_table(rows), ""])
    return "\n".join(lines)


def empty_summary_row(record: SceneRecord) -> dict[str, Any]:
    """Return an empty benchmark summary row for a scene record."""

    return {
        "scene_id": record.scene_id,
        "category": record.category,
        "source": record.source,
        "split": record.split,
        "input_path": str(record.input_path),
        "status": "pending",
        "error": "",
        "gaussian_count": "",
        "proxy_face_count": "",
        "collision_face_count": "",
        "collision_face_reduction_ratio": "",
        "simplification_status": "",
        "floor_area": "",
        "wall_area": "",
        "obstacle_area": "",
        "walkable_area": "",
        "walkable_area_ratio": "",
        "semantic_status": "",
        "affordance_status": "",
        "semantic_label_count": "",
        "affordance_label_count": "",
        "export_readiness_score": "",
        "overall_playability_score": "",
        "warning_count": "",
    }


def load_report(report_path: str | Path) -> dict[str, Any]:
    """Load a playability report JSON object."""

    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"Playability report not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected playability report JSON object: {path}")
    return data


def summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """Extract scalar benchmark fields from a playability report."""

    metrics = report.get("metrics", {})
    summary = report.get("summary", {})
    warnings = report.get("warnings", [])
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(warnings, list):
        warnings = []
    return {
        "gaussian_count": metrics.get("gaussian_count", ""),
        "proxy_face_count": metrics.get("proxy_face_count", ""),
        "collision_face_count": metrics.get("collision_face_count", ""),
        "collision_face_reduction_ratio": metrics.get("collision_face_reduction_ratio", ""),
        "simplification_status": metrics.get("simplification_status", ""),
        "floor_area": metrics.get("floor_area", ""),
        "wall_area": metrics.get("wall_area", ""),
        "obstacle_area": metrics.get("obstacle_area", ""),
        "walkable_area": metrics.get("walkable_area", ""),
        "walkable_area_ratio": metrics.get("walkable_area_ratio", ""),
        "semantic_status": metrics.get("semantic_status", ""),
        "affordance_status": metrics.get("affordance_status", ""),
        "semantic_label_count": metrics.get("semantic_label_count", ""),
        "affordance_label_count": metrics.get("affordance_label_count", ""),
        "export_readiness_score": metrics.get("export_readiness_score", ""),
        "overall_playability_score": metrics.get(
            "overall_playability_score",
            summary.get("overall_playability_score", ""),
        ),
        "warning_count": summary.get("warning_count", len(warnings)),
        "error": "",
    }


def _ensure_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        nested: dict[str, Any] = {}
        config[key] = nested
        return nested
    if not isinstance(value, dict):
        raise ValueError(f"Expected config section '{key}' to be a mapping.")
    return value


def _format_average(rows: Sequence[dict[str, Any]], key: str, *, decimals: int) -> str:
    values = [_float(row.get(key)) for row in rows if _is_float(row.get(key))]
    if not values:
        return "n/a"
    average = sum(values) / len(values)
    if decimals == 0:
        return str(int(round(average)))
    return f"{average:.{decimals}f}"


def _count_status(rows: Sequence[dict[str, Any]], key: str, value: str) -> int:
    return sum(1 for row in rows if row.get(key) == value)


def _category_summary_table(rows: Sequence[dict[str, Any]]) -> str:
    categories = sorted({str(row.get("category", "")) for row in rows if row.get("category")})
    lines = [
        "| category | scenes | successful | avg_collision_faces | avg_walkable_ratio | avg_score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for category in categories:
        category_rows = [row for row in rows if row.get("category") == category]
        successful_rows = [row for row in category_rows if row.get("status") == "ready_prototype"]
        lines.append(
            "| "
            + " | ".join(
                (
                    category,
                    str(len(category_rows)),
                    str(len(successful_rows)),
                    _format_average(successful_rows, "collision_face_count", decimals=0),
                    _format_average(successful_rows, "walkable_area_ratio", decimals=4),
                    _format_average(successful_rows, "overall_playability_score", decimals=4),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _is_float(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _float(value: Any) -> float:
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
