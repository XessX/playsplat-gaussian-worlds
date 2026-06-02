"""Run PlaySplat ablations over voxel and collision simplification settings."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from playsplat.cli import main as run_pipeline_cli  # noqa: E402
from playsplat.utils.config import load_pipeline_settings  # noqa: E402
from playsplat.visualization import generate_scene_previews  # noqa: E402


SUMMARY_FIELDS = (
    "scene_id",
    "input_path",
    "voxel_size",
    "target_face_count",
    "status",
    "error",
    "gaussian_count",
    "proxy_vertex_count",
    "proxy_face_count",
    "collision_face_count",
    "collision_face_reduction_ratio",
    "collision_face_to_proxy_face_ratio",
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
    """Build the ablation experiment CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run PlaySplat ablations over voxel size and collision face targets.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Gaussian Splatting .ply scene file.",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        required=True,
        help="Base scene id used for ablation run names.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Base PlaySplat YAML config to copy and override.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations"),
        help="Directory where ablation outputs and summaries are written.",
    )
    parser.add_argument(
        "--voxel-sizes",
        type=float,
        nargs="+",
        required=True,
        help="Voxel sizes to evaluate.",
    )
    parser.add_argument(
        "--target-face-counts",
        type=int,
        nargs="+",
        required=True,
        help="Collision simplification target face counts to evaluate.",
    )
    parser.add_argument(
        "--generate-previews",
        action="store_true",
        help="Generate debug preview PNGs after each successful ablation run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ablation experiment CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    load_pipeline_settings(args.base_config)
    base_config = load_yaml_config(args.base_config)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for voxel_size in args.voxel_sizes:
        for target_face_count in args.target_face_counts:
            scene_id = format_ablation_scene_id(
                args.scene_id,
                voxel_size=voxel_size,
                target_face_count=target_face_count,
            )
            scene_output_dir = output_root / scene_id
            config_path = scene_output_dir / "ablation_config.yaml"
            print(
                "Running ablation "
                f"'{scene_id}' voxel_size={voxel_size:g} "
                f"target_face_count={target_face_count} -> {scene_output_dir}"
            )
            row = run_single_ablation(
                base_config=base_config,
                input_path=args.input,
                scene_id=scene_id,
                output_dir=scene_output_dir,
                config_path=config_path,
                voxel_size=voxel_size,
                target_face_count=target_face_count,
                generate_previews=args.generate_previews,
            )
            rows.append(row)

    csv_path = write_ablation_summary_csv(rows, output_root / "ablation_summary.csv")
    json_path = write_ablation_summary_json(rows, output_root / "ablation_summary.json")
    best_path = write_best_runs_json(rows, output_root / "best_runs.json")
    print(f"Ablation summary CSV saved: {csv_path}")
    print(f"Ablation summary JSON saved: {json_path}")
    print(f"Best-run summary saved: {best_path}")
    return 0


def run_single_ablation(
    *,
    base_config: dict[str, Any],
    input_path: Path,
    scene_id: str,
    output_dir: Path,
    config_path: Path,
    voxel_size: float,
    target_face_count: int,
    generate_previews: bool = False,
) -> dict[str, Any]:
    """Run one ablation setting and return a summary row."""

    row = empty_summary_row(
        scene_id=scene_id,
        input_path=input_path,
        voxel_size=voxel_size,
        target_face_count=target_face_count,
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        config = build_ablation_config(
            base_config,
            input_path=input_path,
            scene_id=scene_id,
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
            print(f"Generated {len(previews)} preview(s) for ablation '{scene_id}'")
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        row["status"] = "failed"
        row["error"] = str(exc)
    return row


def format_ablation_scene_id(
    scene_id: str,
    *,
    voxel_size: float,
    target_face_count: int,
) -> str:
    """Format a clean deterministic scene id for one ablation setting."""

    base = _sanitize_scene_id(scene_id)
    voxel_token = _format_float_token(voxel_size)
    return f"{base}_vx{voxel_token}_faces{target_face_count}"


def build_ablation_config(
    base_config: dict[str, Any],
    *,
    input_path: Path,
    scene_id: str,
    output_dir: Path,
    voxel_size: float,
    target_face_count: int,
) -> dict[str, Any]:
    """Return a config copy with ablation-specific overrides applied."""

    config = deepcopy(base_config)
    project_config = _ensure_mapping(config, "project")
    input_config = _ensure_mapping(config, "input")
    output_config = _ensure_mapping(config, "output")
    geometry_config = _ensure_mapping(config, "geometry")
    proxy_config = _ensure_mapping(geometry_config, "proxy")
    simplification_config = _ensure_mapping(geometry_config, "simplification")

    project_config["scene_id"] = scene_id
    input_config["path"] = str(input_path)
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
    """Write an ablation config YAML file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_ablation_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write ablation summary rows as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return path


def write_ablation_summary_json(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write ablation summary rows as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": rows}, indent=2) + "\n", encoding="utf-8")
    return path


def write_best_runs_json(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write best-run selections as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(select_best_runs(rows), indent=2) + "\n", encoding="utf-8")
    return path


def select_best_runs(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select simple best-run summaries from completed ablation rows."""

    ready_rows = [row for row in rows if row.get("status") == "ready_prototype"]
    best_lowest_collision = _min_by_metric(ready_rows, "collision_face_count")
    best_walkable_ratio = _max_by_metric(ready_rows, "walkable_area_ratio")
    best_balanced = _best_balanced_run(ready_rows)
    return {
        "selection_policy": {
            "eligible_status": "ready_prototype",
            "lowest_collision_face_count": "Minimum collision_face_count among eligible runs.",
            "highest_walkable_area_ratio": "Maximum walkable_area_ratio among eligible runs.",
            "balanced": (
                "Average of normalized inverse collision_face_count, normalized "
                "walkable_area_ratio, and normalized inverse warning_count."
            ),
        },
        "eligible_run_count": len(ready_rows),
        "best_lowest_collision_face_count": best_lowest_collision,
        "best_highest_walkable_area_ratio": best_walkable_ratio,
        "best_balanced": best_balanced,
    }


def empty_summary_row(
    *,
    scene_id: str,
    input_path: Path,
    voxel_size: float,
    target_face_count: int,
) -> dict[str, Any]:
    """Return an empty ablation summary row."""

    return {
        "scene_id": scene_id,
        "input_path": str(input_path),
        "voxel_size": voxel_size,
        "target_face_count": target_face_count,
        "status": "pending",
        "error": "",
        "gaussian_count": "",
        "proxy_vertex_count": "",
        "proxy_face_count": "",
        "collision_face_count": "",
        "collision_face_reduction_ratio": "",
        "collision_face_to_proxy_face_ratio": "",
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
    """Extract scalar summary fields from a playability report."""

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
        "proxy_vertex_count": metrics.get("proxy_vertex_count", ""),
        "proxy_face_count": metrics.get("proxy_face_count", ""),
        "collision_face_count": metrics.get("collision_face_count", ""),
        "collision_face_reduction_ratio": metrics.get("collision_face_reduction_ratio", ""),
        "collision_face_to_proxy_face_ratio": metrics.get(
            "collision_face_to_proxy_face_ratio",
            "",
        ),
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


def _sanitize_scene_id(scene_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_id.strip())
    normalized = normalized.strip("._-")
    return normalized or "scene"


def _format_float_token(value: float) -> str:
    token = f"{value:g}".replace("-", "m").replace(".", "p")
    return _sanitize_scene_id(token)


def _min_by_metric(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    scored = [(float(row[metric]), row) for row in rows if _is_number(row.get(metric))]
    if not scored:
        return None
    return dict(min(scored, key=lambda item: item[0])[1])


def _max_by_metric(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    scored = [(float(row[metric]), row) for row in rows if _is_number(row.get(metric))]
    if not scored:
        return None
    return dict(max(scored, key=lambda item: item[0])[1])


def _best_balanced_run(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if _is_number(row.get("collision_face_count"))
        and _is_number(row.get("walkable_area_ratio"))
        and _is_number(row.get("warning_count"))
    ]
    if not candidates:
        return None

    collision_counts = [float(row["collision_face_count"]) for row in candidates]
    walkable_ratios = [float(row["walkable_area_ratio"]) for row in candidates]
    warning_counts = [float(row["warning_count"]) for row in candidates]

    scored_rows: list[tuple[float, dict[str, Any]]] = []
    for row in candidates:
        collision_component = 1.0 - _normalize(
            float(row["collision_face_count"]),
            collision_counts,
        )
        walkable_component = _normalize(float(row["walkable_area_ratio"]), walkable_ratios)
        warning_component = 1.0 - _normalize(float(row["warning_count"]), warning_counts)
        balanced_score = round(
            (collision_component + walkable_component + warning_component) / 3.0,
            7,
        )
        row_copy = dict(row)
        row_copy["balanced_selection_score"] = balanced_score
        scored_rows.append((balanced_score, row_copy))
    return max(scored_rows, key=lambda item: item[0])[1]


def _normalize(value: float, values: Sequence[float]) -> float:
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return 1.0
    return (value - minimum) / (maximum - minimum)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
