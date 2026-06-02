"""Run PlaySplat over one or more Gaussian PLY scenes and summarize outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


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
    "status",
    "error",
    "gaussian_count",
    "proxy_vertex_count",
    "proxy_face_count",
    "collision_face_count",
    "collision_face_reduction_ratio",
    "simplification_status",
    "semantic_status",
    "affordance_status",
    "semantic_label_count",
    "affordance_label_count",
    "floor_area",
    "wall_area",
    "obstacle_area",
    "walkable_area",
    "walkable_area_ratio",
    "obstacle_area_ratio",
    "export_readiness_score",
    "overall_playability_score",
    "warning_count",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment runner CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run PlaySplat on one or more Gaussian PLY scenes.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Gaussian Splatting .ply scene files.",
    )
    parser.add_argument(
        "--scene-id",
        nargs="*",
        default=None,
        help="Optional scene id. Provide one id for one input, or one id per input.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/experiments"),
        help="Directory where per-scene outputs and experiment summaries are written.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Base PlaySplat YAML config.",
    )
    parser.add_argument(
        "--generate-previews",
        action="store_true",
        help="Generate debug preview PNGs after each successful scene run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    load_pipeline_settings(args.config)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scene_ids = _scene_ids_for_inputs(args.input, args.scene_id)

    rows: list[dict[str, Any]] = []
    for input_path, scene_id in zip(args.input, scene_ids, strict=True):
        scene_output_dir = output_root / scene_id
        print(f"Running PlaySplat scene '{scene_id}' -> {scene_output_dir}")
        row = run_single_scene(
            input_path=input_path,
            scene_id=scene_id,
            output_dir=scene_output_dir,
            config_path=args.config,
            generate_previews=args.generate_previews,
        )
        rows.append(row)

    csv_path = write_experiment_summary_csv(rows, output_root / "experiment_summary.csv")
    json_path = write_experiment_summary_json(rows, output_root / "experiment_summary.json")
    print(f"Experiment summary CSV saved: {csv_path}")
    print(f"Experiment summary JSON saved: {json_path}")
    return 0


def run_single_scene(
    *,
    input_path: Path,
    scene_id: str,
    output_dir: Path,
    config_path: Path,
    generate_previews: bool = False,
) -> dict[str, Any]:
    """Run one scene and return a summary row."""

    row = _empty_summary_row(scene_id=scene_id, input_path=input_path)
    try:
        run_pipeline_cli(
            [
                "--config",
                str(config_path),
                "--input",
                str(input_path),
                "--scene-id",
                scene_id,
                "--output-dir",
                str(output_dir),
            ]
        )
        report = _load_report(output_dir / "playability_report.json")
        row.update(_summary_from_report(report))
        row["status"] = str(report.get("status", "completed"))
        if generate_previews:
            previews = generate_scene_previews(output_dir)
            print(f"Generated {len(previews)} preview(s) for scene '{scene_id}'")
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        row["status"] = "failed"
        row["error"] = str(exc)
    return row


def write_experiment_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write experiment summary rows as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return path


def write_experiment_summary_json(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write experiment summary rows as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenes": rows}, indent=2) + "\n", encoding="utf-8")
    return path


def _scene_ids_for_inputs(inputs: Sequence[Path], scene_ids: Sequence[str] | None) -> list[str]:
    if scene_ids is None or len(scene_ids) == 0:
        raw_ids = [path.stem for path in inputs]
    elif len(scene_ids) == len(inputs):
        raw_ids = list(scene_ids)
    elif len(scene_ids) == 1 and len(inputs) == 1:
        raw_ids = [scene_ids[0]]
    else:
        raise ValueError("--scene-id must be omitted, supplied once for one input, or supplied per input.")
    return _unique_scene_ids([_sanitize_scene_id(scene_id) for scene_id in raw_ids])


def _sanitize_scene_id(scene_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_id.strip())
    normalized = normalized.strip("._-")
    return normalized or "scene"


def _unique_scene_ids(scene_ids: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for scene_id in scene_ids:
        count = counts.get(scene_id, 0)
        counts[scene_id] = count + 1
        unique.append(scene_id if count == 0 else f"{scene_id}_{count + 1}")
    return unique


def _empty_summary_row(scene_id: str, input_path: Path) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "input_path": str(input_path),
        "status": "pending",
        "error": "",
        "gaussian_count": "",
        "proxy_vertex_count": "",
        "proxy_face_count": "",
        "collision_face_count": "",
        "collision_face_reduction_ratio": "",
        "simplification_status": "",
        "semantic_status": "",
        "affordance_status": "",
        "semantic_label_count": "",
        "affordance_label_count": "",
        "floor_area": "",
        "wall_area": "",
        "obstacle_area": "",
        "walkable_area": "",
        "walkable_area_ratio": "",
        "obstacle_area_ratio": "",
        "export_readiness_score": "",
        "overall_playability_score": "",
        "warning_count": "",
    }


def _load_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(f"Playability report not found: {report_path}")
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected playability report JSON object: {report_path}")
    return data


def _summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
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
        "simplification_status": metrics.get("simplification_status", ""),
        "semantic_status": metrics.get("semantic_status", ""),
        "affordance_status": metrics.get("affordance_status", ""),
        "semantic_label_count": metrics.get("semantic_label_count", ""),
        "affordance_label_count": metrics.get("affordance_label_count", ""),
        "floor_area": metrics.get("floor_area", ""),
        "wall_area": metrics.get("wall_area", ""),
        "obstacle_area": metrics.get("obstacle_area", ""),
        "walkable_area": metrics.get("walkable_area", ""),
        "walkable_area_ratio": metrics.get("walkable_area_ratio", ""),
        "obstacle_area_ratio": metrics.get("obstacle_area_ratio", ""),
        "export_readiness_score": metrics.get("export_readiness_score", ""),
        "overall_playability_score": metrics.get(
            "overall_playability_score",
            summary.get("overall_playability_score", ""),
        ),
        "warning_count": summary.get("warning_count", len(warnings)),
        "error": "",
    }


if __name__ == "__main__":
    raise SystemExit(main())
