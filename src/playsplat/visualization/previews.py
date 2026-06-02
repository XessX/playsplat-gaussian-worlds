"""Preview image generation for PlaySplat debug outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import trimesh

from playsplat.io import load_gaussian_ply


def render_mesh_preview(
    mesh_path: str | Path,
    output_path: str | Path,
    title: str | None = None,
    max_faces: int = 200_000,
) -> Path:
    """Render a simple 3D mesh preview PNG from an OBJ, PLY, or GLB file."""

    if max_faces <= 0:
        raise ValueError(f"max_faces must be positive; got {max_faces}.")
    source = Path(mesh_path)
    if not source.exists():
        raise FileNotFoundError(f"Mesh file not found: {source}")
    mesh = _load_mesh(source)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.shape[0] > max_faces:
        faces = faces[_deterministic_indices(faces.shape[0], max_faces)]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(8, 7), dpi=140)
    axis = figure.add_subplot(111, projection="3d")
    if faces.size > 0:
        triangles = vertices[faces]
        axis.plot_trisurf(
            triangles[:, :, 0].reshape(-1),
            triangles[:, :, 1].reshape(-1),
            triangles[:, :, 2].reshape(-1),
            triangles=np.arange(triangles.shape[0] * 3).reshape(-1, 3),
            color="#9aa6b2",
            edgecolor="#3d4852",
            linewidth=0.05,
            alpha=0.92,
        )
    axis.set_title(title or source.name)
    _style_3d_axis(axis, vertices)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def render_point_cloud_preview(
    ply_path: str | Path,
    output_path: str | Path,
    title: str | None = None,
    max_points: int = 100_000,
) -> Path:
    """Render a Gaussian position scatter preview PNG from a PLY file."""

    if max_points <= 0:
        raise ValueError(f"max_points must be positive; got {max_points}.")
    source = Path(ply_path)
    layer = load_gaussian_ply(source)
    positions = layer.positions
    if positions.shape[0] > max_points:
        positions = positions[_deterministic_indices(positions.shape[0], max_points)]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(8, 7), dpi=140)
    axis = figure.add_subplot(111, projection="3d")
    colors = _point_colors(layer.colors, positions.shape[0])
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=colors,
        s=1.0,
        alpha=0.85,
        linewidths=0,
    )
    axis.set_title(title or source.name)
    _style_3d_axis(axis, positions)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def render_playability_summary_card(report_path: str | Path, output_path: str | Path) -> Path:
    """Render a compact playability metrics summary PNG."""

    source = Path(report_path)
    if not source.exists():
        raise FileNotFoundError(f"Playability report not found: {source}")
    report = json.loads(source.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    summary = report.get("summary", {})
    warnings = report.get("warnings", [])
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(warnings, list):
        warnings = []

    rows = [
        ("status", report.get("status", summary.get("status", "unknown"))),
        ("gaussian_count", metrics.get("gaussian_count", "n/a")),
        ("proxy_face_count", metrics.get("proxy_face_count", "n/a")),
        ("floor_area", metrics.get("floor_area", "n/a")),
        ("wall_area", metrics.get("wall_area", "n/a")),
        ("obstacle_area", metrics.get("obstacle_area", "n/a")),
        ("walkable_area", metrics.get("walkable_area", "n/a")),
        ("export_readiness_score", metrics.get("export_readiness_score", "n/a")),
        ("overall_playability_score", metrics.get("overall_playability_score", "n/a")),
        ("warning_count", summary.get("warning_count", len(warnings))),
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5.5), dpi=150)
    axis.axis("off")
    axis.set_title("PlaySplat Playability Summary", fontsize=16, pad=18)
    table_text = [[label, _format_card_value(value)] for label, value in rows]
    table = axis.table(
        cellText=table_text,
        colLabels=["metric", "value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8edf3")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def generate_scene_previews(scene_output: str | Path) -> dict[str, Path]:
    """Generate all available preview PNGs for a scene output directory."""

    scene_dir = Path(scene_output)
    preview_dir = scene_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    gaussian_path = _find_source_ply(scene_dir)
    if gaussian_path is not None and gaussian_path.exists():
        outputs["gaussian_points"] = render_point_cloud_preview(
            gaussian_path,
            preview_dir / "gaussian_points.png",
            title="Gaussian Points",
        )

    mesh_specs = (
        ("proxy_mesh", "proxy_mesh.obj", "Proxy Mesh"),
        ("floor_mesh", "floor_mesh.obj", "Floor Mesh"),
        ("wall_mesh", "wall_mesh.obj", "Wall Mesh"),
        ("obstacle_mesh", "obstacle_mesh.obj", "Obstacle Mesh"),
        ("walkable_mesh", "walkable_mesh.obj", "Walkable Mesh"),
    )
    for key, filename, title in mesh_specs:
        path = scene_dir / filename
        if path.exists():
            outputs[key] = render_mesh_preview(
                path,
                preview_dir / f"{key}.png",
                title=title,
            )

    report_path = scene_dir / "playability_report.json"
    if report_path.exists():
        outputs["playability_summary"] = render_playability_summary_card(
            report_path,
            preview_dir / "playability_summary.png",
        )
    return outputs


def _load_mesh(path: Path) -> Any:
    load_mesh: Any = trimesh.load
    loaded: Any = load_mesh(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError(f"Unable to load mesh geometry from: {path}")
    return mesh


def _deterministic_indices(count: int, target_count: int) -> NDArray[np.int64]:
    if count <= target_count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, target_count, dtype=np.int64)


def _style_3d_axis(axis: Any, points: NDArray[np.float32]) -> None:
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    if points.size == 0:
        return
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2.0
    radius = float(np.max(maximum - minimum) / 2.0)
    if radius <= 0.0:
        radius = 1.0
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.view_init(elev=28, azim=-45)
    axis.grid(True, alpha=0.3)


def _point_colors(colors: NDArray[np.float32] | None, expected_count: int) -> str | NDArray[np.float32]:
    if colors is None or colors.shape[0] < expected_count:
        return "#2f6fbb"
    color_values = colors[:expected_count]
    return np.clip(color_values, 0.0, 1.0)


def _find_source_ply(scene_dir: Path) -> Path | None:
    direct = scene_dir / "point_cloud.ply"
    if direct.exists():
        return direct
    for manifest in (
        scene_dir / "exports" / "unity" / "manifest.json",
        scene_dir / "exports" / "playcanvas" / "manifest.json",
        scene_dir / "exports" / "webgl" / "manifest.json",
    ):
        source = _source_from_manifest(manifest)
        if source is not None:
            return source
    return None


def _source_from_manifest(manifest_path: Path) -> Path | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    files = manifest.get("files", {})
    if isinstance(files, dict):
        bundled = files.get("visual_gaussian_ply")
        if isinstance(bundled, str):
            candidate = manifest_path.parent / bundled
            if candidate.exists():
                return candidate
    metadata = manifest.get("metadata", {})
    if isinstance(metadata, dict):
        source_path = metadata.get("source_path")
        if isinstance(source_path, str) and source_path:
            return Path(source_path)
    return None


def _format_card_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
