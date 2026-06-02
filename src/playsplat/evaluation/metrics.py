"""Playability evaluation metrics and report helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from playsplat.types import ExportBundle, PlaySplatScene, ProxyMesh, SceneStructure


@dataclass(frozen=True)
class PlayabilityReport:
    """High-level playability readiness summary."""

    status: str
    collision_ready: bool
    navigation_ready: bool
    semantics_ready: bool
    affordances_ready: bool
    metrics: dict[str, Any]
    summary: dict[str, Any]
    warnings: list[str]
    notes: tuple[str, ...] = ()


def compute_playability_metrics(scene: PlaySplatScene) -> dict[str, Any]:
    """Compute JSON-serializable prototype playability metrics for a scene."""

    gaussian_count = scene.visual.gaussian_count
    structure_metadata = _structure_metadata(scene)
    total_structure_area = _float_metric(structure_metadata.get("total_area", 0.0))
    walkable_area = _float_metric(structure_metadata.get("walkable_area", 0.0))
    obstacle_area = _float_metric(structure_metadata.get("obstacle_area", 0.0))
    wall_area = _float_metric(structure_metadata.get("wall_area", 0.0))
    floor_area = _float_metric(structure_metadata.get("floor_area", 0.0))
    engine_exports = _engine_exports(scene)
    collision_mesh = scene.proxy_geometry.attributes.get("collision_mesh")
    collision_metadata = scene.proxy_geometry.attributes.get("collision_mesh_metadata", {})
    if not isinstance(collision_metadata, dict):
        collision_metadata = {}
    if isinstance(collision_mesh, ProxyMesh):
        collision_mesh_available = True
        collision_vertex_count = int(collision_mesh.vertices.shape[0])
        collision_face_count = int(collision_mesh.faces.shape[0])
    else:
        collision_mesh_available = False
        collision_vertex_count = 0
        collision_face_count = 0

    has_proxy_mesh = scene.proxy_geometry.attributes.get("proxy_mesh") is not None
    has_scene_structure = bool(structure_metadata)
    has_walkable_region = scene.navigation.walkable_region_count > 0
    semantic_status = str(scene.semantics.attributes.get("status", "unknown"))
    affordance_status = str(scene.affordances.attributes.get("status", "unknown"))
    geometry_semantics_available = semantic_status == "geometry_semantic_layer"
    geometry_affordances_available = affordance_status == "geometry_affordance_layer"
    semantics_ready = semantic_status != "placeholder_semantic_layer" and scene.semantics.label_count > 0
    affordances_ready = (
        affordance_status != "placeholder_affordance_layer"
        and scene.affordances.affordance_count > 0
    )

    metrics: dict[str, Any] = {
        "gaussian_count": gaussian_count,
        "has_visual_gaussian_layer": scene.visual.gaussians is not None,
        "source_path_present": scene.metadata.source_path is not None,
        "proxy_mesh_available": has_proxy_mesh,
        "proxy_vertex_count": scene.proxy_geometry.vertex_count,
        "proxy_face_count": scene.proxy_geometry.face_count,
        "proxy_mesh_count": scene.proxy_geometry.mesh_count,
        "proxy_face_to_gaussian_ratio": _ratio(scene.proxy_geometry.face_count, gaussian_count),
        "proxy_vertex_to_gaussian_ratio": _ratio(scene.proxy_geometry.vertex_count, gaussian_count),
        "collision_mesh_available": collision_mesh_available,
        "collision_vertex_count": collision_vertex_count,
        "collision_face_count": collision_face_count,
        "collision_face_reduction_ratio": _collision_reduction_ratio(
            collision_metadata,
            scene.proxy_geometry.face_count,
            collision_face_count,
        ),
        "collision_face_to_proxy_face_ratio": _ratio(
            collision_face_count,
            scene.proxy_geometry.face_count,
        ),
        "simplification_status": collision_metadata.get("status", "missing"),
        "semantic_status": semantic_status,
        "affordance_status": affordance_status,
        "geometry_semantics_available": geometry_semantics_available,
        "geometry_affordances_available": geometry_affordances_available,
        "semantic_label_count": scene.semantics.label_count,
        "affordance_label_count": scene.affordances.affordance_count,
        "floor_area": floor_area,
        "wall_area": wall_area,
        "obstacle_area": obstacle_area,
        "walkable_area": walkable_area,
        "total_structure_area": total_structure_area,
        "walkable_area_ratio": _ratio(walkable_area, total_structure_area),
        "obstacle_area_ratio": _ratio(obstacle_area, total_structure_area),
        "wall_area_ratio": _ratio(wall_area, total_structure_area),
        "floor_face_count": _int_metric(structure_metadata.get("floor_face_count", 0)),
        "wall_face_count": _int_metric(structure_metadata.get("wall_face_count", 0)),
        "obstacle_face_count": _int_metric(structure_metadata.get("obstacle_face_count", 0)),
        "walkable_face_count": _int_metric(structure_metadata.get("walkable_face_count", 0)),
        "walkable_region_count": scene.navigation.walkable_region_count,
        "navmesh_polygon_count": scene.navigation.navmesh_polygon_count,
        "agent_radius": scene.navigation.agent_radius,
        "navigation_status": scene.navigation.attributes.get("status", "unknown"),
        "walkable_area_from_navigation": scene.navigation.attributes.get("walkable_area"),
        "collider_count": scene.collision_physics.collider_count,
        "rigid_body_count": scene.collision_physics.rigid_body_count,
        "collision_mode": scene.collision_physics.mode,
        "collision_status": scene.collision_physics.attributes.get("status", "unknown"),
        "has_proxy_mesh": has_proxy_mesh,
        "has_floor_mesh": _region_available(scene, "floor"),
        "has_wall_mesh": _region_available(scene, "walls"),
        "has_obstacle_mesh": _region_available(scene, "obstacles"),
        "has_walkable_mesh": _region_available(scene, "walkable"),
        "has_scene_structure": has_scene_structure,
        "engine_export_target_count": len(engine_exports),
        "engine_export_targets": [bundle.target for bundle in engine_exports],
    }
    metrics["export_readiness_score"] = _export_readiness_score(metrics)
    metrics["overall_playability_score"] = _overall_playability_score(
        visual_present=metrics["has_visual_gaussian_layer"],
        proxy_present=has_proxy_mesh,
        collision_present=collision_mesh_available
        or scene.collision_physics.collider_count > 0
        or scene.proxy_geometry.mesh_count > 0,
        walkable_present=has_walkable_region,
        structure_present=has_scene_structure,
        export_possible=metrics["engine_export_target_count"] > 0,
        semantics_present=semantics_ready,
        affordances_present=affordances_ready,
    )
    return metrics


def evaluate_playability(scene: PlaySplatScene) -> PlayabilityReport:
    """Evaluate whether a layered scene is ready for interaction."""

    metrics = compute_playability_metrics(scene)
    warnings = _warnings_for_metrics(scene, metrics)
    collision_ready = bool(
        metrics["collision_mesh_available"] or metrics["collider_count"] > 0 or metrics["has_proxy_mesh"]
    )
    navigation_ready = bool(metrics["walkable_region_count"] > 0)
    semantics_ready = bool(metrics["semantic_status"] != "placeholder_semantic_layer")
    semantics_ready = semantics_ready and bool(metrics["semantic_label_count"] > 0)
    affordances_ready = bool(metrics["affordance_status"] != "placeholder_affordance_layer")
    affordances_ready = affordances_ready and bool(metrics["affordance_label_count"] > 0)
    score = _float_metric(metrics["overall_playability_score"])
    status = _status_from_score(score, metrics)
    summary = {
        "status": status,
        "overall_playability_score": score,
        "export_readiness_score": metrics["export_readiness_score"],
        "collision_ready": collision_ready,
        "navigation_ready": navigation_ready,
        "semantics_ready": semantics_ready,
        "affordances_ready": affordances_ready,
        "semantic_status": metrics["semantic_status"],
        "affordance_status": metrics["affordance_status"],
        "warning_count": len(warnings),
        "score_definition": (
            "Prototype layer-completeness score averaging visual, proxy, collision, "
            "walkable, structure, export, non-placeholder semantics, and "
            "non-placeholder affordance availability."
        ),
    }
    return PlayabilityReport(
        status=status,
        collision_ready=collision_ready,
        navigation_ready=navigation_ready,
        semantics_ready=semantics_ready,
        affordances_ready=affordances_ready,
        metrics=metrics,
        summary=summary,
        warnings=warnings,
        notes=(
            "This is a prototype completeness/readiness score, not a validated benchmark.",
            "Placeholder semantic and affordance layers are reported but do not count as ready.",
        ),
    )


def playability_report_to_dict(report: PlayabilityReport) -> dict[str, Any]:
    """Convert a playability report to a JSON-serializable dictionary."""

    return {
        "status": report.status,
        "summary": _json_safe(report.summary),
        "metrics": _json_safe(report.metrics),
        "warnings": list(report.warnings),
        "notes": list(report.notes),
    }


def write_playability_report(report: PlayabilityReport, output_path: str | Path) -> Path:
    """Write a playability report JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(playability_report_to_dict(report), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_playability_metrics_csv(report: PlayabilityReport, output_path: str | Path) -> Path:
    """Write scalar playability metrics as a key-value CSV file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in sorted(report.metrics.items()):
            if _is_scalar(value):
                writer.writerow([key, value])
        for key, value in sorted(report.summary.items()):
            if _is_scalar(value):
                writer.writerow([f"summary.{key}", value])
    return path


def _structure_metadata(scene: PlaySplatScene) -> dict[str, Any]:
    structure = scene.proxy_geometry.attributes.get("scene_structure")
    if isinstance(structure, SceneStructure):
        return structure.metadata
    metadata = scene.proxy_geometry.attributes.get("structure_metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def _engine_exports(scene: PlaySplatScene) -> tuple[ExportBundle, ...]:
    exports = scene.proxy_geometry.attributes.get("engine_exports", ())
    if not isinstance(exports, tuple):
        return ()
    return tuple(bundle for bundle in exports if isinstance(bundle, ExportBundle))


def _region_available(scene: PlaySplatScene, attribute_name: str) -> bool:
    structure = scene.proxy_geometry.attributes.get("scene_structure")
    if not isinstance(structure, SceneStructure):
        return False
    region = getattr(structure, attribute_name)
    return region is not None and region.face_count > 0


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    if denominator == 0:
        return None
    return _float_metric(float(numerator) / float(denominator))


def _export_readiness_score(metrics: dict[str, Any]) -> float:
    checks = (
        bool(metrics["has_proxy_mesh"]),
        bool(metrics["collision_mesh_available"]),
        bool(metrics["has_floor_mesh"] or metrics["has_walkable_mesh"]),
        bool(metrics["has_wall_mesh"] or metrics["has_obstacle_mesh"]),
        bool(metrics["has_scene_structure"]),
        bool(metrics["engine_export_target_count"] > 0),
    )
    return _float_metric(sum(1 for check in checks if check) / len(checks))


def _overall_playability_score(
    *,
    visual_present: bool,
    proxy_present: bool,
    collision_present: bool,
    walkable_present: bool,
    structure_present: bool,
    export_possible: bool,
    semantics_present: bool,
    affordances_present: bool,
) -> float:
    checks = (
        visual_present,
        proxy_present,
        collision_present,
        walkable_present,
        structure_present,
        export_possible,
        semantics_present,
        affordances_present,
    )
    return _float_metric(sum(1 for check in checks if check) / len(checks))


def _warnings_for_metrics(scene: PlaySplatScene, metrics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not metrics["has_visual_gaussian_layer"]:
        warnings.append("missing visual Gaussian layer")
    if not metrics["has_proxy_mesh"]:
        warnings.append("missing proxy mesh")
    if metrics["has_proxy_mesh"] and not metrics["collision_mesh_available"]:
        warnings.append("missing collision mesh")
    if metrics["walkable_region_count"] == 0:
        warnings.append("missing walkable region")
    if not metrics["has_scene_structure"]:
        warnings.append("missing structure detection")
    if metrics["semantic_status"] == "placeholder_semantic_layer":
        warnings.append("semantics are placeholder")
    if metrics["affordance_status"] == "placeholder_affordance_layer":
        warnings.append("affordances are placeholder")
    return warnings


def _status_from_score(score: float, metrics: dict[str, Any]) -> str:
    if metrics["gaussian_count"] == 0 and not metrics["has_proxy_mesh"]:
        return "placeholder"
    if score >= 0.75:
        return "ready_prototype"
    if score >= 0.4:
        return "partial"
    return "incomplete"


def _int_metric(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def _float_metric(value: Any) -> float:
    if value is None:
        return 0.0
    return round(float(value), 7)


def _collision_reduction_ratio(
    collision_metadata: dict[str, Any],
    proxy_face_count: int,
    collision_face_count: int,
) -> float:
    metadata_ratio = collision_metadata.get("achieved_reduction_ratio")
    if metadata_ratio is not None:
        return _float_metric(metadata_ratio)
    if proxy_face_count <= 0:
        return 0.0
    return _float_metric(1.0 - (float(collision_face_count) / float(proxy_face_count)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))
