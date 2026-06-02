"""Geometric scene-structure classification from proxy meshes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import trimesh

from playsplat.types import ProxyMesh, SceneStructure, SurfaceRegion


def classify_proxy_mesh_structure(
    mesh: ProxyMesh,
    up_axis: str = "y",
    max_floor_slope_degrees: float = 35.0,
    floor_height_quantile: float = 0.20,
    floor_height_tolerance: float = 0.25,
    wall_normal_tolerance: float = 0.35,
    min_region_area: float = 0.01,
) -> SceneStructure:
    """Classify proxy mesh faces into floor, wall, obstacle, and walkable regions."""

    if mesh.faces.shape[0] == 0:
        return SceneStructure(
            floor=None,
            walls=None,
            obstacles=None,
            walkable=None,
            metadata=_empty_metadata(
                up_axis=up_axis,
                max_floor_slope_degrees=max_floor_slope_degrees,
                floor_height_quantile=floor_height_quantile,
                floor_height_tolerance=floor_height_tolerance,
            ),
        )
    _validate_structure_parameters(
        max_floor_slope_degrees=max_floor_slope_degrees,
        floor_height_quantile=floor_height_quantile,
        floor_height_tolerance=floor_height_tolerance,
        wall_normal_tolerance=wall_normal_tolerance,
        min_region_area=min_region_area,
    )

    trimesh_mesh = _to_trimesh(mesh)
    normals = np.asarray(trimesh_mesh.face_normals, dtype=np.float32)
    centers = np.asarray(trimesh_mesh.triangles_center, dtype=np.float32)
    face_areas = np.asarray(trimesh_mesh.area_faces, dtype=np.float32)

    up_vector = _up_vector(up_axis)
    height_values = centers @ up_vector
    floor_base_height = float(np.quantile(height_values, floor_height_quantile))
    floor_height_limit = floor_base_height + floor_height_tolerance
    normal_up_alignment = normals @ up_vector
    floor_alignment_threshold = float(np.cos(np.deg2rad(max_floor_slope_degrees)))

    floor_mask = (normal_up_alignment >= floor_alignment_threshold) & (
        height_values <= floor_height_limit
    )
    wall_mask = np.abs(normal_up_alignment) <= wall_normal_tolerance
    obstacle_mask = ~(floor_mask | wall_mask)
    walkable_mask = floor_mask.copy()

    floor = _make_region("floor", floor_mask, face_areas, min_region_area)
    walls = _make_region("wall", wall_mask, face_areas, min_region_area)
    obstacles = _make_region("obstacle", obstacle_mask, face_areas, min_region_area)
    walkable = _make_region("walkable", walkable_mask, face_areas, min_region_area)

    metadata = {
        "total_faces": int(mesh.faces.shape[0]),
        "floor_face_count": _face_count(floor),
        "wall_face_count": _face_count(walls),
        "obstacle_face_count": _face_count(obstacles),
        "walkable_face_count": _face_count(walkable),
        "total_area": _clean_float(np.sum(face_areas)),
        "floor_area": _area(floor),
        "wall_area": _area(walls),
        "obstacle_area": _area(obstacles),
        "walkable_area": _area(walkable),
        "up_axis": up_axis,
        "max_floor_slope_degrees": max_floor_slope_degrees,
        "floor_height_quantile": floor_height_quantile,
        "floor_height_tolerance": floor_height_tolerance,
        "wall_normal_tolerance": wall_normal_tolerance,
        "min_region_area": min_region_area,
        "floor_base_height": _clean_float(floor_base_height),
        "floor_height_limit": _clean_float(floor_height_limit),
    }

    return SceneStructure(
        floor=floor,
        walls=walls,
        obstacles=obstacles,
        walkable=walkable,
        metadata=metadata,
    )


def export_structure_meshes(
    mesh: ProxyMesh,
    structure: SceneStructure,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export separate OBJ meshes for detected scene-structure regions."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    exports: dict[str, Path] = {}
    region_specs = (
        ("floor", structure.floor, "floor_mesh.obj"),
        ("wall", structure.walls, "wall_mesh.obj"),
        ("obstacle", structure.obstacles, "obstacle_mesh.obj"),
        ("walkable", structure.walkable, "walkable_mesh.obj"),
    )
    for key, region, filename in region_specs:
        if region is None or region.face_count == 0:
            continue
        path = directory / filename
        _export_region_mesh(mesh, region, path)
        exports[key] = path
    return exports


def scene_structure_to_dict(structure: SceneStructure) -> dict[str, Any]:
    """Convert scene structure into a JSON-friendly summary."""

    return {
        "metadata": structure.metadata,
        "regions": {
            "floor": _region_to_dict(structure.floor),
            "wall": _region_to_dict(structure.walls),
            "obstacle": _region_to_dict(structure.obstacles),
            "walkable": _region_to_dict(structure.walkable),
        },
    }


def _validate_structure_parameters(
    *,
    max_floor_slope_degrees: float,
    floor_height_quantile: float,
    floor_height_tolerance: float,
    wall_normal_tolerance: float,
    min_region_area: float,
) -> None:
    if not 0.0 <= max_floor_slope_degrees <= 90.0:
        raise ValueError(
            "max_floor_slope_degrees must be in [0, 90]; "
            f"got {max_floor_slope_degrees}."
        )
    if not 0.0 <= floor_height_quantile <= 1.0:
        raise ValueError(
            "floor_height_quantile must be in [0, 1]; "
            f"got {floor_height_quantile}."
        )
    if floor_height_tolerance < 0.0:
        raise ValueError(
            f"floor_height_tolerance must be non-negative; got {floor_height_tolerance}."
        )
    if not 0.0 <= wall_normal_tolerance <= 1.0:
        raise ValueError(
            "wall_normal_tolerance must be in [0, 1]; "
            f"got {wall_normal_tolerance}."
        )
    if min_region_area < 0.0:
        raise ValueError(f"min_region_area must be non-negative; got {min_region_area}.")


def _to_trimesh(mesh: ProxyMesh) -> Any:
    trimesh_ctor: Any = trimesh.Trimesh
    return trimesh_ctor(vertices=mesh.vertices, faces=mesh.faces, process=False)


def _up_vector(up_axis: str) -> NDArray[np.float32]:
    normalized = up_axis.strip().lower()
    vectors = {
        "x": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "+x": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "-x": np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "+y": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "-y": np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
        "z": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "+z": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "-z": np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
    }
    try:
        return vectors[normalized]
    except KeyError as exc:
        raise ValueError("up_axis must be one of x, y, z, +x, +y, +z, -x, -y, -z.") from exc


def _make_region(
    label: str,
    mask: NDArray[np.bool_],
    face_areas: NDArray[np.float32],
    min_region_area: float,
) -> SurfaceRegion | None:
    indices = np.flatnonzero(mask).astype(np.int64, copy=False)
    area = _clean_float(np.sum(face_areas[indices])) if indices.size > 0 else 0.0
    if indices.size == 0 or area < min_region_area:
        return None
    return SurfaceRegion(
        label=label,
        face_indices=indices,
        area=area,
        metadata={"face_count": int(indices.size)},
    )


def _export_region_mesh(mesh: ProxyMesh, region: SurfaceRegion, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faces = mesh.faces[region.face_indices]
    trimesh_ctor: Any = trimesh.Trimesh
    region_mesh = trimesh_ctor(vertices=mesh.vertices, faces=faces, process=False)
    region_mesh.export(output_path)


def _region_to_dict(region: SurfaceRegion | None) -> dict[str, Any] | None:
    if region is None:
        return None
    return {
        "label": region.label,
        "face_count": region.face_count,
        "area": region.area,
        "face_indices": region.face_indices.tolist(),
        "metadata": region.metadata,
    }


def _empty_metadata(
    *,
    up_axis: str,
    max_floor_slope_degrees: float,
    floor_height_quantile: float,
    floor_height_tolerance: float,
) -> dict[str, Any]:
    return {
        "total_faces": 0,
        "floor_face_count": 0,
        "wall_face_count": 0,
        "obstacle_face_count": 0,
        "walkable_face_count": 0,
        "total_area": 0.0,
        "floor_area": 0.0,
        "wall_area": 0.0,
        "obstacle_area": 0.0,
        "walkable_area": 0.0,
        "up_axis": up_axis,
        "max_floor_slope_degrees": max_floor_slope_degrees,
        "floor_height_quantile": floor_height_quantile,
        "floor_height_tolerance": floor_height_tolerance,
    }


def _face_count(region: SurfaceRegion | None) -> int:
    return 0 if region is None else region.face_count


def _area(region: SurfaceRegion | None) -> float:
    return 0.0 if region is None else region.area


def _clean_float(value: float | np.floating[Any]) -> float:
    return round(float(value), 7)
