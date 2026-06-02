"""Collision-oriented proxy mesh simplification helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import trimesh

from playsplat.types import ProxyMesh


def simplify_proxy_mesh(
    mesh: ProxyMesh,
    target_face_count: int = 50_000,
    method: str = "vertex_clustering",
    clustering_voxel_size: float | None = None,
    max_iterations: int = 8,
) -> ProxyMesh:
    """Simplify a proxy mesh into a deterministic collision-ready mesh."""

    if target_face_count <= 0:
        raise ValueError(f"target_face_count must be positive; got {target_face_count}.")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive; got {max_iterations}.")

    normalized_method = method.strip().lower()
    if normalized_method != "vertex_clustering":
        raise ValueError(
            f"Unsupported simplification method '{method}'. Use 'vertex_clustering'."
        )

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    original_vertex_count = int(vertices.shape[0])
    original_face_count = int(faces.shape[0])

    if original_face_count <= target_face_count:
        return ProxyMesh(
            vertices=vertices.copy(),
            faces=faces.copy(),
            metadata=_simplification_metadata(
                mesh=mesh,
                original_vertex_count=original_vertex_count,
                original_face_count=original_face_count,
                simplified_vertex_count=original_vertex_count,
                simplified_face_count=original_face_count,
                target_face_count=target_face_count,
                method=normalized_method,
                clustering_voxel_size=clustering_voxel_size,
                iterations=0,
                status="already_within_target",
            ),
        )

    voxel_size = _initial_clustering_voxel_size(vertices, clustering_voxel_size)
    simplified_vertices = vertices.copy()
    simplified_faces = faces.copy()
    status = "max_iterations_reached"
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        simplified_vertices, simplified_faces = _cluster_vertices(
            vertices,
            faces,
            voxel_size,
        )
        if int(simplified_faces.shape[0]) <= target_face_count:
            status = "target_reached"
            break
        voxel_size *= 2.0

    return ProxyMesh(
        vertices=simplified_vertices.astype(np.float32, copy=False),
        faces=simplified_faces.astype(np.int32, copy=False),
        metadata=_simplification_metadata(
            mesh=mesh,
            original_vertex_count=original_vertex_count,
            original_face_count=original_face_count,
            simplified_vertex_count=int(simplified_vertices.shape[0]),
            simplified_face_count=int(simplified_faces.shape[0]),
            target_face_count=target_face_count,
            method=normalized_method,
            clustering_voxel_size=voxel_size,
            iterations=iterations,
            status=status,
        ),
    )


def export_collision_mesh(mesh: ProxyMesh, output_path: str | Path) -> Path:
    """Export a collision mesh as OBJ, PLY, or GLB using trimesh."""

    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix not in {".obj", ".ply", ".glb"}:
        raise ValueError(
            f"Unsupported collision mesh export suffix '{suffix}'. Use .obj, .ply, or .glb."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    trimesh_mesh.export(path)
    return path


def _cluster_vertices(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int32],
    voxel_size: float,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    if vertices.size == 0 or faces.size == 0:
        return vertices.copy(), faces.copy()

    anchor = np.min(vertices, axis=0)
    quantized = np.floor((vertices - anchor) / np.float32(voxel_size)).astype(np.int64)
    unique_result = np.unique(quantized, axis=0, return_inverse=True)
    unique_cells = cast(NDArray[np.int64], unique_result[0])
    inverse = cast(NDArray[np.int64], unique_result[1])
    cluster_count = int(unique_cells.shape[0])

    sums = np.zeros((cluster_count, 3), dtype=np.float64)
    np.add.at(sums, inverse, vertices.astype(np.float64))
    counts = np.bincount(inverse, minlength=cluster_count).astype(np.float64)
    clustered_vertices = (sums / counts[:, np.newaxis]).astype(np.float32)

    valid_faces = _remove_invalid_faces(faces, int(vertices.shape[0]))
    if valid_faces.size == 0:
        return _compact_vertices(clustered_vertices, valid_faces)
    remapped_faces = inverse[valid_faces.astype(np.int64, copy=False)].astype(
        np.int32,
        copy=False,
    )
    remapped_faces = _remove_invalid_faces(remapped_faces, int(clustered_vertices.shape[0]))
    remapped_faces = _remove_degenerate_faces(remapped_faces)
    remapped_faces = _remove_duplicate_faces(remapped_faces)
    return _compact_vertices(clustered_vertices, remapped_faces)


def _remove_invalid_faces(
    faces: NDArray[np.int32],
    vertex_count: int,
) -> NDArray[np.int32]:
    if faces.size == 0:
        return faces.reshape((0, 3)).astype(np.int32, copy=False)
    valid = np.logical_and(
        np.all(faces >= 0, axis=1),
        np.all(faces < vertex_count, axis=1),
    )
    return cast(NDArray[np.int32], faces[valid].astype(np.int32, copy=False))


def _remove_degenerate_faces(faces: NDArray[np.int32]) -> NDArray[np.int32]:
    if faces.size == 0:
        return faces.reshape((0, 3)).astype(np.int32, copy=False)
    valid = np.logical_and.reduce(
        (
            faces[:, 0] != faces[:, 1],
            faces[:, 1] != faces[:, 2],
            faces[:, 0] != faces[:, 2],
        )
    )
    return cast(NDArray[np.int32], faces[valid].astype(np.int32, copy=False))


def _remove_duplicate_faces(faces: NDArray[np.int32]) -> NDArray[np.int32]:
    if faces.size == 0:
        return faces.reshape((0, 3)).astype(np.int32, copy=False)
    canonical_faces = np.sort(faces, axis=1)
    _unique_faces, unique_indices = np.unique(canonical_faces, axis=0, return_index=True)
    sorted_indices = np.sort(unique_indices)
    return faces[sorted_indices].astype(np.int32, copy=False)


def _compact_vertices(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int32],
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    if faces.size == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
        )
    used_vertices = np.unique(faces.reshape(-1))
    index_map = np.full(vertices.shape[0], -1, dtype=np.int64)
    index_map[used_vertices] = np.arange(used_vertices.shape[0], dtype=np.int64)
    compact_faces = index_map[faces].astype(np.int32, copy=False)
    compact_vertices = vertices[used_vertices].astype(np.float32, copy=False)
    return compact_vertices, compact_faces


def _initial_clustering_voxel_size(
    vertices: NDArray[np.float32],
    requested_voxel_size: float | None,
) -> float:
    if requested_voxel_size is not None:
        if requested_voxel_size <= 0.0:
            raise ValueError(
                "clustering_voxel_size must be positive when provided; "
                f"got {requested_voxel_size}."
            )
        return float(requested_voxel_size)
    if vertices.size == 0:
        return 1.0
    bounds = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    diagonal = float(np.linalg.norm(bounds))
    if diagonal <= 0.0:
        return 1.0
    return max(diagonal / 128.0, float(np.finfo(np.float32).eps))


def _simplification_metadata(
    *,
    mesh: ProxyMesh,
    original_vertex_count: int,
    original_face_count: int,
    simplified_vertex_count: int,
    simplified_face_count: int,
    target_face_count: int,
    method: str,
    clustering_voxel_size: float | None,
    iterations: int,
    status: str,
) -> dict[str, Any]:
    return {
        "source_proxy_metadata": dict(mesh.metadata),
        "original_vertex_count": original_vertex_count,
        "original_face_count": original_face_count,
        "simplified_vertex_count": simplified_vertex_count,
        "simplified_face_count": simplified_face_count,
        "target_face_count": target_face_count,
        "method": method,
        "clustering_voxel_size": clustering_voxel_size,
        "achieved_reduction_ratio": _reduction_ratio(
            original_face_count,
            simplified_face_count,
        ),
        "iterations": iterations,
        "status": status,
    }


def _reduction_ratio(original_face_count: int, simplified_face_count: int) -> float:
    if original_face_count <= 0:
        return 0.0
    return round(1.0 - (float(simplified_face_count) / float(original_face_count)), 7)
