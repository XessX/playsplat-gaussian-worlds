"""Proxy geometry extraction and export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
import trimesh

from playsplat.types import ProxyGeometryLayer, ProxyMesh, VisualSplatLayer, VoxelOccupancyGrid


def extract_proxy_geometry(
    visual_layer: VisualSplatLayer,
    *,
    method: str = "placeholder_density_surface",
) -> ProxyGeometryLayer:
    """Extract proxy geometry from a visual splat layer."""

    return ProxyGeometryLayer(
        metadata=visual_layer.metadata,
        method=method,
        attributes={"status": "placeholder_proxy_geometry"},
    )


def extract_proxy_mesh(
    grid: VoxelOccupancyGrid,
    smooth_sigma: float = 0.0,
) -> ProxyMesh:
    """Extract a triangular proxy mesh from a voxel occupancy grid."""

    if smooth_sigma < 0.0:
        raise ValueError(f"smooth_sigma must be non-negative; got {smooth_sigma}.")
    if not np.any(grid.occupied):
        raise ValueError("Cannot extract proxy mesh because the occupancy grid is empty.")

    volume = grid.occupied.astype(np.float32)
    if smooth_sigma > 0.0:
        volume = gaussian_filter(volume, sigma=smooth_sigma).astype(np.float32, copy=False)

    padded_volume = np.pad(volume, pad_width=1, mode="constant", constant_values=0.0)
    volume_min = float(np.min(padded_volume))
    volume_max = float(np.max(padded_volume))
    if volume_max <= volume_min:
        raise ValueError("Cannot extract proxy mesh from a constant occupancy volume.")

    level = 0.5 if volume_min < 0.5 < volume_max else (volume_min + volume_max) / 2.0
    marching_cubes: Any = measure.marching_cubes
    vertices, faces, _normals, _values = marching_cubes(
        padded_volume,
        level=level,
        spacing=(grid.voxel_size, grid.voxel_size, grid.voxel_size),
    )
    world_vertices = (
        grid.origin + vertices.astype(np.float32, copy=False) - np.float32(grid.voxel_size)
    )
    mesh_metadata: dict[str, Any] = {
        "vertex_count": int(world_vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "method": "marching_cubes_occupancy",
        "smooth_sigma": smooth_sigma,
        "voxel_size": grid.voxel_size,
        "level": level,
    }
    return ProxyMesh(
        vertices=world_vertices.astype(np.float32, copy=False),
        faces=faces.astype(np.int32, copy=False),
        metadata=mesh_metadata,
    )


def export_proxy_mesh(mesh: ProxyMesh, output_path: str | Path) -> Path:
    """Export a proxy mesh as OBJ, PLY, or GLB using trimesh."""

    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix not in {".obj", ".ply", ".glb"}:
        raise ValueError(
            f"Unsupported proxy mesh export suffix '{suffix}'. Use .obj, .ply, or .glb."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    trimesh_mesh.export(path)
    return path
