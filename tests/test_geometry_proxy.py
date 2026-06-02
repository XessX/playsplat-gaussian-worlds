from __future__ import annotations

from pathlib import Path

import numpy as np

from playsplat.gaussian.filtering import filter_gaussians_for_geometry
from playsplat.geometry.occupancy import build_voxel_occupancy
from playsplat.geometry.proxy import export_proxy_mesh, extract_proxy_mesh
from playsplat.types import FilteredGaussianLayer, GaussianLayer, VoxelOccupancyGrid


def test_filter_gaussians_removes_invalid_positions_and_low_opacity() -> None:
    layer = GaussianLayer(
        positions=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [np.nan, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        opacity=np.asarray([0.5, 0.001, 0.8, 0.9], dtype=np.float32),
        scales=np.asarray(
            [
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
            ],
            dtype=np.float32,
        ),
    )

    filtered = filter_gaussians_for_geometry(
        layer,
        opacity_threshold=0.01,
        bounds_quantile=1.0,
    )

    assert filtered.gaussian_count == 2
    assert filtered.removed_count == 2
    assert filtered.source_indices.tolist() == [0, 3]
    assert filtered.filter_metadata["removed_invalid_count"] == 1
    assert filtered.filter_metadata["removed_low_opacity_count"] == 1


def test_voxel_occupancy_creates_non_empty_grid_from_tiny_cloud() -> None:
    filtered = FilteredGaussianLayer(
        positions=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float32,
        ),
        opacity=None,
        scales=None,
        source_indices=np.asarray([0, 1, 2], dtype=np.int64),
        removed_count=0,
    )

    grid = build_voxel_occupancy(
        filtered,
        voxel_size=0.05,
        density_threshold=1.0,
        padding_voxels=1,
    )

    assert grid.occupied.any()
    assert grid.metadata["occupied_voxel_count"] == 3
    assert grid.grid_shape == grid.density.shape


def test_proxy_mesh_extraction_returns_vertices_and_faces_for_cube() -> None:
    grid = _cube_grid()

    mesh = extract_proxy_mesh(grid)

    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
    assert mesh.metadata["method"] == "marching_cubes_occupancy"


def test_export_proxy_mesh_writes_mesh_file(tmp_path: Path) -> None:
    mesh = extract_proxy_mesh(_cube_grid())
    output_path = tmp_path / "proxy_mesh.obj"

    exported = export_proxy_mesh(mesh, output_path)

    assert exported == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _cube_grid() -> VoxelOccupancyGrid:
    occupied = np.zeros((6, 6, 6), dtype=np.bool_)
    occupied[2:4, 2:4, 2:4] = True
    density = occupied.astype(np.float32)
    return VoxelOccupancyGrid(
        origin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        voxel_size=0.1,
        density=density,
        occupied=occupied,
        grid_shape=(6, 6, 6),
    )
