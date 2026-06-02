"""Voxel occupancy construction from filtered Gaussian splats."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from playsplat.types import FilteredGaussianLayer, VoxelOccupancyGrid


def build_voxel_occupancy(
    filtered: FilteredGaussianLayer,
    voxel_size: float = 0.05,
    density_threshold: float = 1.0,
    padding_voxels: int = 2,
    max_grid_voxels: int = 20_000_000,
) -> VoxelOccupancyGrid:
    """Build a coarse density and occupancy grid from filtered Gaussian centers."""

    if filtered.gaussian_count == 0:
        raise ValueError("Cannot build voxel occupancy from an empty filtered Gaussian layer.")
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be positive; got {voxel_size}.")
    if density_threshold <= 0.0:
        raise ValueError(f"density_threshold must be positive; got {density_threshold}.")
    if padding_voxels < 0:
        raise ValueError(f"padding_voxels must be non-negative; got {padding_voxels}.")
    if max_grid_voxels <= 0:
        raise ValueError(f"max_grid_voxels must be positive; got {max_grid_voxels}.")

    positions = filtered.positions
    minimum = np.min(positions, axis=0)
    maximum = np.max(positions, axis=0)
    origin = (minimum - padding_voxels * voxel_size).astype(np.float32)
    padded_maximum = maximum + padding_voxels * voxel_size
    grid_shape_array = np.ceil((padded_maximum - origin) / voxel_size).astype(np.int64) + 1
    grid_shape = (
        int(grid_shape_array[0]),
        int(grid_shape_array[1]),
        int(grid_shape_array[2]),
    )
    total_voxels = int(np.prod(grid_shape_array))

    if total_voxels > max_grid_voxels:
        raise ValueError(
            "Estimated voxel grid is too large: "
            f"{total_voxels:,} voxels for shape {grid_shape}. "
            f"Increase voxel_size above {voxel_size} or raise max_grid_voxels."
        )

    density = np.zeros(grid_shape, dtype=np.float32)
    voxel_indices = np.floor((positions - origin) / voxel_size).astype(np.int64)
    voxel_indices = np.clip(voxel_indices, 0, grid_shape_array - 1)

    if filtered.scales is None:
        np.add.at(
            density,
            (voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]),
            1.0,
        )
    else:
        _rasterize_with_scale_spread(density, voxel_indices, filtered.scales, voxel_size)

    occupied = density >= density_threshold
    occupied_voxel_count = int(np.count_nonzero(occupied))
    metadata: dict[str, Any] = {
        "voxel_size": voxel_size,
        "density_threshold": density_threshold,
        "padding_voxels": padding_voxels,
        "num_input_gaussians": filtered.gaussian_count,
        "occupied_voxel_count": occupied_voxel_count,
        "occupancy_ratio": float(occupied_voxel_count / total_voxels),
        "grid_shape": grid_shape,
        "max_grid_voxels": max_grid_voxels,
    }

    return VoxelOccupancyGrid(
        origin=origin,
        voxel_size=voxel_size,
        density=density,
        occupied=occupied,
        grid_shape=grid_shape,
        metadata=metadata,
    )


def _rasterize_with_scale_spread(
    density: NDArray[np.float32],
    voxel_indices: NDArray[np.int64],
    scales: NDArray[np.float32],
    voxel_size: float,
) -> None:
    shape = np.asarray(density.shape, dtype=np.int64)
    for center, scale in zip(voxel_indices, scales, strict=True):
        radius_voxels = _scale_radius_voxels(scale, voxel_size)
        lower = np.maximum(center - radius_voxels, 0)
        upper = np.minimum(center + radius_voxels + 1, shape)
        density[
            lower[0] : upper[0],
            lower[1] : upper[1],
            lower[2] : upper[2],
        ] += 1.0


def _scale_radius_voxels(scale: NDArray[np.float32], voxel_size: float) -> int:
    finite_scale = scale[np.isfinite(scale)]
    if finite_scale.size == 0:
        return 0
    if np.any(finite_scale < 0.0):
        radii = np.exp(np.clip(finite_scale, -10.0, 10.0))
    else:
        radii = finite_scale
    radius = float(np.max(np.abs(radii)))
    if radius <= 0.0:
        return 0
    return max(1, int(np.ceil(radius / voxel_size)))
