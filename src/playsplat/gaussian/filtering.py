"""Gaussian filtering utilities for geometry extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from playsplat.types import FilteredGaussianLayer, GaussianLayer


def filter_gaussians_for_geometry(
    layer: GaussianLayer,
    opacity_threshold: float = 0.01,
    bounds_quantile: float = 0.995,
    max_gaussians: int | None = None,
) -> FilteredGaussianLayer:
    """Filter loaded Gaussians into a deterministic geometry-ready subset."""

    if opacity_threshold < 0.0:
        raise ValueError(f"opacity_threshold must be non-negative; got {opacity_threshold}.")
    if not 0.0 < bounds_quantile <= 1.0:
        raise ValueError(f"bounds_quantile must be in (0, 1]; got {bounds_quantile}.")
    if max_gaussians is not None and max_gaussians <= 0:
        raise ValueError(f"max_gaussians must be positive or None; got {max_gaussians}.")

    original_count = layer.gaussian_count
    source_indices: NDArray[np.int64] = np.arange(original_count, dtype=np.int64)
    positions: NDArray[np.float32] = layer.positions
    opacity: NDArray[np.float32] | None = layer.opacity
    scales: NDArray[np.float32] | None = layer.scales

    valid_mask = np.all(np.isfinite(positions), axis=1)
    removed_invalid_count = int(original_count - np.count_nonzero(valid_mask))
    positions, opacity, scales, source_indices = _apply_mask(
        valid_mask,
        positions,
        opacity,
        scales,
        source_indices,
    )

    removed_low_opacity_count = 0
    if opacity is not None:
        opacity_mask = np.isfinite(opacity) & (opacity >= opacity_threshold)
        removed_low_opacity_count = int(opacity.shape[0] - np.count_nonzero(opacity_mask))
        positions, opacity, scales, source_indices = _apply_mask(
            opacity_mask,
            positions,
            opacity,
            scales,
            source_indices,
        )

    removed_outlier_count = 0
    if positions.shape[0] >= 10 and bounds_quantile < 1.0:
        lower_quantile = 1.0 - bounds_quantile
        lower = np.quantile(positions, lower_quantile, axis=0)
        upper = np.quantile(positions, bounds_quantile, axis=0)
        outlier_mask = np.all((positions >= lower) & (positions <= upper), axis=1)
        removed_outlier_count = int(positions.shape[0] - np.count_nonzero(outlier_mask))
        positions, opacity, scales, source_indices = _apply_mask(
            outlier_mask,
            positions,
            opacity,
            scales,
            source_indices,
        )

    downsampled_count = 0
    if max_gaussians is not None and positions.shape[0] > max_gaussians:
        selected = np.linspace(0, positions.shape[0] - 1, max_gaussians, dtype=np.int64)
        downsampled_count = int(positions.shape[0] - max_gaussians)
        positions = positions[selected]
        source_indices = source_indices[selected]
        opacity = opacity[selected] if opacity is not None else None
        scales = scales[selected] if scales is not None else None

    kept_count = int(positions.shape[0])
    removed_count = original_count - kept_count
    metadata: dict[str, Any] = {
        "original_count": original_count,
        "kept_count": kept_count,
        "removed_invalid_count": removed_invalid_count,
        "removed_low_opacity_count": removed_low_opacity_count,
        "removed_outlier_count": removed_outlier_count,
        "downsampled_count": downsampled_count,
        "opacity_threshold": opacity_threshold,
        "bounds_quantile": bounds_quantile,
        "max_gaussians": max_gaussians,
    }

    return FilteredGaussianLayer(
        positions=positions.astype(np.float32, copy=False),
        opacity=opacity.astype(np.float32, copy=False) if opacity is not None else None,
        scales=scales.astype(np.float32, copy=False) if scales is not None else None,
        source_indices=source_indices.astype(np.int64, copy=False),
        removed_count=removed_count,
        filter_metadata=metadata,
    )


def _apply_mask(
    mask: NDArray[np.bool_],
    positions: NDArray[np.float32],
    opacity: NDArray[np.float32] | None,
    scales: NDArray[np.float32] | None,
    source_indices: NDArray[np.int64],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32] | None,
    NDArray[np.float32] | None,
    NDArray[np.int64],
]:
    return (
        positions[mask],
        opacity[mask] if opacity is not None else None,
        scales[mask] if scales is not None else None,
        source_indices[mask],
    )
