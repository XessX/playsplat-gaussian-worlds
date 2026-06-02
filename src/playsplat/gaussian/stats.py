"""Statistics helpers for Gaussian splatting layers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from playsplat.types import GaussianLayer


def compute_gaussian_stats(layer: GaussianLayer) -> dict[str, Any]:
    """Compute JSON-serializable summary statistics for a Gaussian layer."""

    positions = layer.positions
    bounding_box = _bounding_box(positions)
    memory_bytes = _estimated_memory_bytes(layer)

    stats: dict[str, Any] = {
        "num_gaussians": layer.gaussian_count,
        "bounding_box": bounding_box,
        "scene_center": _scene_center(bounding_box),
        "scene_size": _scene_size(bounding_box),
        "opacity": _vector_summary(layer.opacity),
        "scales": _matrix_summary(layer.scales),
        "color_field_count": _color_field_count(layer),
        "estimated_memory_footprint": {
            "bytes": memory_bytes,
            "megabytes": round(memory_bytes / (1024 * 1024), 6),
        },
    }
    return stats


def _bounding_box(positions: NDArray[np.float32]) -> dict[str, list[float]]:
    if positions.shape[0] == 0:
        return {"min": [], "max": []}
    return {
        "min": _float_list(np.min(positions, axis=0)),
        "max": _float_list(np.max(positions, axis=0)),
    }


def _scene_center(bounding_box: dict[str, list[float]]) -> list[float]:
    if not bounding_box["min"] or not bounding_box["max"]:
        return []
    minimum = np.asarray(bounding_box["min"], dtype=np.float32)
    maximum = np.asarray(bounding_box["max"], dtype=np.float32)
    return _float_list((minimum + maximum) / 2.0)


def _scene_size(bounding_box: dict[str, list[float]]) -> list[float]:
    if not bounding_box["min"] or not bounding_box["max"]:
        return []
    minimum = np.asarray(bounding_box["min"], dtype=np.float32)
    maximum = np.asarray(bounding_box["max"], dtype=np.float32)
    return _float_list(maximum - minimum)


def _vector_summary(values: NDArray[np.float32] | None) -> dict[str, float] | None:
    if values is None or values.size == 0:
        return None
    return {
        "min": _clean_float(np.min(values)),
        "max": _clean_float(np.max(values)),
        "mean": _clean_float(np.mean(values)),
    }


def _matrix_summary(values: NDArray[np.float32] | None) -> dict[str, list[float]] | None:
    if values is None or values.size == 0:
        return None
    return {
        "min": _float_list(np.min(values, axis=0)),
        "max": _float_list(np.max(values, axis=0)),
        "mean": _float_list(np.mean(values, axis=0)),
    }


def _color_field_count(layer: GaussianLayer) -> int:
    field_names = tuple(str(field) for field in layer.metadata.get("field_names", ()))
    if field_names:
        return sum(
            1
            for field in field_names
            if field in {"red", "green", "blue"}
            or field.startswith("f_dc_")
            or field.startswith("f_rest_")
        )

    color_fields = 0
    if layer.colors is not None:
        color_fields += int(layer.colors.shape[1])
    if layer.features_dc is not None and layer.color_format != "dc":
        color_fields += int(layer.features_dc.shape[1])
    if layer.features_rest is not None:
        color_fields += int(layer.features_rest.shape[1])
    return color_fields


def _estimated_memory_bytes(layer: GaussianLayer) -> int:
    arrays = (
        layer.positions,
        layer.opacity,
        layer.scales,
        layer.rotations,
        layer.colors,
        layer.features_dc,
        layer.features_rest,
    )
    return int(sum(array.nbytes for array in arrays if array is not None))


def _float_list(values: NDArray[np.float32]) -> list[float]:
    return [_clean_float(value) for value in values.tolist()]


def _clean_float(value: float | np.floating[Any]) -> float:
    return round(float(value), 7)
