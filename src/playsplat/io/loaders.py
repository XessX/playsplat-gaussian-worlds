"""Scene loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from plyfile import PlyData

from playsplat.types import GaussianLayer, GaussianSplatScene, SceneMetadata


XYZ_FIELDS = ("x", "y", "z")
SCALE_FIELDS = ("scale_0", "scale_1", "scale_2")
ROTATION_FIELDS = ("rot_0", "rot_1", "rot_2", "rot_3")
DC_FIELDS = ("f_dc_0", "f_dc_1", "f_dc_2")
RGB_FIELDS = ("red", "green", "blue")


def load_gaussian_scene(input_path: Path | None, scene_id: str) -> GaussianSplatScene:
    """Load a Gaussian splatting scene.

    PLY inputs are parsed into a structured :class:`GaussianLayer`. Other
    formats intentionally remain placeholders until later research milestones.
    """

    gaussian_layer: GaussianLayer | None = None
    gaussian_count = 0
    notes = "Placeholder Gaussian scene. No file has been parsed yet."

    if input_path is not None and input_path.suffix.lower() == ".ply":
        gaussian_layer = load_gaussian_ply(input_path)
        gaussian_count = gaussian_layer.gaussian_count
        notes = "Loaded Gaussian scene from PLY."

    metadata = SceneMetadata(
        scene_id=scene_id,
        source_path=input_path,
        notes=notes,
    )
    attributes: dict[str, Any] = {}
    if gaussian_layer is not None:
        gaussian_layer.metadata["scene_id"] = scene_id
        attributes["gaussian_layer"] = gaussian_layer

    return GaussianSplatScene(
        metadata=metadata,
        gaussian_count=gaussian_count,
        attributes=attributes,
    )


def load_gaussian_ply(path: str | Path) -> GaussianLayer:
    """Load a common 3D Gaussian Splatting PLY file.

    Supported fields include ``x``, ``y``, ``z``, ``opacity``, ``scale_*``,
    ``rot_*``, RGB colors, DC color features, and ``f_rest_*`` spherical
    harmonic/rest features. Only XYZ is required.
    """

    ply_path = Path(path)
    if not ply_path.exists():
        raise FileNotFoundError(f"Gaussian PLY file does not exist: {ply_path}")
    if not ply_path.is_file():
        raise ValueError(f"Gaussian PLY path is not a file: {ply_path}")

    ply_data = PlyData.read(ply_path)
    if "vertex" not in ply_data:
        raise ValueError(f"Gaussian PLY must contain a 'vertex' element: {ply_path}")

    vertex_data = ply_data["vertex"].data
    field_names = tuple(vertex_data.dtype.names or ())
    _require_fields(field_names, XYZ_FIELDS, ply_path)

    positions = _stack_fields(vertex_data, XYZ_FIELDS)
    opacity = _single_field(vertex_data, "opacity") if "opacity" in field_names else None
    scales = _stack_fields(vertex_data, SCALE_FIELDS) if _has_all(field_names, SCALE_FIELDS) else None
    rotations = (
        _stack_fields(vertex_data, ROTATION_FIELDS)
        if _has_all(field_names, ROTATION_FIELDS)
        else None
    )
    features_dc = _stack_fields(vertex_data, DC_FIELDS) if _has_all(field_names, DC_FIELDS) else None
    rest_fields = _indexed_fields(field_names, "f_rest_")
    features_rest = _stack_fields(vertex_data, rest_fields) if rest_fields else None

    rgb_colors = _stack_fields(vertex_data, RGB_FIELDS) if _has_all(field_names, RGB_FIELDS) else None
    colors: NDArray[np.float32] | None = None
    color_format: str | None = None
    if rgb_colors is not None:
        colors = _normalize_rgb(rgb_colors)
        color_format = "rgb"
    elif features_dc is not None:
        colors = features_dc
        color_format = "dc"

    metadata = {
        "source_path": str(ply_path),
        "format": "ply",
        "ply_text": bool(getattr(ply_data, "text", False)),
        "byte_order": str(getattr(ply_data, "byte_order", "")),
        "field_names": field_names,
        "gaussian_count": int(positions.shape[0]),
        "has_opacity": opacity is not None,
        "has_scales": scales is not None,
        "has_rotations": rotations is not None,
        "has_rgb": rgb_colors is not None,
        "has_features_dc": features_dc is not None,
        "features_rest_count": len(rest_fields),
        "comments": tuple(str(comment) for comment in getattr(ply_data, "comments", ())),
    }

    return GaussianLayer(
        positions=positions,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        colors=colors,
        color_format=color_format,
        features_dc=features_dc,
        features_rest=features_rest,
        metadata=metadata,
    )


def _require_fields(field_names: tuple[str, ...], required: tuple[str, ...], path: Path) -> None:
    missing = tuple(field for field in required if field not in field_names)
    if missing:
        missing_text = ", ".join(missing)
        available_text = ", ".join(field_names) or "none"
        raise ValueError(
            f"Gaussian PLY is missing required vertex field(s): {missing_text}. "
            f"Available fields in {path}: {available_text}."
        )


def _has_all(field_names: tuple[str, ...], fields: tuple[str, ...]) -> bool:
    return all(field in field_names for field in fields)


def _single_field(data: NDArray[Any], field: str) -> NDArray[np.float32]:
    return np.asarray(data[field], dtype=np.float32)


def _stack_fields(data: NDArray[Any], fields: tuple[str, ...]) -> NDArray[np.float32]:
    if not fields:
        return np.empty((len(data), 0), dtype=np.float32)
    columns = [np.asarray(data[field], dtype=np.float32) for field in fields]
    return np.column_stack(columns).astype(np.float32, copy=False)


def _indexed_fields(field_names: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    fields = tuple(field for field in field_names if field.startswith(prefix))

    def sort_key(field: str) -> tuple[int, int | str]:
        suffix = field.removeprefix(prefix)
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, suffix)

    return tuple(sorted(fields, key=sort_key))


def _normalize_rgb(rgb_colors: NDArray[np.float32]) -> NDArray[np.float32]:
    if rgb_colors.size == 0:
        return rgb_colors
    if float(np.nanmax(rgb_colors)) > 1.0:
        return (rgb_colors / 255.0).astype(np.float32, copy=False)
    return rgb_colors
