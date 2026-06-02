"""Proxy geometry extraction."""

from playsplat.geometry.occupancy import build_voxel_occupancy
from playsplat.geometry.proxy import extract_proxy_geometry, extract_proxy_mesh, export_proxy_mesh
from playsplat.geometry.simplify import export_collision_mesh, simplify_proxy_mesh
from playsplat.geometry.structure import (
    classify_proxy_mesh_structure,
    export_structure_meshes,
    scene_structure_to_dict,
)

__all__ = [
    "build_voxel_occupancy",
    "classify_proxy_mesh_structure",
    "export_collision_mesh",
    "export_proxy_mesh",
    "export_structure_meshes",
    "extract_proxy_geometry",
    "extract_proxy_mesh",
    "scene_structure_to_dict",
    "simplify_proxy_mesh",
]
