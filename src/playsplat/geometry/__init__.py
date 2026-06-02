"""Proxy geometry extraction."""

from playsplat.geometry.occupancy import build_voxel_occupancy
from playsplat.geometry.proxy import extract_proxy_geometry, extract_proxy_mesh, export_proxy_mesh

__all__ = [
    "build_voxel_occupancy",
    "export_proxy_mesh",
    "extract_proxy_geometry",
    "extract_proxy_mesh",
]
