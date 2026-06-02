from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from playsplat.geometry.structure import classify_proxy_mesh_structure, export_structure_meshes
from playsplat.types import ProxyMesh


def test_flat_horizontal_plane_is_floor_and_walkable() -> None:
    mesh = _floor_plane_mesh()

    structure = classify_proxy_mesh_structure(mesh, min_region_area=0.0)

    assert structure.floor is not None
    assert structure.walkable is not None
    assert structure.floor.face_count == 2
    assert structure.walkable.face_count == 2
    assert structure.metadata["floor_area"] > 0.0


def test_vertical_faces_are_classified_as_wall() -> None:
    mesh = _vertical_wall_mesh()

    structure = classify_proxy_mesh_structure(mesh, min_region_area=0.0)

    assert structure.walls is not None
    assert structure.walls.face_count == 2
    assert structure.metadata["wall_area"] > 0.0


def test_raised_cube_geometry_contains_obstacle_faces() -> None:
    mesh = _raised_cube_mesh()

    structure = classify_proxy_mesh_structure(
        mesh,
        floor_height_tolerance=0.05,
        min_region_area=0.0,
    )

    assert structure.obstacles is not None
    assert structure.obstacles.face_count > 0
    assert structure.metadata["obstacle_area"] > 0.0


def test_structure_mesh_export_writes_expected_files(tmp_path: Path) -> None:
    mesh = _mixed_scene_mesh()
    structure = classify_proxy_mesh_structure(
        mesh,
        floor_height_tolerance=0.05,
        min_region_area=0.0,
    )

    exports = export_structure_meshes(mesh, structure, tmp_path)

    expected = {"floor", "wall", "obstacle", "walkable"}
    assert expected.issubset(exports.keys())
    for path in exports.values():
        assert path.exists()
        assert path.stat().st_size > 0


def _floor_plane_mesh() -> ProxyMesh:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    return ProxyMesh(vertices=vertices, faces=faces)


def _vertical_wall_mesh() -> ProxyMesh:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return ProxyMesh(vertices=vertices, faces=faces)


def _raised_cube_mesh() -> ProxyMesh:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = np.asarray(box.vertices, dtype=np.float32)
    vertices[:, 1] += 1.0
    return ProxyMesh(vertices=vertices, faces=np.asarray(box.faces, dtype=np.int32))


def _mixed_scene_mesh() -> ProxyMesh:
    floor = _floor_plane_mesh()
    wall = _vertical_wall_mesh()
    cube = _raised_cube_mesh()

    vertices = np.concatenate([floor.vertices, wall.vertices, cube.vertices], axis=0)
    wall_faces = wall.faces + floor.vertices.shape[0]
    cube_faces = cube.faces + floor.vertices.shape[0] + wall.vertices.shape[0]
    faces = np.concatenate([floor.faces, wall_faces, cube_faces], axis=0)
    return ProxyMesh(
        vertices=vertices.astype(np.float32, copy=False),
        faces=faces.astype(np.int32, copy=False),
    )
