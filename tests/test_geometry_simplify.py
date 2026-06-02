from __future__ import annotations

from pathlib import Path

import numpy as np

from playsplat.geometry import export_collision_mesh, simplify_proxy_mesh
from playsplat.types import ProxyMesh


def test_simplification_reduces_dense_mesh_below_target() -> None:
    mesh = _grid_mesh(resolution=24)

    simplified = simplify_proxy_mesh(mesh, target_face_count=120)

    assert simplified.faces.shape[0] <= 120
    assert simplified.metadata["original_face_count"] == mesh.faces.shape[0]
    assert simplified.metadata["simplified_face_count"] == simplified.faces.shape[0]
    assert simplified.metadata["target_face_count"] == 120
    assert simplified.metadata["method"] == "vertex_clustering"
    assert simplified.metadata["achieved_reduction_ratio"] > 0.0
    assert simplified.metadata["status"] == "target_reached"
    assert mesh.faces.shape[0] == 24 * 24 * 2


def test_simplification_returns_copy_for_tiny_mesh() -> None:
    mesh = ProxyMesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
    )

    simplified = simplify_proxy_mesh(mesh, target_face_count=10)

    assert simplified.faces.tolist() == [[0, 1, 2]]
    assert simplified.vertices.tolist() == mesh.vertices.tolist()
    assert simplified is not mesh
    assert simplified.metadata["status"] == "already_within_target"
    assert simplified.metadata["iterations"] == 0


def test_simplification_removes_degenerate_and_duplicate_faces() -> None:
    mesh = ProxyMesh(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray(
            [
                [0, 1, 2],
                [0, 1, 1],
                [0, 2, 1],
            ],
            dtype=np.int32,
        ),
    )

    simplified = simplify_proxy_mesh(
        mesh,
        target_face_count=1,
        clustering_voxel_size=0.001,
        max_iterations=1,
    )

    assert simplified.faces.shape == (1, 3)
    assert len({int(index) for index in simplified.faces[0]}) == 3


def test_export_collision_mesh_writes_file(tmp_path: Path) -> None:
    mesh = _grid_mesh(resolution=1)
    output_path = tmp_path / "collision_mesh.obj"

    exported = export_collision_mesh(mesh, output_path)

    assert exported == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _grid_mesh(resolution: int) -> ProxyMesh:
    vertices = []
    for y_index in range(resolution + 1):
        for x_index in range(resolution + 1):
            vertices.append(
                [
                    float(x_index) / float(resolution),
                    float(y_index) / float(resolution),
                    0.0,
                ]
            )

    faces = []
    stride = resolution + 1
    for y_index in range(resolution):
        for x_index in range(resolution):
            lower_left = y_index * stride + x_index
            lower_right = lower_left + 1
            upper_left = lower_left + stride
            upper_right = upper_left + 1
            faces.append([lower_left, lower_right, upper_left])
            faces.append([lower_right, upper_right, upper_left])

    return ProxyMesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        metadata={"method": "grid"},
    )
