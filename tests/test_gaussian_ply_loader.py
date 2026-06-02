from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from playsplat.cli import main
from playsplat.gaussian.stats import compute_gaussian_stats
from playsplat.io.loaders import load_gaussian_ply


def test_load_gaussian_ply_reads_common_3dgs_fields(tmp_path: Path) -> None:
    ply_path = _write_gaussian_fixture(tmp_path)

    layer = load_gaussian_ply(ply_path)

    assert layer.gaussian_count == 2
    np.testing.assert_allclose(
        layer.positions,
        np.asarray([[0.0, 1.0, 2.0], [1.0, 3.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(layer.opacity, np.asarray([0.1, 0.9], dtype=np.float32))
    np.testing.assert_allclose(
        layer.scales,
        np.asarray([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        layer.rotations,
        np.asarray([[1.0, 0.0, 0.0, 0.0], [0.707, 0.0, 0.707, 0.0]], dtype=np.float32),
    )
    assert layer.color_format == "rgb"
    np.testing.assert_allclose(layer.colors[0], np.asarray([1.0, 128.0 / 255.0, 0.0]))
    np.testing.assert_allclose(
        layer.features_rest,
        np.asarray([[0.11, 0.12], [0.21, 0.22]], dtype=np.float32),
    )
    assert layer.metadata["has_rgb"] is True


def test_load_gaussian_ply_missing_xyz_raises_clear_error(tmp_path: Path) -> None:
    ply_path = tmp_path / "missing_xyz.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float opacity",
                "end_header",
                "0.0 1.0 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required vertex field.*z"):
        load_gaussian_ply(ply_path)


def test_compute_gaussian_stats_returns_expected_summary(tmp_path: Path) -> None:
    layer = load_gaussian_ply(_write_gaussian_fixture(tmp_path))

    stats = compute_gaussian_stats(layer)

    assert stats["num_gaussians"] == 2
    assert stats["bounding_box"]["min"] == [0.0, 1.0, 2.0]
    assert stats["bounding_box"]["max"] == [1.0, 3.0, 5.0]
    assert stats["scene_center"] == [0.5, 2.0, 3.5]
    assert stats["scene_size"] == [1.0, 2.0, 3.0]
    assert stats["opacity"]["min"] == 0.1
    assert stats["opacity"]["max"] == 0.9
    assert stats["scales"]["min"] == [0.01, 0.02, 0.03]
    assert stats["color_field_count"] == 5
    assert stats["estimated_memory_footprint"]["bytes"] > 0


def test_cli_writes_gaussian_stats_for_ply_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ply_path = _write_gaussian_fixture(tmp_path)
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  scene_id: fixture_scene",
                "input:",
                f"  path: \"{ply_path.as_posix()}\"",
                "output:",
                f"  directory: \"{output_dir.as_posix()}\"",
                "semantics:",
                "  vocabulary: []",
                "affordance:",
                "  labels: []",
                "export:",
                "  targets: []",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path)]) == 0

    captured = capsys.readouterr()
    stats_path = output_dir / "gaussian_stats.json"
    proxy_path = output_dir / "proxy_mesh.obj"
    proxy_metadata_path = output_dir / "proxy_metadata.json"
    structure_path = output_dir / "scene_structure.json"
    assert "Gaussian statistics" in captured.out
    assert stats_path.exists()
    assert proxy_path.exists()
    assert proxy_metadata_path.exists()
    assert structure_path.exists()
    assert json.loads(stats_path.read_text(encoding="utf-8"))["num_gaussians"] == 2


def _write_gaussian_fixture(tmp_path: Path) -> Path:
    ply_path = tmp_path / "gaussians_ascii.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "comment PlaySplat generated Gaussian fixture",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property float opacity",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property float f_rest_0",
                "property float f_rest_1",
                "end_header",
                "0.0 1.0 2.0 0.1 0.01 0.02 0.03 1.0 0.0 0.0 0.0 255 128 0 0.11 0.12",
                "1.0 3.0 5.0 0.9 0.04 0.05 0.06 0.707 0.0 0.707 0.0 64 32 255 0.21 0.22",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return ply_path
