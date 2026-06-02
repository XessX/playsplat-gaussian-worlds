from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_previews import main as preview_main
from playsplat.visualization import (
    generate_scene_previews,
    render_mesh_preview,
    render_playability_summary_card,
)


def test_mesh_preview_generation_from_tiny_mesh(tmp_path: Path) -> None:
    mesh_path = _write_tiny_obj(tmp_path / "mesh.obj")
    output_path = tmp_path / "mesh.png"

    rendered = render_mesh_preview(mesh_path, output_path, title="Tiny Mesh")

    assert rendered == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_playability_summary_card_generation_from_fake_report(tmp_path: Path) -> None:
    report_path = _write_fake_report(tmp_path / "playability_report.json")
    output_path = tmp_path / "summary.png"

    rendered = render_playability_summary_card(report_path, output_path)

    assert rendered == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_preview_directory_creation(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    _write_tiny_obj(scene_dir / "proxy_mesh.obj")
    _write_tiny_obj(scene_dir / "collision_mesh.obj")
    _write_fake_report(scene_dir / "playability_report.json")

    previews = generate_scene_previews(scene_dir)

    assert (scene_dir / "previews").exists()
    assert (scene_dir / "previews" / "proxy_mesh.png").exists()
    assert (scene_dir / "previews" / "collision_mesh.png").exists()
    assert (scene_dir / "previews" / "playability_summary.png").exists()
    assert set(previews) == {"proxy_mesh", "collision_mesh", "playability_summary"}


def test_missing_files_do_not_crash_preview_script(tmp_path: Path) -> None:
    scene_dir = tmp_path / "empty_scene"
    scene_dir.mkdir()

    assert preview_main(["--scene-output", str(scene_dir)]) == 0

    assert (scene_dir / "previews").exists()
    assert list((scene_dir / "previews").iterdir()) == []


def _write_tiny_obj(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_fake_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "ready_prototype",
                "summary": {"warning_count": 1},
                "metrics": {
                    "gaussian_count": 10,
                    "proxy_face_count": 1,
                    "collision_face_count": 1,
                    "collision_face_reduction_ratio": 0.0,
                    "floor_area": 1.0,
                    "wall_area": 2.0,
                    "obstacle_area": 3.0,
                    "walkable_area": 1.0,
                    "export_readiness_score": 0.8,
                    "overall_playability_score": 0.75,
                },
                "warnings": ["placeholder"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path
