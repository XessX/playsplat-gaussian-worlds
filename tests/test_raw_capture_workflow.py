from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

import pytest

from scripts.create_scene_checklist import create_scene_checklist, render_scene_checklist
from scripts.extract_video_frames import build_parser as build_frame_parser
from scripts.extract_video_frames import extract_video_frames
from scripts.find_3dgs_point_cloud import (
    extract_iteration,
    find_point_cloud_candidates,
    select_best_candidate,
    write_point_cloud_candidates_csv,
)
from scripts.generate_3dgs_training_plan import generate_training_plan
from scripts.validate_capture_images import (
    read_image_dimensions,
    validate_capture_images,
    write_capture_quality_csv,
    write_capture_quality_report,
)


def test_frame_extraction_parser_and_missing_video_error(tmp_path: Path) -> None:
    parser = build_frame_parser()
    args = parser.parse_args(
        [
            "--video",
            "capture.mp4",
            "--output-dir",
            "raw_captures/room01/images",
            "--fps",
            "2",
            "--max-frames",
            "25",
            "--start-time",
            "1.5",
            "--end-time",
            "10",
            "--overwrite",
        ]
    )

    assert args.video == Path("capture.mp4")
    assert args.output_dir == Path("raw_captures/room01/images")
    assert args.fps == 2.0
    assert args.max_frames == 25
    assert args.start_time == 1.5
    assert args.end_time == 10.0
    assert args.overwrite is True

    with pytest.raises(FileNotFoundError, match="Video file does not exist"):
        extract_video_frames(
            video_path=tmp_path / "missing.mp4",
            output_dir=tmp_path / "frames",
        )


def test_image_validator_on_tiny_generated_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "raw_captures" / "room01" / "images"
    _write_png(images_dir / "frame_000001.png", width=320, height=240)
    _write_png(images_dir / "frame_000002.png", width=320, height=240)
    output_path = tmp_path / "capture_quality_report.json"
    csv_path = tmp_path / "capture_quality_report.csv"

    report = validate_capture_images(images_dir)
    write_capture_quality_report(report, output_path)
    write_capture_quality_csv(report, csv_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert report.status == "needs_more_images"
    assert report.image_count == 2
    assert report.resolution_summary["min_width"] == 320
    assert report.resolution_summary["max_height"] == 240
    assert any("fewer than 80 images" in warning for warning in report.warnings)
    assert all("very_small_resolution" in image.warnings for image in report.images)
    assert all("duplicate_file_size" in image.warnings for image in report.images)
    assert payload["image_count"] == 2
    assert csv_path.read_text(encoding="utf-8").startswith("image_path,width,height")
    assert read_image_dimensions(images_dir / "frame_000001.png") == (320, 240)


def test_image_validator_handles_missing_folder_gracefully(tmp_path: Path) -> None:
    report = validate_capture_images(tmp_path / "missing" / "images")

    assert report.status == "missing_images_dir"
    assert report.image_count == 0
    assert report.images == ()
    assert "missing image folder" in report.warnings[0]


def test_training_plan_generation(tmp_path: Path) -> None:
    images_dir = tmp_path / "raw_captures" / "room01" / "images"
    images_dir.mkdir(parents=True)
    output_dir = tmp_path / "raw_captures" / "room01"

    plan = generate_training_plan(
        scene_id="room01",
        images_dir=images_dir,
        output_dir=output_dir,
        trainer_name="graphdeco_3dgs",
        trainer_root=tmp_path / "tools" / "gaussian-splatting",
    )

    markdown = (output_dir / "training_commands.md").read_text(encoding="utf-8")
    payload = json.loads((output_dir / "training_plan.json").read_text(encoding="utf-8"))
    assert plan.scene_id == "room01"
    assert plan.trainer_name == "graphdeco_3dgs"
    assert "python convert.py" in markdown
    assert "python train.py" in markdown
    assert payload["staging_path"] == "data/scenes/room01/point_cloud.ply"
    assert payload["expected_point_cloud"].endswith(
        "training_output/point_cloud/iteration_30000/point_cloud.ply"
    )


def test_scene_checklist_generation(tmp_path: Path) -> None:
    output_path = tmp_path / "raw_captures" / "room01" / "notes.md"

    created = create_scene_checklist(
        scene_id="room01",
        category="indoor_room",
        output_path=output_path,
    )
    text = output_path.read_text(encoding="utf-8")

    assert created == output_path
    assert "# Scene Checklist: room01" in text
    assert "- Scene category: `indoor_room`" in text
    assert "- [ ] Final `point_cloud.ply` exists." in text
    assert "Registry validation passed" in render_scene_checklist(
        scene_id="room01",
        category="indoor_room",
    )


def test_find_point_cloud_prefers_highest_iteration(tmp_path: Path) -> None:
    low = _write_ply(
        tmp_path / "training_output" / "point_cloud" / "iteration_7000" / "point_cloud.ply",
    )
    high = _write_ply(
        tmp_path / "training_output" / "point_cloud" / "iteration_30000" / "point_cloud.ply",
    )
    no_iteration = _write_ply(tmp_path / "training_output" / "point_cloud.ply")
    csv_path = tmp_path / "point_cloud_candidates.csv"

    candidates = find_point_cloud_candidates(tmp_path / "training_output")
    best = select_best_candidate(candidates)
    write_point_cloud_candidates_csv(candidates, csv_path)

    assert len(candidates) == 3
    assert extract_iteration(low) == 7000
    assert extract_iteration(no_iteration) is None
    assert best is not None
    assert best.path == high
    assert best.iteration == 30000
    assert "iteration_30000" in csv_path.read_text(encoding="utf-8")
    assert find_point_cloud_candidates(tmp_path / "missing_root") == []


def _write_ply(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_png(path: Path, *, width: int, height: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = b"".join(b"\x00" + (b"\x80\x80\x80" * width) for _ in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw_rows))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)
    return path


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)
