"""Extract still frames from a raw capture video for external 3DGS training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Sequence, cast


@dataclass(frozen=True)
class FrameExtractionResult:
    """Summary of a video frame extraction run."""

    video_path: Path
    output_dir: Path
    extracted_frame_count: int
    video_fps: float
    duration_seconds: float


def build_parser() -> argparse.ArgumentParser:
    """Build the frame extraction CLI parser."""

    parser = argparse.ArgumentParser(
        description="Extract frames from a video into raw_captures/<scene_id>/images.",
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where frame_000001.jpg files will be written.",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Target extraction FPS.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to write.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time in seconds.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Optional end time in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing frame_*.jpg outputs before extraction.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run frame extraction from the command line."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = extract_video_frames(
            video_path=args.video,
            output_dir=args.output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
            start_time=args.start_time,
            end_time=args.end_time,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print_frame_extraction_summary(result)
    return 0


def extract_video_frames(
    *,
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 2.0,
    max_frames: int | None = None,
    start_time: float = 0.0,
    end_time: float | None = None,
    overwrite: bool = False,
) -> FrameExtractionResult:
    """Extract video frames using OpenCV if it is installed."""

    source_path = Path(video_path)
    destination = Path(output_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {source_path}")
    if fps <= 0:
        raise ValueError("--fps must be greater than zero.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be greater than zero when provided.")
    if start_time < 0:
        raise ValueError("--start-time must be non-negative.")
    if end_time is not None and end_time <= start_time:
        raise ValueError("--end-time must be greater than --start-time.")

    cv2 = _load_cv2()
    destination.mkdir(parents=True, exist_ok=True)
    existing_outputs = sorted(destination.glob("frame_*.jpg"))
    if existing_outputs and not overwrite:
        raise ValueError(
            f"{destination} already contains frame_*.jpg files; use --overwrite to replace them."
        )
    if overwrite:
        for frame_path in existing_outputs:
            frame_path.unlink()

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source_path}")

    try:
        video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = total_frames / video_fps if video_fps > 0 else 0.0
        frame_step = max(1, int(round(video_fps / fps))) if video_fps > 0 else 1
        start_frame = int(round(start_time * video_fps)) if video_fps > 0 else 0
        end_frame = int(round(end_time * video_fps)) if end_time and video_fps > 0 else None
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        extracted = 0
        frame_index = start_frame
        while True:
            if end_frame is not None and frame_index >= end_frame:
                break
            ok, frame = capture.read()
            if not ok:
                break
            should_write = (frame_index - start_frame) % frame_step == 0
            if should_write:
                extracted += 1
                output_path = destination / f"frame_{extracted:06d}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"OpenCV failed to write frame: {output_path}")
                if max_frames is not None and extracted >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()

    return FrameExtractionResult(
        video_path=source_path,
        output_dir=destination,
        extracted_frame_count=extracted,
        video_fps=video_fps,
        duration_seconds=duration_seconds,
    )


def print_frame_extraction_summary(result: FrameExtractionResult) -> None:
    """Print a concise extraction summary."""

    print(f"extracted frame count: {result.extracted_frame_count}")
    print(f"video FPS: {result.video_fps:.3f}")
    print(f"duration: {result.duration_seconds:.3f} seconds")
    print(f"output directory: {result.output_dir}")


def _load_cv2() -> Any:
    try:
        return cast(Any, importlib.import_module("cv2"))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for video frame extraction. "
            'Install it with: pip install opencv-python'
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
