"""Validate raw capture image sets before external 3DGS training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import statistics
import struct
from typing import Any, Sequence, cast


SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
MIN_RECOMMENDED_IMAGES = 80
STRONG_RECOMMENDED_IMAGES = 150
VERY_SMALL_WIDTH = 640
VERY_SMALL_HEIGHT = 480


@dataclass(frozen=True)
class ImageQualityEntry:
    """Quality signals for one capture image."""

    image_path: str
    width: int | None
    height: int | None
    size_mb: float
    blur_score: float | None
    brightness_mean: float | None
    brightness_std: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CaptureQualityReport:
    """Capture image-set validation report."""

    images_dir: str
    status: str
    image_count: int
    minimum_recommended_images: int
    strong_recommended_images: int
    resolution_summary: dict[str, int | float | None]
    warnings: tuple[str, ...]
    images: tuple[ImageQualityEntry, ...]


def build_parser() -> argparse.ArgumentParser:
    """Build the capture image validator CLI parser."""

    parser = argparse.ArgumentParser(
        description="Validate raw capture images before external 3DGS training.",
    )
    parser.add_argument("--images", type=Path, required=True, help="Capture image directory.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON quality report path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional per-image CSV report path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run capture image validation."""

    parser = build_parser()
    args = parser.parse_args(argv)
    report = validate_capture_images(args.images)
    write_capture_quality_report(report, args.output)
    if args.csv_output is not None:
        write_capture_quality_csv(report, args.csv_output)
    print_capture_quality_summary(report, args.output, args.csv_output)
    return 0 if report.status != "missing_images_dir" else 1


def validate_capture_images(images_dir: str | Path) -> CaptureQualityReport:
    """Validate one capture image directory."""

    directory = Path(images_dir)
    if not directory.exists() or not directory.is_dir():
        return CaptureQualityReport(
            images_dir=str(directory),
            status="missing_images_dir",
            image_count=0,
            minimum_recommended_images=MIN_RECOMMENDED_IMAGES,
            strong_recommended_images=STRONG_RECOMMENDED_IMAGES,
            resolution_summary=_empty_resolution_summary(),
            warnings=(f"missing image folder: {directory}",),
            images=(),
        )

    image_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    duplicate_sizes = _duplicate_file_sizes(image_paths)
    entries = tuple(_inspect_image(path, duplicate_sizes) for path in image_paths)
    warnings = _image_set_warnings(entries)
    status = "ready" if len(image_paths) >= MIN_RECOMMENDED_IMAGES else "needs_more_images"
    return CaptureQualityReport(
        images_dir=str(directory),
        status=status,
        image_count=len(image_paths),
        minimum_recommended_images=MIN_RECOMMENDED_IMAGES,
        strong_recommended_images=STRONG_RECOMMENDED_IMAGES,
        resolution_summary=_resolution_summary(entries),
        warnings=warnings,
        images=entries,
    )


def write_capture_quality_report(
    report: CaptureQualityReport,
    output_path: str | Path,
) -> Path:
    """Write the JSON capture quality report."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(report), indent=2),
        encoding="utf-8",
    )
    return destination


def write_capture_quality_csv(report: CaptureQualityReport, output_path: str | Path) -> Path:
    """Write optional per-image capture quality details as CSV."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "width",
                "height",
                "size_mb",
                "blur_score",
                "brightness_mean",
                "brightness_std",
                "warnings",
            ],
        )
        writer.writeheader()
        for entry in report.images:
            writer.writerow(
                {
                    "image_path": entry.image_path,
                    "width": entry.width,
                    "height": entry.height,
                    "size_mb": f"{entry.size_mb:.6f}",
                    "blur_score": _optional_float_text(entry.blur_score),
                    "brightness_mean": _optional_float_text(entry.brightness_mean),
                    "brightness_std": _optional_float_text(entry.brightness_std),
                    "warnings": "; ".join(entry.warnings),
                }
            )
    return destination


def print_capture_quality_summary(
    report: CaptureQualityReport,
    output_path: str | Path,
    csv_output: str | Path | None = None,
) -> None:
    """Print a concise image-set validation summary."""

    print(f"status: {report.status}")
    print(f"image count: {report.image_count}")
    print(f"minimum recommended images: {report.minimum_recommended_images}")
    print(f"strong recommended images: {report.strong_recommended_images}")
    print(f"report: {output_path}")
    if csv_output is not None:
        print(f"csv report: {csv_output}")
    if report.warnings:
        print("warnings:")
        for warning in report.warnings:
            print(f"  {warning}")


def _inspect_image(
    path: Path,
    duplicate_sizes: set[int],
) -> ImageQualityEntry:
    width, height = read_image_dimensions(path)
    blur_score, brightness_mean, brightness_std = optional_image_quality_signals(path)
    size_mb = path.stat().st_size / (1024 * 1024)
    warnings: list[str] = []
    if width is None or height is None:
        warnings.append("could_not_read_dimensions")
    elif width < VERY_SMALL_WIDTH or height < VERY_SMALL_HEIGHT:
        warnings.append("very_small_resolution")
    if path.stat().st_size in duplicate_sizes:
        warnings.append("duplicate_file_size")
    return ImageQualityEntry(
        image_path=path.as_posix(),
        width=width,
        height=height,
        size_mb=size_mb,
        blur_score=blur_score,
        brightness_mean=brightness_mean,
        brightness_std=brightness_std,
        warnings=tuple(warnings),
    )


def read_image_dimensions(path: str | Path) -> tuple[int | None, int | None]:
    """Read PNG or JPEG image dimensions without mandatory image dependencies."""

    image_path = Path(path)
    suffix = image_path.suffix.lower()
    with image_path.open("rb") as handle:
        if suffix == ".png":
            signature = handle.read(24)
            if len(signature) >= 24 and signature.startswith(b"\x89PNG\r\n\x1a\n"):
                width, height = struct.unpack(">II", signature[16:24])
                return int(width), int(height)
            return None, None
        if suffix in {".jpg", ".jpeg"}:
            return _read_jpeg_dimensions(handle.read())
    return None, None


def optional_image_quality_signals(path: Path) -> tuple[float | None, float | None, float | None]:
    """Return optional blur and brightness signals when OpenCV or Pillow is installed."""

    cv2 = _load_optional_module("cv2")
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            blur_score = float(cv2.Laplacian(image, cv2.CV_64F).var())
            brightness_mean = float(image.mean())
            brightness_std = float(image.std())
            return blur_score, brightness_mean, brightness_std

    image_module = _load_optional_module("PIL.Image")
    image_stat_module = _load_optional_module("PIL.ImageStat")
    if image_module is not None and image_stat_module is not None:
        with image_module.open(str(path)) as image:
            grayscale = image.convert("L")
            stat = image_stat_module.Stat(grayscale)
            return None, float(stat.mean[0]), float(stat.stddev[0])

    return None, None, None


def _read_jpeg_dimensions(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 4 or not data.startswith(b"\xff\xd8"):
        return None, None
    index = 2
    while index < len(data):
        while index < len(data) and data[index] != 0xFF:
            index += 1
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            break
        segment_length = int.from_bytes(data[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(data):
            break
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if segment_length >= 7:
                height = int.from_bytes(data[index + 3 : index + 5], "big")
                width = int.from_bytes(data[index + 5 : index + 7], "big")
                return width, height
            return None, None
        index += segment_length
    return None, None


def _duplicate_file_sizes(image_paths: Sequence[Path]) -> set[int]:
    sizes: dict[int, int] = {}
    for path in image_paths:
        size = path.stat().st_size
        sizes[size] = sizes.get(size, 0) + 1
    return {size for size, count in sizes.items() if count > 1}


def _image_set_warnings(entries: Sequence[ImageQualityEntry]) -> tuple[str, ...]:
    warnings: list[str] = []
    image_count = len(entries)
    if image_count < MIN_RECOMMENDED_IMAGES:
        warnings.append(
            f"fewer than {MIN_RECOMMENDED_IMAGES} images; collect more before training."
        )
    elif image_count < STRONG_RECOMMENDED_IMAGES:
        warnings.append(
            f"fewer than {STRONG_RECOMMENDED_IMAGES} images; more coverage is recommended."
        )
    duplicate_count = sum("duplicate_file_size" in entry.warnings for entry in entries)
    if duplicate_count:
        warnings.append(f"{duplicate_count} image(s) share duplicate file sizes.")
    small_count = sum("very_small_resolution" in entry.warnings for entry in entries)
    if small_count:
        warnings.append(f"{small_count} image(s) are below {VERY_SMALL_WIDTH}x{VERY_SMALL_HEIGHT}.")
    return tuple(warnings)


def _resolution_summary(entries: Sequence[ImageQualityEntry]) -> dict[str, int | float | None]:
    widths = [entry.width for entry in entries if entry.width is not None]
    heights = [entry.height for entry in entries if entry.height is not None]
    if not widths or not heights:
        return _empty_resolution_summary()
    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "mean_width": round(float(statistics.fmean(widths)), 3),
        "min_height": min(heights),
        "max_height": max(heights),
        "mean_height": round(float(statistics.fmean(heights)), 3),
    }


def _empty_resolution_summary() -> dict[str, int | float | None]:
    return {
        "min_width": None,
        "max_width": None,
        "mean_width": None,
        "min_height": None,
        "max_height": None,
        "mean_height": None,
    }


def _optional_float_text(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _load_optional_module(module_name: str) -> Any | None:
    try:
        return cast(Any, importlib.import_module(module_name))
    except ModuleNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
