"""Find final 3DGS point_cloud.ply candidates from an external training output."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Sequence


ITERATION_PATTERN = re.compile(r"iteration_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class PointCloudCandidate:
    """One discovered 3DGS point_cloud.ply candidate."""

    path: Path
    iteration: int | None
    size_mb: float
    modified_time: str


def build_parser() -> argparse.ArgumentParser:
    """Build the point cloud discovery CLI parser."""

    parser = argparse.ArgumentParser(
        description="Find point_cloud.ply candidates and prefer the highest iteration.",
    )
    parser.add_argument("--root", type=Path, required=True, help="Training output root to search.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV candidate report path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run point cloud candidate discovery."""

    parser = build_parser()
    args = parser.parse_args(argv)
    candidates = find_point_cloud_candidates(args.root)
    if args.output is not None:
        write_point_cloud_candidates_csv(candidates, args.output)
        print(f"candidate CSV: {args.output}")
    best = select_best_candidate(candidates)
    print(f"candidates found: {len(candidates)}")
    if best is None:
        print(f"best candidate: none found under {args.root}")
        return 0
    print(f"best candidate: {best.path}")
    print(f"iteration: {best.iteration if best.iteration is not None else 'unknown'}")
    print(f"size MB: {best.size_mb:.6f}")
    return 0


def find_point_cloud_candidates(root: str | Path) -> list[PointCloudCandidate]:
    """Find point_cloud.ply files under a training output root."""

    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []
    candidates: list[PointCloudCandidate] = []
    for path in sorted(root_path.rglob("point_cloud.ply")):
        stat = path.stat()
        candidates.append(
            PointCloudCandidate(
                path=path,
                iteration=extract_iteration(path),
                size_mb=stat.st_size / (1024 * 1024),
                modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            )
        )
    return candidates


def select_best_candidate(
    candidates: Sequence[PointCloudCandidate],
) -> PointCloudCandidate | None:
    """Select the best candidate, preferring the highest iteration folder."""

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            candidate.iteration if candidate.iteration is not None else -1,
            candidate.size_mb,
            candidate.modified_time,
            candidate.path.as_posix(),
        ),
    )


def extract_iteration(path: str | Path) -> int | None:
    """Extract the largest iteration number from a candidate path."""

    matches = [
        int(match.group(1))
        for part in Path(path).parts
        for match in [ITERATION_PATTERN.fullmatch(part)]
        if match is not None
    ]
    if not matches:
        return None
    return max(matches)


def write_point_cloud_candidates_csv(
    candidates: Sequence[PointCloudCandidate],
    output_path: str | Path,
) -> Path:
    """Write point_cloud.ply candidate details to CSV."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "iteration", "size_mb", "modified_time"],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "path": candidate.path.as_posix(),
                    "iteration": (
                        "" if candidate.iteration is None else str(candidate.iteration)
                    ),
                    "size_mb": f"{candidate.size_mb:.6f}",
                    "modified_time": candidate.modified_time,
                }
            )
    return destination


if __name__ == "__main__":
    raise SystemExit(main())
