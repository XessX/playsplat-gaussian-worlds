"""Discover candidate PLY files without automatically registering them."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_EXCLUDE_PATTERNS = (
    "sparse3d_scirep_starter",
    "Scientific Reports",
    "scientific_reports",
    "outputs/",
    "outputs\\",
    "playsplat/outputs",
    "playsplat\\outputs",
)


CSV_FIELDS = (
    "path",
    "file_name",
    "size_mb",
    "parent_folder",
    "suspected_source",
    "excluded",
    "exclusion_reason",
)


@dataclass(frozen=True)
class CandidatePly:
    """One discovered PLY candidate row."""

    path: Path
    file_name: str
    size_mb: float
    parent_folder: str
    suspected_source: str
    excluded: bool
    exclusion_reason: str


def build_parser() -> argparse.ArgumentParser:
    """Build the discovery CLI parser."""

    parser = argparse.ArgumentParser(description="Discover candidate Gaussian Splat PLY files.")
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        required=True,
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/candidate_plys.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=(),
        help="Additional case-insensitive path substring to mark as excluded.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the candidate discovery CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    candidates = discover_candidate_plys(
        roots=args.root,
        exclude_patterns=tuple(args.exclude),
    )
    output_path = write_candidates_csv(candidates, args.output)
    print(f"Candidate PLY report saved: {output_path}")
    print(f"Discovered files: {len(candidates)}")
    print(f"Excluded files: {sum(1 for candidate in candidates if candidate.excluded)}")
    return 0


def discover_candidate_plys(
    *,
    roots: Sequence[Path],
    exclude_patterns: Sequence[str] = (),
) -> list[CandidatePly]:
    """Find .ply files under roots and annotate suspicious sources."""

    patterns = tuple(DEFAULT_EXCLUDE_PATTERNS) + tuple(exclude_patterns)
    candidates: list[CandidatePly] = []
    seen_paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.ply"):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            exclusion_reason = exclusion_reason_for_path(path, patterns)
            candidates.append(
                CandidatePly(
                    path=path,
                    file_name=path.name,
                    size_mb=round(path.stat().st_size / (1024.0 * 1024.0), 6),
                    parent_folder=path.parent.name,
                    suspected_source=suspected_source_for_path(path),
                    excluded=exclusion_reason != "",
                    exclusion_reason=exclusion_reason,
                )
            )
    return sorted(candidates, key=lambda candidate: str(candidate.path).lower())


def write_candidates_csv(candidates: Sequence[CandidatePly], output_path: str | Path) -> Path:
    """Write candidate rows to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "path": str(candidate.path),
                    "file_name": candidate.file_name,
                    "size_mb": f"{candidate.size_mb:.6f}",
                    "parent_folder": candidate.parent_folder,
                    "suspected_source": candidate.suspected_source,
                    "excluded": candidate.excluded,
                    "exclusion_reason": candidate.exclusion_reason,
                }
            )
    return path


def exclusion_reason_for_path(path: Path, patterns: Sequence[str]) -> str:
    """Return the first matching exclusion reason for a path."""

    normalized = _normalized_path(path)
    for pattern in patterns:
        normalized_pattern = pattern.lower().replace("\\", "/")
        if normalized_pattern in normalized:
            return pattern
    return ""


def suspected_source_for_path(path: Path) -> str:
    """Assign a coarse source label for a path."""

    normalized = _normalized_path(path)
    if "playsplat/outputs" in normalized:
        return "playsplat_generated_output"
    if "sparse3d_scirep_starter" in normalized:
        return "scientific_reports_sparse_view_project"
    if "scientific_reports" in normalized or "scientific reports" in normalized:
        return "scientific_reports"
    if "/outputs/" in normalized:
        return "generated_output"
    if path.name == "point_cloud.ply":
        return "point_cloud_ply"
    return "ply_file"


def _normalized_path(path: Path) -> str:
    return str(path).lower().replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
