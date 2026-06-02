"""Generate paper-ready assets from a PlaySplat benchmark summary."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CLEAN_COLUMNS = (
    "scene_id",
    "category",
    "source",
    "split",
    "status",
    "gaussian_count",
    "proxy_face_count",
    "collision_face_count",
    "collision_face_reduction_ratio",
    "walkable_area",
    "walkable_area_ratio",
    "semantic_status",
    "affordance_status",
    "overall_playability_score",
    "warning_count",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark asset generator parser."""

    parser = argparse.ArgumentParser(description="Generate PlaySplat benchmark assets.")
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        required=True,
        help="Path to benchmark_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_assets/benchmark"),
        help="Directory where benchmark paper assets are written.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="PlaySplat Benchmark Summary",
        help="Title used in the generated Markdown summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark asset generator."""

    parser = build_parser()
    args = parser.parse_args(argv)
    generate_benchmark_assets(
        benchmark_summary=args.benchmark_summary,
        output_dir=args.output_dir,
        title=args.title,
    )
    print(f"Benchmark assets written: {args.output_dir}")
    return 0


def generate_benchmark_assets(
    *,
    benchmark_summary: str | Path,
    output_dir: str | Path,
    title: str = "PlaySplat Benchmark Summary",
) -> dict[str, Path]:
    """Generate all benchmark paper assets."""

    rows = read_csv_rows(benchmark_summary)
    clean_rows = clean_benchmark_rows(rows)
    root = Path(output_dir)
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clean_csv_path = write_clean_csv(clean_rows, tables_dir / "benchmark_summary_clean.csv")
    markdown_path = write_text(
        markdown_table(clean_rows, CLEAN_COLUMNS),
        tables_dir / "benchmark_summary.md",
    )
    latex_path = write_text(
        latex_table(clean_rows, CLEAN_COLUMNS, caption="PlaySplat benchmark summary."),
        tables_dir / "benchmark_summary.tex",
    )
    figure_paths = generate_figures(rows, figures_dir)
    summary_path = write_text(
        generate_summary_markdown(rows, title=title),
        root / "benchmark_summary.md",
    )
    return {
        "benchmark_summary_clean_csv": clean_csv_path,
        "benchmark_summary_markdown": markdown_path,
        "benchmark_summary_latex": latex_path,
        "benchmark_summary": summary_path,
        **figure_paths,
    }


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read benchmark summary CSV rows."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def clean_benchmark_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Format benchmark rows for paper tables."""

    return [
        {
            "scene_id": str(row.get("scene_id", "")),
            "category": str(row.get("category", "")),
            "source": str(row.get("source", "")),
            "split": str(row.get("split", "")),
            "status": str(row.get("status", "")),
            "gaussian_count": format_int(row.get("gaussian_count")),
            "proxy_face_count": format_int(row.get("proxy_face_count")),
            "collision_face_count": format_int(row.get("collision_face_count")),
            "collision_face_reduction_ratio": format_float(
                row.get("collision_face_reduction_ratio"),
                decimals=4,
            ),
            "walkable_area": format_float(row.get("walkable_area"), decimals=3),
            "walkable_area_ratio": format_float(row.get("walkable_area_ratio"), decimals=4),
            "semantic_status": str(row.get("semantic_status", "")),
            "affordance_status": str(row.get("affordance_status", "")),
            "overall_playability_score": format_float(
                row.get("overall_playability_score"),
                decimals=4,
            ),
            "warning_count": format_int(row.get("warning_count")),
        }
        for row in rows
    ]


def write_clean_csv(rows: Sequence[Mapping[str, str]], output_path: str | Path) -> Path:
    """Write clean benchmark CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEAN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def markdown_table(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    """Render a Markdown table."""

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def latex_table(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
    *,
    caption: str,
) -> str:
    """Render a simple LaTeX table."""

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{escape_latex(caption)}}}",
        f"\\begin{{tabular}}{{{'l' * len(columns)}}}",
        "\\hline",
        " & ".join(escape_latex(column) for column in columns) + r" \\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(escape_latex(str(row.get(column, ""))) for column in columns) + r" \\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def generate_figures(rows: Sequence[Mapping[str, str]], output_dir: str | Path) -> dict[str, Path]:
    """Generate benchmark figure PNGs."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "collision_faces_by_scene": directory / "collision_faces_by_scene.png",
        "walkable_ratio_by_scene": directory / "walkable_ratio_by_scene.png",
        "reduction_ratio_by_scene": directory / "reduction_ratio_by_scene.png",
    }
    plot_bar(rows, "collision_face_count", "Collision Faces by Scene", paths["collision_faces_by_scene"])
    plot_bar(rows, "walkable_area_ratio", "Walkable Ratio by Scene", paths["walkable_ratio_by_scene"])
    plot_bar(
        rows,
        "collision_face_reduction_ratio",
        "Collision Reduction Ratio by Scene",
        paths["reduction_ratio_by_scene"],
    )
    return paths


def plot_bar(
    rows: Sequence[Mapping[str, str]],
    metric: str,
    title: str,
    output_path: str | Path,
) -> Path:
    """Plot a single benchmark metric by scene."""

    path = Path(output_path)
    valid_rows = [row for row in rows if _is_float(row.get(metric))]
    labels = [str(row.get("scene_id", "")) for row in valid_rows]
    values = [float(row[metric]) for row in valid_rows]
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    axis.bar(labels, values)
    axis.set_title(title)
    axis.set_xlabel("Scene")
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def generate_summary_markdown(rows: Sequence[Mapping[str, str]], *, title: str) -> str:
    """Generate a compact Markdown benchmark summary."""

    successful_rows = [row for row in rows if row.get("status") == "ready_prototype"]
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- Total scenes: {len(rows)}",
            f"- Successful scenes: {len(successful_rows)}",
            f"- Failed scenes: {len(failed_rows)}",
            "- Average collision face count: "
            + format_average(successful_rows, "collision_face_count", decimals=0),
            "- Average collision reduction ratio: "
            + format_average(successful_rows, "collision_face_reduction_ratio", decimals=4),
            "- Average walkable area ratio: "
            + format_average(successful_rows, "walkable_area_ratio", decimals=4),
            "",
        ]
    )


def write_text(text: str, output_path: str | Path) -> Path:
    """Write a text file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def format_int(value: Any) -> str:
    """Format a count-like value."""

    if value is None or value == "":
        return ""
    return str(int(round(float(value))))


def format_float(value: Any, *, decimals: int) -> str:
    """Format a float value with trimmed zeros."""

    if value is None or value == "":
        return ""
    return f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")


def format_average(rows: Sequence[Mapping[str, str]], metric: str, *, decimals: int) -> str:
    """Format an average metric for rows with numeric values."""

    values = [float(row[metric]) for row in rows if _is_float(row.get(metric))]
    if not values:
        return "n/a"
    average = sum(values) / len(values)
    if decimals == 0:
        return str(int(round(average)))
    return f"{average:.{decimals}f}"


def escape_latex(value: str) -> str:
    """Escape a small set of LaTeX table characters."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _is_float(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
