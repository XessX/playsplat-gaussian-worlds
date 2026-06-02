"""Generate paper-ready tables, figures, and summaries from PlaySplat ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CLEAN_COLUMNS = (
    "run_id",
    "voxel_size",
    "target_face_count",
    "status",
    "proxy_face_count",
    "collision_face_count",
    "collision_face_reduction_ratio",
    "collision_face_to_proxy_face_ratio",
    "walkable_area",
    "walkable_area_ratio",
    "semantic_status",
    "affordance_status",
    "overall_playability_score",
    "warning_count",
)


BEST_RUN_KEYS = (
    ("best_lowest_collision_face_count", "Best by collision complexity"),
    ("best_highest_walkable_area_ratio", "Best by walkable ratio"),
    ("best_balanced", "Best balanced run"),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the paper asset generator CLI parser."""

    parser = argparse.ArgumentParser(
        description="Generate paper-ready PlaySplat ablation tables and figures.",
    )
    parser.add_argument(
        "--ablation-summary",
        type=Path,
        required=True,
        help="Path to ablation_summary.csv.",
    )
    parser.add_argument(
        "--best-runs",
        type=Path,
        default=None,
        help="Optional path to best_runs.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where paper assets are written.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="PlaySplat Ablation Study",
        help="Title used in summary assets.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the paper asset generator."""

    parser = build_parser()
    args = parser.parse_args(argv)
    generate_paper_assets(
        ablation_summary=args.ablation_summary,
        best_runs=args.best_runs,
        output_dir=args.output_dir,
        title=args.title,
    )
    print(f"Paper assets written: {args.output_dir}")
    return 0


def generate_paper_assets(
    *,
    ablation_summary: str | Path,
    best_runs: str | Path | None,
    output_dir: str | Path,
    title: str = "PlaySplat Ablation Study",
) -> dict[str, Path]:
    """Generate all paper assets and return their paths."""

    rows = read_csv_rows(ablation_summary)
    clean_rows = clean_ablation_rows(rows)
    best_runs_data = load_best_runs(best_runs)

    root = Path(output_dir)
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clean_csv_path = write_clean_csv(clean_rows, tables_dir / "ablation_summary_clean.csv")
    ablation_markdown = markdown_table(clean_rows, CLEAN_COLUMNS)
    ablation_markdown_path = write_text(
        ablation_markdown,
        tables_dir / "ablation_summary.md",
    )
    ablation_latex = latex_table(clean_rows, CLEAN_COLUMNS, caption="PlaySplat ablation summary.")
    ablation_latex_path = write_text(
        ablation_latex,
        tables_dir / "ablation_summary.tex",
    )

    best_rows = clean_best_run_rows(best_runs_data)
    best_markdown = markdown_table(best_rows, tuple(best_rows[0])) if best_rows else "No best runs available.\n"
    best_markdown_path = write_text(best_markdown, tables_dir / "best_runs.md")
    best_latex = (
        latex_table(best_rows, tuple(best_rows[0]), caption="Selected PlaySplat ablation runs.")
        if best_rows
        else "No best runs available.\n"
    )
    best_latex_path = write_text(best_latex, tables_dir / "best_runs.tex")

    figure_paths = generate_figures(rows, figures_dir)
    summary_markdown = generate_summary_markdown(
        rows=rows,
        best_runs=best_runs_data,
        title=title,
    )
    summary_path = write_text(summary_markdown, root / "summary.md")
    markdown_summary_path = write_text(summary_markdown, root / "markdown_summary.md")
    latex_tables_path = write_text(
        "\n\n".join((ablation_latex, best_latex)),
        root / "latex_tables.tex",
    )

    return {
        "ablation_summary_clean_csv": clean_csv_path,
        "ablation_summary_markdown": ablation_markdown_path,
        "ablation_summary_latex": ablation_latex_path,
        "best_runs_markdown": best_markdown_path,
        "best_runs_latex": best_latex_path,
        "summary_markdown": summary_path,
        "markdown_summary": markdown_summary_path,
        "latex_tables": latex_tables_path,
        **figure_paths,
    }


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def clean_ablation_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Return a formatted table for paper notes."""

    return [
        {
            "run_id": str(row.get("scene_id", "")),
            "voxel_size": format_float(row.get("voxel_size"), decimals=3),
            "target_face_count": format_int(row.get("target_face_count")),
            "status": str(row.get("status", "")),
            "proxy_face_count": format_int(row.get("proxy_face_count")),
            "collision_face_count": format_int(row.get("collision_face_count")),
            "collision_face_reduction_ratio": format_float(
                row.get("collision_face_reduction_ratio"),
                decimals=4,
            ),
            "collision_face_to_proxy_face_ratio": format_float(
                row.get("collision_face_to_proxy_face_ratio"),
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


def clean_best_run_rows(best_runs: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Return clean best-run table rows."""

    if best_runs is None:
        return []
    rows: list[dict[str, str]] = []
    for key, label in BEST_RUN_KEYS:
        run = best_runs.get(key)
        if not isinstance(run, Mapping):
            run = _best_run_alias(best_runs, key)
        if not isinstance(run, Mapping):
            continue
        rows.append(
            {
                "selection": label,
                "run_id": str(run.get("scene_id", "")),
                "voxel_size": format_float(run.get("voxel_size"), decimals=3),
                "target_face_count": format_int(run.get("target_face_count")),
                "collision_face_count": format_int(run.get("collision_face_count")),
                "walkable_area_ratio": format_float(run.get("walkable_area_ratio"), decimals=4),
                "overall_playability_score": format_float(
                    run.get("overall_playability_score"),
                    decimals=4,
                ),
                "warning_count": format_int(run.get("warning_count")),
            }
        )
    return rows


def write_clean_csv(rows: Sequence[Mapping[str, str]], output_path: str | Path) -> Path:
    """Write the cleaned ablation table as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEAN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def markdown_table(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    """Render a simple GitHub-flavored Markdown table."""

    if not columns:
        return "\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def latex_table(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
    *,
    caption: str | None = None,
) -> str:
    """Render a copyable standard LaTeX tabular table."""

    if not columns:
        return "\n"
    column_spec = "l" * len(columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
    ]
    if caption is not None:
        lines.append(f"\\caption{{{escape_latex(caption)}}}")
    lines.extend(
        [
            f"\\begin{{tabular}}{{{column_spec}}}",
            "\\hline",
            " & ".join(escape_latex(column) for column in columns) + r" \\",
            "\\hline",
        ]
    )
    for row in rows:
        lines.append(
            " & ".join(escape_latex(str(row.get(column, ""))) for column in columns) + r" \\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_figures(rows: Sequence[Mapping[str, str]], figures_dir: str | Path) -> dict[str, Path]:
    """Generate ablation figure PNGs."""

    directory = Path(figures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    numeric_rows = [row for row in rows if _row_has_numeric_fields(row)]
    paths = {
        "collision_faces_vs_voxel_size": directory / "collision_faces_vs_voxel_size.png",
        "reduction_ratio_vs_target_faces": directory / "reduction_ratio_vs_target_faces.png",
        "walkable_ratio_vs_voxel_size": directory / "walkable_ratio_vs_voxel_size.png",
        "playability_score_grid": directory / "playability_score_grid.png",
    }
    plot_collision_faces_vs_voxel_size(numeric_rows, paths["collision_faces_vs_voxel_size"])
    plot_reduction_ratio_vs_target_faces(numeric_rows, paths["reduction_ratio_vs_target_faces"])
    plot_walkable_ratio_vs_voxel_size(numeric_rows, paths["walkable_ratio_vs_voxel_size"])
    plot_playability_score_grid(numeric_rows, paths["playability_score_grid"])
    return paths


def plot_collision_faces_vs_voxel_size(
    rows: Sequence[Mapping[str, str]],
    output_path: str | Path,
) -> Path:
    """Plot collision face count versus voxel size grouped by target face count."""

    path = Path(output_path)
    figure, axis = plt.subplots(figsize=(7, 4.5), dpi=150)
    for target_face_count in sorted({_float(row["target_face_count"]) for row in rows}):
        group = sorted(
            (row for row in rows if _float(row["target_face_count"]) == target_face_count),
            key=lambda row: _float(row["voxel_size"]),
        )
        axis.plot(
            [_float(row["voxel_size"]) for row in group],
            [_float(row["collision_face_count"]) for row in group],
            marker="o",
            label=f"target {int(target_face_count)}",
        )
    axis.set_xlabel("Voxel size")
    axis.set_ylabel("Collision face count")
    axis.set_title("Collision Face Count vs Voxel Size")
    axis.grid(True, alpha=0.3)
    if rows:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_reduction_ratio_vs_target_faces(
    rows: Sequence[Mapping[str, str]],
    output_path: str | Path,
) -> Path:
    """Plot collision face reduction ratio versus target face count grouped by voxel size."""

    path = Path(output_path)
    figure, axis = plt.subplots(figsize=(7, 4.5), dpi=150)
    for voxel_size in sorted({_float(row["voxel_size"]) for row in rows}):
        group = sorted(
            (row for row in rows if _float(row["voxel_size"]) == voxel_size),
            key=lambda row: _float(row["target_face_count"]),
        )
        axis.plot(
            [_float(row["target_face_count"]) for row in group],
            [_float(row["collision_face_reduction_ratio"]) for row in group],
            marker="o",
            label=f"voxel {voxel_size:g}",
        )
    axis.set_xlabel("Target collision face count")
    axis.set_ylabel("Collision face reduction ratio")
    axis.set_title("Reduction Ratio vs Target Face Count")
    axis.grid(True, alpha=0.3)
    if rows:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_walkable_ratio_vs_voxel_size(
    rows: Sequence[Mapping[str, str]],
    output_path: str | Path,
) -> Path:
    """Plot walkable area ratio versus voxel size."""

    path = Path(output_path)
    figure, axis = plt.subplots(figsize=(7, 4.5), dpi=150)
    for target_face_count in sorted({_float(row["target_face_count"]) for row in rows}):
        group = sorted(
            (row for row in rows if _float(row["target_face_count"]) == target_face_count),
            key=lambda row: _float(row["voxel_size"]),
        )
        axis.plot(
            [_float(row["voxel_size"]) for row in group],
            [_float(row["walkable_area_ratio"]) for row in group],
            marker="o",
            label=f"target {int(target_face_count)}",
        )
    axis.set_xlabel("Voxel size")
    axis.set_ylabel("Walkable area ratio")
    axis.set_title("Walkable Area Ratio vs Voxel Size")
    axis.grid(True, alpha=0.3)
    if rows:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_playability_score_grid(
    rows: Sequence[Mapping[str, str]],
    output_path: str | Path,
) -> Path:
    """Plot a playability score heatmap over voxel size and target face count."""

    path = Path(output_path)
    voxel_sizes = sorted({_float(row["voxel_size"]) for row in rows})
    target_face_counts = sorted({_float(row["target_face_count"]) for row in rows})
    grid = np.full((len(target_face_counts), len(voxel_sizes)), np.nan, dtype=np.float32)
    voxel_to_index = {value: index for index, value in enumerate(voxel_sizes)}
    target_to_index = {value: index for index, value in enumerate(target_face_counts)}
    for row in rows:
        grid[
            target_to_index[_float(row["target_face_count"])],
            voxel_to_index[_float(row["voxel_size"])],
        ] = np.float32(_float(row["overall_playability_score"]))

    figure, axis = plt.subplots(figsize=(6.5, 4.8), dpi=150)
    image = axis.imshow(grid, aspect="auto", origin="lower")
    axis.set_xticks(range(len(voxel_sizes)))
    axis.set_xticklabels([f"{value:g}" for value in voxel_sizes])
    axis.set_yticks(range(len(target_face_counts)))
    axis.set_yticklabels([f"{int(value)}" for value in target_face_counts])
    axis.set_xlabel("Voxel size")
    axis.set_ylabel("Target collision face count")
    axis.set_title("Playability Score Grid")
    figure.colorbar(image, ax=axis, label="Overall playability score")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def generate_summary_markdown(
    *,
    rows: Sequence[Mapping[str, str]],
    best_runs: Mapping[str, Any] | None,
    title: str,
) -> str:
    """Generate a concise paper-ready Markdown summary."""

    successful_rows = [row for row in rows if row.get("status") == "ready_prototype"]
    lowest_collision = _min_row(rows, "collision_face_count")
    highest_walkable = _max_row(rows, "walkable_area_ratio")
    all_ready = len(rows) > 0 and len(successful_rows) == len(rows)
    convergence_note = convergence_observation(rows)

    lines = [
        f"# {title}",
        "",
        "## Run Summary",
        "",
        f"- Number of runs: {len(rows)}",
        f"- Successful ready_prototype runs: {len(successful_rows)}",
        f"- All runs reached ready_prototype: {'yes' if all_ready else 'no'}",
        "",
        "## Best Runs",
        "",
        _best_run_line(best_runs, "best_lowest_collision_face_count", "Best by collision complexity"),
        _best_run_line(best_runs, "best_highest_walkable_area_ratio", "Best by walkable ratio"),
        _best_run_line(best_runs, "best_balanced", "Balanced best run"),
        "",
        "## Automatic Observations",
        "",
        _metric_observation(
            "Lowest collision face count",
            lowest_collision,
            "collision_face_count",
        ),
        _metric_observation(
            "Highest walkable area ratio",
            highest_walkable,
            "walkable_area_ratio",
        ),
        f"- {convergence_note}",
        "",
    ]
    return "\n".join(lines)


def convergence_observation(rows: Sequence[Mapping[str, str]]) -> str:
    """Describe whether target face counts converged to similar collision counts."""

    messages: list[str] = []
    voxel_sizes = sorted({_float(row["voxel_size"]) for row in rows if _is_float(row.get("voxel_size"))})
    for voxel_size in voxel_sizes:
        group = [
            row
            for row in rows
            if _is_float(row.get("voxel_size"))
            and _float(row["voxel_size"]) == voxel_size
            and _is_float(row.get("collision_face_count"))
        ]
        if _has_similar_collision_counts(group):
            messages.append(f"voxel {voxel_size:g}")
    if messages:
        return (
            "Some target face counts converged to similar collision face counts at "
            + ", ".join(messages)
            + "."
        )
    return "No target face count convergence was detected with the current tolerance."


def load_best_runs(path: str | Path | None) -> dict[str, Any] | None:
    """Load best-runs JSON if available."""

    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return None
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    return data


def write_text(text: str, output_path: str | Path) -> Path:
    """Write text to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def format_int(value: Any) -> str:
    """Format a count-like value as an integer string."""

    if value is None or value == "":
        return ""
    return str(int(round(float(value))))


def format_float(value: Any, *, decimals: int) -> str:
    """Format a numeric value with fixed precision and trimmed zeros."""

    if value is None or value == "":
        return ""
    number = float(value)
    if math.isnan(number):
        return ""
    return f"{number:.{decimals}f}".rstrip("0").rstrip(".")


def escape_latex(value: str) -> str:
    """Escape a small set of LaTeX table special characters."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _row_has_numeric_fields(row: Mapping[str, str]) -> bool:
    required = (
        "voxel_size",
        "target_face_count",
        "collision_face_count",
        "collision_face_reduction_ratio",
        "walkable_area_ratio",
        "overall_playability_score",
    )
    return all(_is_float(row.get(field)) for field in required)


def _best_run_line(
    best_runs: Mapping[str, Any] | None,
    key: str,
    label: str,
) -> str:
    if best_runs is None:
        return f"- {label}: unavailable"
    run = best_runs.get(key)
    if not isinstance(run, Mapping):
        run = _best_run_alias(best_runs, key)
    if not isinstance(run, Mapping):
        return f"- {label}: unavailable"
    return (
        f"- {label}: {run.get('scene_id', 'unknown')} "
        f"(voxel_size={format_float(run.get('voxel_size'), decimals=3)}, "
        f"target_face_count={format_int(run.get('target_face_count'))}, "
        f"collision_face_count={format_int(run.get('collision_face_count'))}, "
        f"walkable_area_ratio={format_float(run.get('walkable_area_ratio'), decimals=4)})"
    )


def _best_run_alias(best_runs: Mapping[str, Any], key: str) -> Any:
    aliases = {
        "best_lowest_collision_face_count": "best_lowest_collision",
        "best_highest_walkable_area_ratio": "best_highest_walkable_ratio",
    }
    alias = aliases.get(key)
    if alias is None:
        return None
    return best_runs.get(alias)


def _metric_observation(
    label: str,
    row: Mapping[str, str] | None,
    metric: str,
) -> str:
    if row is None:
        return f"- {label}: unavailable"
    value = row.get(metric, "")
    formatted = format_int(value) if metric.endswith("count") else format_float(value, decimals=4)
    return f"- {label}: {formatted} from {row.get('scene_id', 'unknown')}"


def _min_row(rows: Sequence[Mapping[str, str]], metric: str) -> Mapping[str, str] | None:
    candidates = [row for row in rows if _is_float(row.get(metric))]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _float(row[metric]))


def _max_row(rows: Sequence[Mapping[str, str]], metric: str) -> Mapping[str, str] | None:
    candidates = [row for row in rows if _is_float(row.get(metric))]
    if not candidates:
        return None
    return max(candidates, key=lambda row: _float(row[metric]))


def _has_similar_collision_counts(rows: Sequence[Mapping[str, str]]) -> bool:
    counts = [_float(row["collision_face_count"]) for row in rows]
    for first_index, first in enumerate(counts):
        for second in counts[first_index + 1 :]:
            tolerance = max(1.0, 0.01 * max(first, second))
            if abs(first - second) <= tolerance:
                return True
    return False


def _is_float(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _float(value: Any) -> float:
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
