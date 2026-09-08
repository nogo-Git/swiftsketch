#!/usr/bin/env python3
"""Plot Recognition scores against disconnected-stroke ratios.

The discontinuity CSV is joined to the three evaluation CSVs by method,
class, sample ID, and stroke count.  Colors identify initialization methods,
while marker shapes identify the number of strokes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


METHOD_SOURCES = {
    "SDXL_normal": ("normal", "org"),
    "SDXL_semantic": ("semantic", "ours"),
    "SDXL_semantic_overlap": ("semantic_overlap", "overlap"),
}
METHOD_COLORS = {
    "org": "#4C78A8",
    "ours": "#F58518",
    "overlap": "#54A24B",
}
STROKE_MARKERS = {16: "o", 24: "s", 32: "^"}


@dataclass(frozen=True)
class PlotRecord:
    method: str
    class_name: str
    sample_id: int
    stroke_count: int
    recognition_score: float
    disconnected_stroke_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discontinuity-csv",
        type=Path,
        default=Path("scores/bezier_discontinuity_scores.csv"),
        help="CSV produced by compute_bezier_discontinuity.py.",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path.home() / "sketch/eval/outputs_eval_20260825",
        help="Directory containing normal/semantic/semantic_overlap scores.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scores/recognition_vs_discontinuity"),
        help="Directory in which plot images are written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png",),
        help="Output image formats (default: png).",
    )
    return parser.parse_args()


def split_category(category: str) -> tuple[str, int]:
    """Split a discontinuity category such as ``airplane_3``."""
    try:
        class_name, sample_text = category.rsplit("_", 1)
        return class_name, int(sample_text)
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"invalid category {category!r}; expected <class>_<sample_id>"
        ) from error


def read_recognition_scores(
    eval_root: Path,
) -> dict[tuple[str, str, int, int], float]:
    scores: dict[tuple[str, str, int, int], float] = {}
    for _, (directory, method) in METHOD_SOURCES.items():
        csv_path = eval_root / directory / "scores.csv"
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                key = (
                    method,
                    row["class_name"],
                    int(row["sample_id"]),
                    int(row["stroke_count"]),
                )
                if key in scores:
                    raise ValueError(f"duplicate Recognition score for {key}")
                scores[key] = float(row["recognition_score"])
    return scores


def join_scores(discontinuity_csv: Path, eval_root: Path) -> list[PlotRecord]:
    recognition_scores = read_recognition_scores(eval_root)
    records: list[PlotRecord] = []
    missing: list[tuple[str, str, int, int]] = []

    with discontinuity_csv.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            dataset = row["dataset"]
            if dataset not in METHOD_SOURCES:
                raise ValueError(f"unknown dataset {dataset!r}")
            method = METHOD_SOURCES[dataset][1]
            class_name, sample_id = split_category(row["category"])
            stroke_count = int(row["stroke_count"])
            key = (method, class_name, sample_id, stroke_count)
            recognition_score = recognition_scores.get(key)
            if recognition_score is None:
                missing.append(key)
                continue
            records.append(
                PlotRecord(
                    method=method,
                    class_name=class_name,
                    sample_id=sample_id,
                    stroke_count=stroke_count,
                    recognition_score=recognition_score,
                    disconnected_stroke_ratio=float(
                        row["disconnected_stroke_ratio"]
                    ),
                )
            )

    if missing:
        preview = ", ".join(str(key) for key in missing[:5])
        raise ValueError(
            f"Recognition scores are missing for {len(missing)} rows: {preview}"
        )
    return records


def scatter_records(axis: object, records: Iterable[PlotRecord]) -> None:
    for method in METHOD_COLORS:
        for stroke_count, marker in STROKE_MARKERS.items():
            selected = [
                record
                for record in records
                if record.method == method
                and record.stroke_count == stroke_count
            ]
            if not selected:
                continue
            axis.scatter(
                [record.recognition_score for record in selected],
                [record.disconnected_stroke_ratio for record in selected],
                color=METHOD_COLORS[method],
                marker=marker,
                s=34,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.35,
            )


def configure_axis(axis: object, title: str | None = None) -> None:
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
    axis.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
    axis.grid(True, alpha=0.2, linewidth=0.6)
    if title is not None:
        axis.set_title(title)


def add_figure_legend(figure: object) -> None:
    from matplotlib.lines import Line2D

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=METHOD_COLORS[method],
            label=method,
            markersize=6,
        )
        for method in METHOD_COLORS
    ]
    stroke_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="#555555",
            label=f"{stroke_count} strokes",
            markersize=6,
        )
        for stroke_count, marker in STROKE_MARKERS.items()
    ]
    figure.legend(
        handles=method_handles + stroke_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
    )


def save_figure(
    figure: object,
    output_stem: Path,
    formats: Sequence[str],
) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for image_format in formats:
        path = output_stem.with_suffix(f".{image_format}")
        figure.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    return paths


def create_plots(
    records: Sequence[PlotRecord],
    output_dir: Path,
    formats: Sequence[str],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("no joined scores to plot")

    output_paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    scatter_records(axis, records)
    configure_axis(axis)
    axis.set_xlabel("Recognition score")
    axis.set_ylabel("Disconnected-stroke ratio")
    axis.set_title("Recognition vs. disconnected-stroke ratio")
    add_figure_legend(figure)
    figure.subplots_adjust(bottom=0.20)
    output_paths.extend(save_figure(figure, output_dir / "overall", formats))
    plt.close(figure)

    class_names = sorted({record.class_name for record in records})
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(15, 6.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, class_name in zip(axes.flat, class_names):
        class_records = [
            record for record in records if record.class_name == class_name
        ]
        scatter_records(axis, class_records)
        configure_axis(axis, class_name)
    for axis in axes.flat[len(class_names):]:
        axis.set_visible(False)
    figure.supxlabel("Recognition score", y=0.10)
    figure.supylabel("Disconnected-stroke ratio", x=0.06)
    figure.suptitle("Within-class relationship", y=0.98)
    add_figure_legend(figure)
    figure.subplots_adjust(
        left=0.10, right=0.98, top=0.90, bottom=0.19, wspace=0.13, hspace=0.30
    )
    output_paths.extend(
        save_figure(figure, output_dir / "by_class", formats)
    )
    plt.close(figure)

    class_dir = output_dir / "classes"
    for class_name in class_names:
        class_records = [
            record for record in records if record.class_name == class_name
        ]
        figure, axis = plt.subplots(figsize=(6.4, 5.2))
        scatter_records(axis, class_records)
        configure_axis(axis, class_name)
        axis.set_xlabel("Recognition score")
        axis.set_ylabel("Disconnected-stroke ratio")
        add_figure_legend(figure)
        figure.subplots_adjust(bottom=0.22)
        output_paths.extend(
            save_figure(figure, class_dir / class_name, formats)
        )
        plt.close(figure)

    return output_paths


def main() -> int:
    args = parse_args()
    try:
        records = join_scores(args.discontinuity_csv, args.eval_root)
        output_paths = create_plots(records, args.output_dir, args.formats)
    except (KeyError, OSError, ValueError) as error:
        print(f"error: {error}")
        return 1

    print(f"Joined {len(records)} sketches.")
    for path in output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
