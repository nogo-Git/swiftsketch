#!/usr/bin/env python3
"""Measure disconnected Bezier strokes in generated ControlSketch SVGs.

A stroke is considered disconnected when at least one of its two endpoints is
farther than the distance threshold from every other stroke. Distances from an
endpoint to cubic Bezier curves are approximated by flattening each curve into
short line segments.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DATASETS = (
    "SDXL_normal",
    "SDXL_semantic",
    "SDXL_semantic_overlap",
)
SUPPORTED_STROKE_COUNTS = (16, 24, 32)
RUN_STROKE_COUNT_RE = re.compile(r"_(16|24|32)_strokes$")
COMMAND_RE = re.compile(r"([MmCcZz])([^MmCcZz]*)")
UNSUPPORTED_COMMAND_RE = re.compile(r"[AaHhLlQqSsTtVv]")
NUMBER_RE = re.compile(
    r"[-+]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)

Point = tuple[float, float]
LineSegment = tuple[Point, Point]


@dataclass(frozen=True)
class CubicBezier:
    p0: Point
    p1: Point
    p2: Point
    p3: Point

    def point_at(self, t: float) -> Point:
        one_minus_t = 1.0 - t
        x = (
            one_minus_t**3 * self.p0[0]
            + 3.0 * one_minus_t**2 * t * self.p1[0]
            + 3.0 * one_minus_t * t**2 * self.p2[0]
            + t**3 * self.p3[0]
        )
        y = (
            one_minus_t**3 * self.p0[1]
            + 3.0 * one_minus_t**2 * t * self.p1[1]
            + 3.0 * one_minus_t * t**2 * self.p2[1]
            + t**3 * self.p3[1]
        )
        return x, y


@dataclass(frozen=True)
class Stroke:
    curves: tuple[CubicBezier, ...]

    @property
    def endpoints(self) -> tuple[Point, Point]:
        return self.curves[0].p0, self.curves[-1].p3

    def flatten(self, samples_per_curve: int) -> tuple[LineSegment, ...]:
        segments: list[LineSegment] = []
        for curve in self.curves:
            previous = curve.p0
            for index in range(1, samples_per_curve + 1):
                current = curve.point_at(index / samples_per_curve)
                segments.append((previous, current))
                previous = current
        return tuple(segments)


@dataclass(frozen=True)
class DiscontinuityResult:
    num_strokes: int
    num_disconnected_strokes: int
    num_disconnected_endpoints: int

    @property
    def disconnected_stroke_ratio(self) -> float:
        if self.num_strokes == 0:
            return 0.0
        return self.num_disconnected_strokes / self.num_strokes

    @property
    def disconnected_endpoint_ratio(self) -> float:
        if self.num_strokes == 0:
            return 0.0
        return self.num_disconnected_endpoints / (2 * self.num_strokes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("ControlSketch/output_sketches"),
        help="Directory containing the SDXL_* output directories.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset directories below --input-root.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Maximum endpoint-to-other-stroke distance in SVG pixels.",
    )
    parser.add_argument(
        "--samples-per-curve",
        type=int,
        default=100,
        help="Number of line segments used to approximate each cubic curve.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scores/bezier_discontinuity_scores.csv"),
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        help=(
            "Optional directory for SVG visualizations of endpoint-to-stroke "
            "distances."
        ),
    )
    parser.add_argument(
        "--visualize-all-endpoints",
        action="store_true",
        help=(
            "Draw distance lines for connected endpoints too. By default, only "
            "disconnected endpoint distances are drawn."
        ),
    )
    return parser.parse_args()


def _numbers(arguments: str) -> list[float]:
    return [float(value) for value in NUMBER_RE.findall(arguments)]


def _line_as_cubic(start: Point, end: Point) -> CubicBezier:
    delta_x = (end[0] - start[0]) / 3.0
    delta_y = (end[1] - start[1]) / 3.0
    return CubicBezier(
        start,
        (start[0] + delta_x, start[1] + delta_y),
        (start[0] + 2.0 * delta_x, start[1] + 2.0 * delta_y),
        end,
    )


def parse_path_data(path_data: str) -> Stroke:
    """Parse the M/C/Z subset emitted by ControlSketch's SVG writer."""
    curves: list[CubicBezier] = []
    current: Point | None = None
    subpath_start: Point | None = None

    unsupported_command = UNSUPPORTED_COMMAND_RE.search(path_data)
    if unsupported_command is not None:
        raise ValueError(
            f"unsupported SVG command {unsupported_command.group()!r}; "
            "expected M, C, or Z"
        )
    commands = COMMAND_RE.findall(path_data)
    if not commands:
        raise ValueError("SVG path has no commands")

    for command, arguments in commands:
        values = _numbers(arguments)
        if command in {"M", "m"}:
            if len(values) != 2:
                raise ValueError(f"{command} requires exactly two coordinates")
            point = values[0], values[1]
            if command == "m" and current is not None:
                point = current[0] + point[0], current[1] + point[1]
            current = point
            subpath_start = point
        elif command in {"C", "c"}:
            if current is None:
                raise ValueError(f"{command} appears before M")
            if len(values) == 0 or len(values) % 6 != 0:
                raise ValueError(f"{command} requires groups of six coordinates")
            for offset in range(0, len(values), 6):
                p1 = values[offset], values[offset + 1]
                p2 = values[offset + 2], values[offset + 3]
                p3 = values[offset + 4], values[offset + 5]
                if command == "c":
                    p1 = current[0] + p1[0], current[1] + p1[1]
                    p2 = current[0] + p2[0], current[1] + p2[1]
                    p3 = current[0] + p3[0], current[1] + p3[1]
                curves.append(CubicBezier(current, p1, p2, p3))
                current = p3
        elif command in {"Z", "z"}:
            if values:
                raise ValueError(f"{command} does not accept coordinates")
            if current is None or subpath_start is None:
                raise ValueError(f"{command} appears before M")
            if current != subpath_start:
                curves.append(_line_as_cubic(current, subpath_start))
            current = subpath_start
        else:
            raise ValueError(
                f"unsupported SVG command {command!r}; expected M, C, or Z"
            )

    if not curves:
        raise ValueError("SVG path contains no drawable curve")
    return Stroke(tuple(curves))


def read_svg_strokes(svg_path: Path) -> list[Stroke]:
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"invalid SVG XML: {error}") from error

    strokes: list[Stroke] = []
    for element in root.iter():
        if element.tag.rsplit("}", maxsplit=1)[-1] != "path":
            continue
        path_data = element.get("d")
        if path_data:
            strokes.append(parse_path_data(path_data))
    if not strokes:
        raise ValueError("SVG contains no path strokes")
    return strokes


def closest_point_on_segment(
    point: Point,
    segment: LineSegment,
) -> tuple[Point, float]:
    start, end = segment
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared == 0.0:
        distance_squared = (
            (point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2
        )
        return start, distance_squared

    projection = (
        (point[0] - start[0]) * delta_x
        + (point[1] - start[1]) * delta_y
    ) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest_x = start[0] + projection * delta_x
    closest_y = start[1] + projection * delta_y
    closest_point = closest_x, closest_y
    distance_squared = (
        (point[0] - closest_x) ** 2 + (point[1] - closest_y) ** 2
    )
    return closest_point, distance_squared


def point_segment_distance_squared(point: Point, segment: LineSegment) -> float:
    return closest_point_on_segment(point, segment)[1]


def nearest_point_on_other_strokes(
    endpoint: Point,
    own_stroke_index: int,
    flattened_strokes: Sequence[Sequence[LineSegment]],
) -> tuple[Point | None, float]:
    nearest_point: Point | None = None
    nearest_distance_squared = math.inf
    for stroke_index, segments in enumerate(flattened_strokes):
        if stroke_index == own_stroke_index:
            continue
        for segment in segments:
            candidate_point, distance_squared = closest_point_on_segment(
                endpoint,
                segment,
            )
            if distance_squared < nearest_distance_squared:
                nearest_point = candidate_point
                nearest_distance_squared = distance_squared
    return nearest_point, math.sqrt(nearest_distance_squared)


def endpoint_is_disconnected(
    endpoint: Point,
    own_stroke_index: int,
    flattened_strokes: Sequence[Sequence[LineSegment]],
    threshold: float,
) -> bool:
    threshold_squared = threshold * threshold
    for stroke_index, segments in enumerate(flattened_strokes):
        if stroke_index == own_stroke_index:
            continue
        for segment in segments:
            if point_segment_distance_squared(endpoint, segment) <= threshold_squared:
                return False
    return True


def evaluate_strokes(
    strokes: Sequence[Stroke],
    threshold: float,
    samples_per_curve: int,
) -> DiscontinuityResult:
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    if samples_per_curve < 1:
        raise ValueError("samples_per_curve must be at least 1")

    flattened = [stroke.flatten(samples_per_curve) for stroke in strokes]
    num_disconnected_strokes = 0
    num_disconnected_endpoints = 0
    for stroke_index, stroke in enumerate(strokes):
        endpoint_states = [
            endpoint_is_disconnected(
                endpoint,
                stroke_index,
                flattened,
                threshold,
            )
            for endpoint in stroke.endpoints
        ]
        num_disconnected_endpoints += sum(endpoint_states)
        if any(endpoint_states):
            num_disconnected_strokes += 1

    return DiscontinuityResult(
        num_strokes=len(strokes),
        num_disconnected_strokes=num_disconnected_strokes,
        num_disconnected_endpoints=num_disconnected_endpoints,
    )


def _svg_number(value: float) -> str:
    return f"{value:.6g}"


def write_distance_visualization(
    source_svg_path: Path,
    strokes: Sequence[Stroke],
    threshold: float,
    samples_per_curve: int,
    output_path: Path,
    visualize_all_endpoints: bool = False,
) -> None:
    """Overlay endpoint-to-nearest-stroke distances on the source SVG."""
    tree = ET.parse(source_svg_path)
    root = tree.getroot()
    namespace = root.tag.partition("}")[0].lstrip("{")
    if namespace:
        ET.register_namespace("", namespace)

    def tag(name: str) -> str:
        return f"{{{namespace}}}{name}" if namespace else name

    flattened = [stroke.flatten(samples_per_curve) for stroke in strokes]
    overlay = ET.SubElement(root, tag("g"), {"id": "endpoint-distance-overlay"})
    for stroke_index, stroke in enumerate(strokes):
        for endpoint in stroke.endpoints:
            nearest_point, distance = nearest_point_on_other_strokes(
                endpoint,
                stroke_index,
                flattened,
            )
            disconnected = distance > threshold
            color = "#d62728" if disconnected else "#2ca02c"
            endpoint_x, endpoint_y = endpoint

            if (
                nearest_point is not None
                and (disconnected or visualize_all_endpoints)
            ):
                nearest_x, nearest_y = nearest_point
                ET.SubElement(
                    overlay,
                    tag("line"),
                    {
                        "x1": _svg_number(endpoint_x),
                        "y1": _svg_number(endpoint_y),
                        "x2": _svg_number(nearest_x),
                        "y2": _svg_number(nearest_y),
                        "stroke": color,
                        "stroke-width": "1.5",
                        "stroke-dasharray": "5 4",
                    },
                )
                ET.SubElement(
                    overlay,
                    tag("circle"),
                    {
                        "cx": _svg_number(nearest_x),
                        "cy": _svg_number(nearest_y),
                        "r": "2.5",
                        "fill": "#1f77b4",
                    },
                )
                label = ET.SubElement(
                    overlay,
                    tag("text"),
                    {
                        "x": _svg_number(
                            (endpoint_x + nearest_x) / 2.0 + 3.0
                        ),
                        "y": _svg_number(
                            (endpoint_y + nearest_y) / 2.0 - 3.0
                        ),
                        "fill": color,
                        "font-size": "10",
                        "font-family": "sans-serif",
                        "paint-order": "stroke",
                        "stroke": "white",
                        "stroke-width": "2",
                    },
                )
                label.text = f"{distance:.1f}"

            ET.SubElement(
                overlay,
                tag("circle"),
                {
                    "cx": _svg_number(endpoint_x),
                    "cy": _svg_number(endpoint_y),
                    "r": "4",
                    "fill": color,
                    "stroke": "white",
                    "stroke-width": "1",
                },
            )

    description = ET.SubElement(overlay, tag("desc"))
    description.text = (
        "Red endpoints are farther than the threshold from every other "
        "stroke; green endpoints are within the threshold. Blue points are "
        "the nearest points on other strokes."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def discover_svgs(input_root: Path, datasets: Iterable[str]) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for dataset in datasets:
        dataset_root = input_root / dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")
        for svg_path in dataset_root.rglob("final_svg.svg"):
            match = RUN_STROKE_COUNT_RE.search(svg_path.parent.name)
            if match and int(match.group(1)) in SUPPORTED_STROKE_COUNTS:
                discovered.append((dataset, svg_path))
    return sorted(discovered, key=lambda item: (item[0], str(item[1])))


def output_row(
    dataset: str,
    svg_path: Path,
    input_root: Path,
    result: DiscontinuityResult,
    threshold: float,
    samples_per_curve: int,
) -> dict[str, object]:
    match = RUN_STROKE_COUNT_RE.search(svg_path.parent.name)
    if match is None:
        raise ValueError(f"cannot determine stroke count from {svg_path.parent.name}")
    expected_stroke_count = int(match.group(1))
    if result.num_strokes != expected_stroke_count:
        print(
            f"warning: {svg_path} contains {result.num_strokes} paths, "
            f"but its directory says {expected_stroke_count} strokes",
            file=sys.stderr,
        )

    relative_path = svg_path.relative_to(input_root)
    return {
        "dataset": dataset,
        "category": relative_path.parts[1],
        "run": svg_path.parent.name,
        "stroke_count": expected_stroke_count,
        "num_svg_strokes": result.num_strokes,
        "num_disconnected_strokes": result.num_disconnected_strokes,
        "disconnected_stroke_ratio": result.disconnected_stroke_ratio,
        "disconnected_stroke_percentage": 100.0
        * result.disconnected_stroke_ratio,
        "num_disconnected_endpoints": result.num_disconnected_endpoints,
        "disconnected_endpoint_ratio": result.disconnected_endpoint_ratio,
        "threshold": threshold,
        "samples_per_curve": samples_per_curve,
        "svg_path": str(relative_path),
    }


def write_results(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("no SVG evaluation results to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.threshold < 0.0:
        raise SystemExit("--threshold must be non-negative")
    if args.samples_per_curve < 1:
        raise SystemExit("--samples-per-curve must be at least 1")

    svg_paths = discover_svgs(args.input_root, args.datasets)
    if not svg_paths:
        raise SystemExit("no final_svg.svg files were found")

    rows: list[dict[str, object]] = []
    for dataset, svg_path in svg_paths:
        try:
            strokes = read_svg_strokes(svg_path)
            result = evaluate_strokes(
                strokes,
                threshold=args.threshold,
                samples_per_curve=args.samples_per_curve,
            )
        except ValueError as error:
            raise SystemExit(f"failed to evaluate {svg_path}: {error}") from error
        rows.append(
            output_row(
                dataset,
                svg_path,
                args.input_root,
                result,
                args.threshold,
                args.samples_per_curve,
            )
        )
        if args.visualization_dir is not None:
            relative_directory = svg_path.parent.relative_to(args.input_root)
            visualization_path = (
                args.visualization_dir
                / relative_directory
                / "endpoint_distances.svg"
            )
            write_distance_visualization(
                svg_path,
                strokes,
                args.threshold,
                args.samples_per_curve,
                visualization_path,
                args.visualize_all_endpoints,
            )

    write_results(rows, args.output)
    print(f"Evaluated {len(rows)} SVG files and wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
