import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from semantic_colors import add_semantic_legend, semantic_part_bgr


CSV_FIELDS = [
    "step",
    "stroke_id",
    "assigned_part_name",
    "part_distance_mean_px",
    "retention_4px",
    "initial_length_px",
    "stroke_length_px",
    "length_ratio",
]


def analysis_checkpoint_steps(num_iter):
    """Return the five requested progress points, without duplicates."""
    return sorted({
        0,
        int(round(num_iter * 0.25)),
        int(round(num_iter * 0.50)),
        int(round(num_iter * 0.75)),
        int(num_iter),
    })


def _evaluate_bezier(control_points, t_values):
    curve = np.broadcast_to(
        control_points[None, :, :],
        (len(t_values), len(control_points), 2),
    ).copy()
    t = t_values[:, None, None]
    for _ in range(len(control_points) - 1):
        curve = (1.0 - t) * curve[:, :-1] + t * curve[:, 1:]
    return curve[:, 0]


def sample_bezier_path(control_points, num_control_points, num_samples=100):
    """Sample a pydiffvg path approximately uniformly by curve arc length."""
    control_points = np.asarray(control_points, dtype=np.float64)
    num_control_points = np.asarray(num_control_points, dtype=np.int64)
    if num_samples <= 0:
        return np.empty((0, 2), dtype=np.float64)

    dense_segments = []
    point_offset = 0
    segment_count = max(len(num_control_points), 1)
    dense_per_segment = max(
        32,
        int(math.ceil(num_samples * 4 / segment_count)),
    )
    for segment_index, n_ctrl in enumerate(num_control_points):
        segment = control_points[
            point_offset:point_offset + int(n_ctrl) + 2
        ]
        t_values = np.linspace(0.0, 1.0, dense_per_segment, endpoint=True)
        sampled = _evaluate_bezier(segment, t_values)
        if segment_index:
            sampled = sampled[1:]
        dense_segments.append(sampled)
        point_offset += int(n_ctrl) + 1

    if not dense_segments:
        if len(control_points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.repeat(control_points[:1], num_samples, axis=0)

    dense = np.concatenate(dense_segments, axis=0)
    segment_lengths = np.linalg.norm(dense[1:] - dense[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    if total_length <= 1e-12:
        return np.repeat(dense[:1], num_samples, axis=0)

    targets = np.linspace(0.0, total_length, num_samples, endpoint=True)
    x = np.interp(targets, cumulative, dense[:, 0])
    y = np.interp(targets, cumulative, dense[:, 1])
    return np.column_stack([x, y])


def polyline_length(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def distances_to_contours(points, contours):
    """Compute exact point-to-polyline nearest distances."""
    points = np.asarray(points, dtype=np.float64)
    minimum = np.full(len(points), np.inf, dtype=np.float64)

    for contour in contours:
        contour_points = np.asarray(contour["points"], dtype=np.float64)
        if len(contour_points) == 0:
            continue
        if len(contour_points) == 1:
            minimum = np.minimum(
                minimum,
                np.linalg.norm(points - contour_points[0], axis=1),
            )
            continue

        if contour.get("closed", True):
            contour_points = np.vstack([contour_points, contour_points[0]])
        starts = contour_points[:-1]
        vectors = contour_points[1:] - starts
        squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
        valid = squared_lengths > 1e-12
        starts = starts[valid]
        vectors = vectors[valid]
        squared_lengths = squared_lengths[valid]
        if not len(starts):
            continue

        offsets = points[:, None, :] - starts[None, :, :]
        projections = np.einsum(
            "msd,sd->ms", offsets, vectors
        ) / squared_lengths[None, :]
        projections = np.clip(projections, 0.0, 1.0)
        nearest = starts[None, :, :] + projections[:, :, None] * vectors
        distances = np.linalg.norm(points[:, None, :] - nearest, axis=2)
        minimum = np.minimum(minimum, distances.min(axis=1))

    if not np.all(np.isfinite(minimum)):
        raise ValueError("Assigned part has no usable contour")
    return minimum


def _path_arrays(path):
    points = path.points.detach().cpu().numpy()
    num_control_points = path.num_control_points.detach().cpu().numpy()
    return points, num_control_points


class SemanticStrokeAnalyzer:
    def __init__(
        self,
        shapes,
        assigned_parts,
        part_contours,
        canvas_width,
        canvas_height,
        num_samples=100,
    ):
        if len(shapes) != len(assigned_parts):
            raise ValueError("Each stroke must have one assigned semantic part")

        self.assigned_parts = list(assigned_parts)
        self.part_contours = part_contours
        self.canvas_width = int(canvas_width)
        self.canvas_height = int(canvas_height)
        self.num_samples = int(num_samples)
        self._shape_ids = tuple(id(shape) for shape in shapes)
        self.initial_strokes = self._sample_shapes(shapes)
        self.initial_lengths = [
            polyline_length(points) for points in self.initial_strokes
        ]
        self.rows_by_step = {}
        self.final_strokes = self.initial_strokes

    def _check_stroke_order(self, shapes):
        if tuple(id(shape) for shape in shapes) != self._shape_ids:
            raise RuntimeError(
                "Stroke order changed during semantic analysis; stroke_id tracking "
                "is no longer valid"
            )

    def _sample_shapes(self, shapes):
        sampled = []
        for path in shapes:
            points, num_control_points = _path_arrays(path)
            sampled.append(sample_bezier_path(
                points,
                num_control_points,
                num_samples=self.num_samples,
            ))
        return sampled

    def capture(self, step, shapes):
        self._check_stroke_order(shapes)
        current_strokes = self._sample_shapes(shapes)
        rows = []
        for stroke_index, (part, stroke_points) in enumerate(zip(
            self.assigned_parts, current_strokes
        )):
            distances = distances_to_contours(
                stroke_points,
                self.part_contours.get(part, []),
            )
            stroke_length = polyline_length(stroke_points)
            initial_length = self.initial_lengths[stroke_index]
            length_ratio = (
                stroke_length / initial_length
                if initial_length > 1e-12 else float("nan")
            )
            rows.append({
                "step": int(step),
                "stroke_id": f"stroke_{stroke_index}",
                "assigned_part_name": part,
                "part_distance_mean_px": float(distances.mean()),
                "retention_4px": float(np.mean(distances <= 4.0)),
                "initial_length_px": initial_length,
                "stroke_length_px": stroke_length,
                "length_ratio": length_ratio,
            })
        self.rows_by_step[int(step)] = rows
        self.final_strokes = current_strokes

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "semantic_stroke_analysis.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for step in sorted(self.rows_by_step):
                writer.writerows(self.rows_by_step[step])

        visualization_path = output_dir / "semantic_stroke_analysis.png"
        self._save_visualization(visualization_path)
        return csv_path, visualization_path

    def _save_visualization(self, output_path):
        image = np.full(
            (self.canvas_height, self.canvas_width, 3),
            255,
            dtype=np.uint8,
        )

        for part in set(self.assigned_parts):
            for contour in self.part_contours.get(part, []):
                points = np.round(contour["points"]).astype(np.int32)
                if len(points) >= 2:
                    cv2.polylines(
                        image,
                        [points.reshape(-1, 1, 2)],
                        bool(contour.get("closed", True)),
                        (80, 180, 80),
                        1,
                        cv2.LINE_AA,
                    )

        for stroke in self.initial_strokes:
            points = np.round(stroke).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [points], False, (220, 110, 40), 1, cv2.LINE_AA)

        for stroke_index, (part, stroke) in enumerate(zip(
            self.assigned_parts, self.final_strokes
        )):
            points = np.round(stroke).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                image,
                [points],
                False,
                semantic_part_bgr(part),
                2,
                cv2.LINE_AA,
            )
            label_point = tuple(np.round(stroke[len(stroke) // 2]).astype(int))
            cv2.putText(
                image,
                f"stroke_{stroke_index}:{part}",
                label_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(image, "contour", (8, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (80, 180, 80), 1, cv2.LINE_AA)
        cv2.putText(image, "initial", (8, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220, 110, 40), 1, cv2.LINE_AA)
        cv2.putText(image, "final: colored by part", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1,
                    cv2.LINE_AA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.asarray(add_semantic_legend(
            Image.fromarray(image),
            self.assigned_parts,
        ))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Failed to write visualization: {output_path}")
