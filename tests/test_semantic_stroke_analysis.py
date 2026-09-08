import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch


CONTROL_SKETCH_DIR = Path(__file__).resolve().parents[1] / "ControlSketch"
sys.path.insert(0, str(CONTROL_SKETCH_DIR))

import semantic_init  # noqa: E402
from semantic_stroke_analysis import (  # noqa: E402
    SemanticStrokeAnalyzer,
    analysis_checkpoint_steps,
    distances_to_contours,
    sample_bezier_path,
)


class FakePath:
    def __init__(self, points):
        self.points = torch.as_tensor(np.asarray(points), dtype=torch.float32)
        self.num_control_points = torch.tensor([2], dtype=torch.int32)


class SemanticStrokeAnalysisTests(unittest.TestCase):
    def test_checkpoint_steps(self):
        self.assertEqual(
            analysis_checkpoint_steps(500),
            [0, 125, 250, 375, 500],
        )

    def test_bezier_sampling_and_contour_distance(self):
        sampled = sample_bezier_path(
            [[0, 0], [3, 0], [7, 0], [10, 0]],
            [2],
            num_samples=100,
        )
        contour = [{
            "points": np.array([[0, 2], [10, 2]]),
            "closed": False,
        }]
        distances = distances_to_contours(sampled, contour)
        self.assertEqual(sampled.shape, (100, 2))
        np.testing.assert_allclose(distances, 2.0, atol=1e-5)

    def test_sixteen_strokes_write_five_checkpoints_and_visualization(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(mask, (32, 32), 24, 1, thickness=-1)
        result = semantic_init.build_semantic_initial_points(
            mask=mask,
            total_points=16,
            canvas_width=64,
            canvas_height=64,
            parts_text="outline",
            weights_text="",
            return_metadata=True,
            return_analysis_contours=True,
        )
        points, _, parts, _, contours = result
        shapes = [
            FakePath([
                point,
                point + [1.0, 0.0],
                point + [2.0, 0.0],
                point + [3.0, 0.0],
            ])
            for point in points
        ]

        analyzer = SemanticStrokeAnalyzer(
            shapes,
            parts,
            contours,
            canvas_width=64,
            canvas_height=64,
        )
        for step in analysis_checkpoint_steps(500):
            analyzer.capture(step, shapes)

        with tempfile.TemporaryDirectory() as directory:
            csv_path, image_path = analyzer.save(directory)
            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 16 * 5)
            self.assertEqual(rows[0]["stroke_id"], "stroke_0")
            self.assertEqual(rows[-1]["step"], "500")
            self.assertAlmostEqual(
                float(rows[0]["length_ratio"]), 1.0, places=5
            )
            self.assertTrue(image_path.is_file())
            self.assertIsNotNone(cv2.imread(str(image_path)))

    def test_stroke_reordering_is_rejected(self):
        shapes = [
            FakePath([[0, 0], [1, 0], [2, 0], [3, 0]]),
            FakePath([[0, 1], [1, 1], [2, 1], [3, 1]]),
        ]
        contours = {"outline": [{
            "points": np.array([[0, 0], [3, 0]], dtype=np.float32),
            "closed": False,
        }]}
        analyzer = SemanticStrokeAnalyzer(
            shapes,
            ["outline", "outline"],
            contours,
            8,
            8,
        )
        with self.assertRaises(RuntimeError):
            analyzer.capture(1, list(reversed(shapes)))


if __name__ == "__main__":
    unittest.main()
