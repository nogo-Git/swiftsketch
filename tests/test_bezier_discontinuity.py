import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from compute_bezier_discontinuity import (
    CubicBezier,
    Stroke,
    evaluate_strokes,
    nearest_point_on_other_strokes,
    parse_path_data,
    read_svg_strokes,
    write_distance_visualization,
)


def line_stroke(start: tuple[float, float], end: tuple[float, float]) -> Stroke:
    delta_x = (end[0] - start[0]) / 3.0
    delta_y = (end[1] - start[1]) / 3.0
    return Stroke(
        (
            CubicBezier(
                start,
                (start[0] + delta_x, start[1] + delta_y),
                (start[0] + 2.0 * delta_x, start[1] + 2.0 * delta_y),
                end,
            ),
        )
    )


class BezierDiscontinuityTest(unittest.TestCase):
    def test_parse_controlsketch_path(self) -> None:
        stroke = parse_path_data("M 0 0 C 1 0 2 0 3 0")

        self.assertEqual(stroke.endpoints, ((0.0, 0.0), (3.0, 0.0)))
        self.assertEqual(len(stroke.curves), 1)

    def test_stroke_is_disconnected_when_either_endpoint_is_far(self) -> None:
        strokes = [
            line_stroke((0.0, 0.0), (10.0, 0.0)),
            line_stroke((10.0, 0.0), (20.0, 0.0)),
        ]

        result = evaluate_strokes(strokes, threshold=0.1, samples_per_curve=10)

        self.assertEqual(result.num_disconnected_strokes, 2)
        self.assertEqual(result.num_disconnected_endpoints, 2)
        self.assertEqual(result.disconnected_stroke_ratio, 1.0)
        self.assertEqual(result.disconnected_endpoint_ratio, 0.5)

    def test_closed_chain_has_no_disconnected_strokes(self) -> None:
        strokes = [
            line_stroke((0.0, 0.0), (10.0, 0.0)),
            line_stroke((10.0, 0.0), (10.0, 10.0)),
            line_stroke((10.0, 10.0), (0.0, 10.0)),
            line_stroke((0.0, 10.0), (0.0, 0.0)),
        ]

        result = evaluate_strokes(strokes, threshold=0.1, samples_per_curve=10)

        self.assertEqual(result.num_disconnected_strokes, 0)
        self.assertEqual(result.num_disconnected_endpoints, 0)

    def test_threshold_is_inclusive(self) -> None:
        strokes = [
            line_stroke((0.0, 0.0), (10.0, 0.0)),
            line_stroke((11.0, 0.0), (20.0, 0.0)),
        ]

        result = evaluate_strokes(strokes, threshold=1.0, samples_per_curve=10)

        self.assertEqual(result.num_disconnected_endpoints, 2)

    def test_nearest_point_on_other_strokes(self) -> None:
        strokes = [
            line_stroke((0.0, 0.0), (10.0, 0.0)),
            line_stroke((11.0, 0.0), (20.0, 0.0)),
        ]
        flattened = [stroke.flatten(10) for stroke in strokes]

        nearest_point, distance = nearest_point_on_other_strokes(
            (10.0, 0.0),
            own_stroke_index=0,
            flattened_strokes=flattened,
        )

        self.assertEqual(nearest_point, (11.0, 0.0))
        self.assertAlmostEqual(distance, 1.0)

    def test_read_svg_strokes(self) -> None:
        svg = """\
<svg xmlns="http://www.w3.org/2000/svg">
  <path d="M 0 0 C 1 0 2 0 3 0" />
  <path d="M 3 0 C 4 0 5 0 6 0" />
</svg>
"""
        with tempfile.TemporaryDirectory() as directory:
            svg_path = Path(directory) / "final_svg.svg"
            svg_path.write_text(svg, encoding="utf-8")

            strokes = read_svg_strokes(svg_path)

        self.assertEqual(len(strokes), 2)

    def test_write_distance_visualization(self) -> None:
        svg = """\
<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32">
  <path d="M 0 0 C 3 0 7 0 10 0" />
  <path d="M 11 0 C 14 0 17 0 20 0" />
</svg>
"""
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "final_svg.svg"
            output_path = Path(directory) / "endpoint_distances.svg"
            source_path.write_text(svg, encoding="utf-8")
            strokes = read_svg_strokes(source_path)

            write_distance_visualization(
                source_path,
                strokes,
                threshold=1.0,
                samples_per_curve=10,
                output_path=output_path,
            )
            output_root = ET.parse(output_path).getroot()

        namespace = {"svg": "http://www.w3.org/2000/svg"}
        self.assertEqual(len(output_root.findall(".//svg:line", namespace)), 2)
        self.assertEqual(len(output_root.findall(".//svg:path", namespace)), 2)


if __name__ == "__main__":
    unittest.main()
