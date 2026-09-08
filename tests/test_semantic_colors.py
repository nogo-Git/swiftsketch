import sys
import unittest
from pathlib import Path

from PIL import Image


CONTROL_SKETCH_DIR = Path(__file__).resolve().parents[1] / "ControlSketch"
sys.path.insert(0, str(CONTROL_SKETCH_DIR))

from semantic_colors import (  # noqa: E402
    add_semantic_legend,
    semantic_part_bgr,
    semantic_part_rgba,
)


class SemanticColorsTests(unittest.TestCase):
    def test_singular_and_plural_aliases_share_colors(self):
        self.assertEqual(semantic_part_rgba("eye"), semantic_part_rgba("eyes"))
        self.assertEqual(semantic_part_rgba("paw"), semantic_part_rgba("paws"))

    def test_known_parts_have_distinct_colors(self):
        parts = ["outline", "eyes", "nose", "ears", "chest", "paws"]
        self.assertEqual(len({semantic_part_rgba(part) for part in parts}), 6)

    def test_unknown_part_color_is_stable(self):
        self.assertEqual(
            semantic_part_rgba("stripe"),
            semantic_part_rgba("stripe"),
        )

    def test_bgr_color_uses_byte_range(self):
        color = semantic_part_bgr("eyes")
        self.assertEqual(len(color), 3)
        self.assertTrue(all(0 <= channel <= 255 for channel in color))

    def test_legend_is_appended_to_right_edge(self):
        source = Image.new("RGB", (32, 24), "white")
        result = add_semantic_legend(
            source,
            ["outline", "eyes", "eyes", "paws"],
            panel_width=120,
        )

        self.assertEqual(result.width, 152)
        self.assertGreaterEqual(result.height, source.height)
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
        self.assertNotEqual(result.getpixel((44, 40)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
