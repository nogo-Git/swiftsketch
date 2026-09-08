from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_existing_controlsketch_inputs import (
    align_input_image,
    prepare_run,
)


class AlignInputImageTests(unittest.TestCase):
    def test_inverse_transform_restores_original_center_and_scale(self) -> None:
        source = Image.new("RGB", (100, 100), "white")
        ImageDraw.Draw(source).rectangle((45, 45, 54, 54), fill="black")
        config = {
            "scale_w": 0.5,
            "scale_h": 0.5,
            "original_center_x": 0.25,
            "original_center_y": 0.75,
        }

        aligned, transformed = align_input_image(source, (200, 200), config)

        self.assertTrue(transformed)
        self.assertEqual(aligned.size, (200, 200))
        self.assertLess(sum(aligned.getpixel((50, 150))), 30)
        self.assertGreater(sum(aligned.getpixel((100, 100))), 700)

    def test_missing_transform_metadata_only_resizes(self) -> None:
        source = Image.new("RGB", (20, 10), "black")

        aligned, transformed = align_input_image(source, (40, 40), {})

        self.assertFalse(transformed)
        self.assertEqual(aligned.size, (40, 40))

    def test_invalid_scale_is_rejected(self) -> None:
        config = {
            "scale_w": 0,
            "scale_h": 1,
            "original_center_x": 0.5,
            "original_center_y": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "positive"):
            align_input_image(Image.new("RGB", (10, 10)), (10, 10), config)


class PrepareRunTests(unittest.TestCase):
    def test_existing_input_is_backed_up_and_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "cat" / "cat_16_strokes"
            run_dir.mkdir(parents=True)
            Image.new("RGB", (50, 50), "white").save(run_dir / "input.png")
            Image.new("RGB", (100, 100), "white").save(
                run_dir / "final_sketch.png"
            )
            np.save(
                run_dir / "config.npy",
                {
                    "scale_w": 0.8,
                    "scale_h": 0.8,
                    "original_center_x": 0.4,
                    "original_center_y": 0.6,
                },
            )

            result = prepare_run(run_dir, dry_run=False, force=False)

            self.assertEqual(result.status, "prepared")
            self.assertTrue(result.transformed)
            with Image.open(run_dir / "optimization_input.png") as backup:
                self.assertEqual(backup.size, (50, 50))
            with Image.open(run_dir / "input.png") as aligned:
                self.assertEqual(aligned.size, (100, 100))

            skipped = prepare_run(run_dir, dry_run=False, force=False)
            self.assertEqual(skipped.status, "skipped")

            regenerated = prepare_run(run_dir, dry_run=False, force=True)
            self.assertEqual(regenerated.status, "prepared")


if __name__ == "__main__":
    unittest.main()
