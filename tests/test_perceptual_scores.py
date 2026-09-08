from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from compute_controlsketch_perceptual_scores import (
    METRIC_DREAMSIM,
    METRIC_MS_SSIM,
    summarize_results,
)
from image_similarity_metrics import XDoGConfig, _chunks, resize_image


class XDoGConfigTests(unittest.TestCase):
    def test_defaults_are_serializable(self) -> None:
        config = XDoGConfig()
        config.validate()
        self.assertEqual(config.to_dict()["sigma"], 0.5)
        self.assertEqual(config.to_dict()["phi"], 200.0)

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sigma"):
            XDoGConfig(sigma=0).validate()
        with self.assertRaisesRegex(ValueError, "k"):
            XDoGConfig(k=1).validate()


class ResizeTests(unittest.TestCase):
    def test_stretch_produces_requested_square(self) -> None:
        image = Image.new("RGB", (20, 10), "black")
        self.assertEqual(resize_image(image, 32, "stretch").size, (32, 32))

    def test_letterbox_uses_white_background(self) -> None:
        image = Image.new("RGB", (20, 10), "black")
        resized = resize_image(image, 32, "letterbox")
        self.assertEqual(resized.size, (32, 32))
        self.assertEqual(resized.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(resized.getpixel((16, 16)), (0, 0, 0))

    def test_unknown_resize_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported resize mode"):
            resize_image(Image.new("RGB", (1, 1)), 32, "crop")


class BatchTests(unittest.TestCase):
    def test_chunks_preserve_order(self) -> None:
        self.assertEqual(list(_chunks([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])

    def test_invalid_batch_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Batch size"):
            list(_chunks([1], 0))


class SummaryTests(unittest.TestCase):
    def test_summary_groups_by_category_and_stroke_count(self) -> None:
        results = [
            {
                "category": "cat",
                "num_strokes": 20,
                "dreamsim_distance": 0.2,
                "ms_ssim": 0.8,
            },
            {
                "category": "cat",
                "num_strokes": 40,
                "dreamsim_distance": 0.4,
                "ms_ssim": 0.6,
            },
            {
                "category": "dog",
                "num_strokes": 20,
                "dreamsim_distance": 0.3,
                "ms_ssim": 0.7,
            },
        ]
        summary = summarize_results(
            results,
            [METRIC_DREAMSIM, METRIC_MS_SSIM],
        )
        self.assertAlmostEqual(summary["mean_dreamsim_distance"], 0.3)
        self.assertAlmostEqual(summary["mean_ms_ssim"], 0.7)
        self.assertEqual(summary["by_category"]["cat"]["count"], 2)
        self.assertEqual(summary["by_num_strokes"]["20"]["count"], 2)

    def test_empty_summary_uses_null_means(self) -> None:
        summary = summarize_results([], [METRIC_DREAMSIM, METRIC_MS_SSIM])
        self.assertEqual(summary["num_scored"], 0)
        self.assertIsNone(summary["mean_dreamsim_distance"])
        self.assertIsNone(summary["mean_ms_ssim"])


class ImageLoadingSmokeTest(unittest.TestCase):
    def test_temporary_directory_is_available_for_integration_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(path)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
