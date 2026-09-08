import csv
import tempfile
import unittest
from pathlib import Path

from plot_recognition_vs_discontinuity import join_scores, split_category


class RecognitionDiscontinuityPlotTest(unittest.TestCase):
    def test_split_category_preserves_underscores_in_class(self) -> None:
        self.assertEqual(split_category("fire_truck_12"), ("fire_truck", 12))

    def test_join_scores_uses_method_class_sample_and_strokes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_root = root / "eval"
            for subdirectory in ("normal", "semantic", "semantic_overlap"):
                score_path = eval_root / subdirectory / "scores.csv"
                score_path.parent.mkdir(parents=True, exist_ok=True)
                with score_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=(
                            "class_name",
                            "sample_id",
                            "stroke_count",
                            "recognition_score",
                        ),
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "class_name": "airplane",
                            "sample_id": 3,
                            "stroke_count": 16,
                            "recognition_score": 0.75,
                        }
                    )

            discontinuity_path = root / "discontinuity.csv"
            with discontinuity_path.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=(
                        "dataset",
                        "category",
                        "stroke_count",
                        "disconnected_stroke_ratio",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "SDXL_normal",
                        "category": "airplane_3",
                        "stroke_count": 16,
                        "disconnected_stroke_ratio": 0.625,
                    }
                )

            records = join_scores(discontinuity_path, eval_root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].method, "org")
        self.assertEqual(records[0].recognition_score, 0.75)
        self.assertEqual(records[0].disconnected_stroke_ratio, 0.625)


if __name__ == "__main__":
    unittest.main()
