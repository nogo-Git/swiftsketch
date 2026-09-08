from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from compute_controlsketch_clip_scores import discover_pairs


class LocalInputPairingTests(unittest.TestCase):
    def test_adjacent_input_png_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "sketches" / "rabbit" / "rabbit_32_strokes"
            run_dir.mkdir(parents=True)
            sketch = run_dir / "final_sketch.png"
            local_input = run_dir / "input.png"
            external_input = root / "inputs" / "rabbit.png"
            external_input.parent.mkdir()
            sketch.touch()
            local_input.touch()
            external_input.touch()

            pairs, errors = discover_pairs(
                sketch_root=root / "sketches",
                input_index={"rabbit": external_input},
                sketch_glob="*/*_strokes/final_sketch.png",
                fail_on_missing_input=False,
            )

            self.assertEqual(errors, [])
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].input_image, local_input)

    def test_external_input_is_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "sketches" / "rabbit" / "rabbit_32_strokes"
            run_dir.mkdir(parents=True)
            sketch = run_dir / "final_sketch.png"
            external_input = root / "inputs" / "rabbit.png"
            external_input.parent.mkdir()
            sketch.touch()
            external_input.touch()

            pairs, errors = discover_pairs(
                sketch_root=root / "sketches",
                input_index={"rabbit": external_input},
                sketch_glob="*/*_strokes/final_sketch.png",
                fail_on_missing_input=False,
            )

            self.assertEqual(errors, [])
            self.assertEqual(pairs[0].input_image, external_input)


if __name__ == "__main__":
    unittest.main()
