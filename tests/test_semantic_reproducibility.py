import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONTROL_SKETCH = ROOT / "ControlSketch"
sys.path.insert(0, str(CONTROL_SKETCH))

import semantic_pipeline
from semantic_init import allocate
from semantic_part_enumerator import (
    SemanticComponent,
    SemanticPartEnumeration,
    enumerate_semantic_parts,
    parse_components,
)
from semantic_reference_cache import ReferenceCache, compute_cache_hash


class SemanticSchemaTests(unittest.TestCase):
    def test_weights_are_normalized_even_when_not_100(self):
        parts = parse_components(json.dumps({"parts": [
            {"name": "wheels", "phrase": "wheel", "count": 2, "weight": 20},
            {"name": "seat", "phrase": "seat", "count": 1, "weight": 30},
        ]}), max_parts=8)
        self.assertAlmostEqual(sum(part.weight for part in parts), 1.0)
        self.assertAlmostEqual(parts[0].weight, 0.4)
        self.assertEqual(parts[0].count, 2)

    def test_zero_weight_sum_falls_back_to_uniform(self):
        parts = parse_components(json.dumps({"parts": [
            {"name": "ears", "phrase": "ears", "weight": 0},
            {"name": "tail", "phrase": "tail", "weight": 0},
        ]}), max_parts=8)
        self.assertEqual([part.weight for part in parts], [0.5, 0.5])

    def test_max_parts_and_invalid_fields(self):
        payload = {"parts": [
            {"name": f"part{i}", "phrase": f"part {i}", "count": 0,
             "weight": "bad" if i == 0 else 1}
            for i in range(12)
        ]}
        with self.assertWarns(UserWarning):
            parts = parse_components(json.dumps(payload), max_parts=8)
        self.assertEqual(len(parts), 8)
        self.assertEqual(parts[0].count, 1)
        self.assertEqual(parts[0].weight, 0.0)
        self.assertAlmostEqual(sum(part.weight for part in parts), 1.0)

    def test_empty_and_broken_json_retry_then_fallback(self):
        for raw in ('{"parts":[]}', "not json"):
            calls = []

            def runner(**kwargs):
                calls.append(kwargs)
                return raw

            result = enumerate_semantic_parts(None, runner=runner)
            self.assertTrue(result.fallback)
            self.assertTrue(result.parse_failed)
            self.assertEqual(result.parts_text, "outline")
            self.assertEqual(len(calls), 2)

    def test_same_seed_produces_same_parsed_parts_three_times(self):
        def runner(seed, **kwargs):
            generator = random.Random(seed)
            weight = generator.randint(20, 80)
            return json.dumps({"parts": [
                {"name": "ears", "phrase": "animal ears", "count": 2,
                 "weight": weight},
                {"name": "tail", "phrase": "animal tail", "count": 1,
                 "weight": 100 - weight},
            ]})

        runs = [enumerate_semantic_parts(None, seed=7, runner=runner) for _ in range(3)]
        signatures = [
            [(part.name, part.weight) for part in run.components]
            for run in runs
        ]
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])

    def test_missing_phrase_falls_back_to_name(self):
        parts = parse_components(json.dumps({"parts": [
            {"name": "ears", "weight": 100},
        ]}), max_parts=8)
        self.assertEqual(parts[0].name, "ears")
        self.assertEqual(parts[0].phrase, "ears")

    def test_new_top_level_array_is_supported(self):
        parts = parse_components(json.dumps([
            {"name": "ears", "phrase": "rabbit ears", "weight": 60},
            {"name": "face", "phrase": "rabbit face", "weight": 40},
        ]), max_parts=8)
        self.assertEqual([part.name for part in parts], ["ears", "face"])

    def test_legacy_importance_confidence_cache_is_supported(self):
        parts = parse_components(json.dumps({"components": [
            {"name": "ear", "grounding_phrase": "ear", "importance": 4,
             "confidence": 0.5}
        ]}), max_parts=8)
        self.assertEqual(parts[0].name, "ear")
        self.assertEqual(parts[0].weight, 1.0)
        cached = SemanticPartEnumeration.from_cache({"parts": [{
            "name": "ear", "importance": 4, "confidence": 0.5
        }]})
        self.assertEqual(cached.components[0].phrase, "ear")
        self.assertEqual(cached.components[0].weight, 1.0)


class AllocationTests(unittest.TestCase):
    def test_random_weights_exact_budget_and_lower_bound(self):
        generator = np.random.default_rng(123)
        for _ in range(1000):
            size = int(generator.integers(1, 17))
            weights = generator.random(size)
            weights /= weights.sum()
            n_total = int(generator.integers(max(size, 1), 65))
            result = allocate(weights, n_total)
            self.assertEqual(int(result.sum()), n_total)
            self.assertTrue(np.all(result[weights >= 0.05] >= 1))

    def test_insufficient_budget_prioritizes_largest_weights(self):
        result = allocate([0.4, 0.3, 0.2, 0.1], 2)
        np.testing.assert_array_equal(result, [1, 1, 0, 0])


class ReferenceCacheTests(unittest.TestCase):
    def _args(self, root, image_path, rebuild=False):
        output_dir = Path(root) / "run"
        output_dir.mkdir(exist_ok=True)
        return types.SimpleNamespace(
            target=str(image_path), cache_dir=str(Path(root) / "cache"),
            output_dir=str(output_dir), object_name="cat", caption="a cat",
            semantic_vlm_model="fake/qwen2.5-vl", max_parts=8, vlm_seed=0,
            vlm_temperature=0.3, rebuild_reference=rebuild,
            semantic_segmenter="sam3", sam3_python="fake-python",
            sam3_checkpoint_path="", sam3_confidence_threshold=0.5,
            semantic_min_area_ratio=0.0, semantic_max_masks_per_part=4,
            semantic_max_part_area_ratio=0.9,
        )

    def test_second_build_skips_vlm_and_sam_and_rebuild_bypasses_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cat.png"
            image = Image.new("RGB", (8, 8), "white")
            image.save(image_path)
            args = self._args(temp_dir, image_path)
            enumeration = SemanticPartEnumeration.from_components([
                SemanticComponent("ears", "cat ears", 2, 0.6),
                SemanticComponent("tail", "cat tail", 1, 0.4),
            ], raw_text='{"parts":[]}')
            segmenter = mock.Mock()
            segmenter.segment_parts.return_value = {
                "ears": np.ones((8, 8), dtype=np.uint8)
            }
            with mock.patch.object(
                semantic_pipeline, "enumerate_semantic_parts",
                return_value=enumeration,
            ) as vlm_call, mock.patch.object(
                semantic_pipeline, "_make_segmenter", return_value=segmenter
            ) as sam_factory:
                first = semantic_pipeline.prepare_auto_semantic_data(
                    args, image, np.ones((8, 8)), "cpu"
                )
                second = semantic_pipeline.prepare_auto_semantic_data(
                    args, image, np.ones((8, 8)), "cpu"
                )
                self.assertEqual(vlm_call.call_count, 1)
                self.assertEqual(sam_factory.call_count, 1)
                self.assertEqual(segmenter.segment_parts.call_count, 1)
                self.assertFalse(first[3]["cache_hit"])
                self.assertTrue(second[3]["cache_hit"])
                self.assertEqual(first[3]["sam_dropped_parts"], ["tail"])
                self.assertEqual(first[3]["n_parts_returned"], 2)
                self.assertEqual(first[3]["parts"][0]["weight"], 1.0)

                args.rebuild_reference = True
                semantic_pipeline.prepare_auto_semantic_data(
                    args, image, np.ones((8, 8)), "cpu"
                )
                self.assertEqual(vlm_call.call_count, 2)
                self.assertEqual(segmenter.segment_parts.call_count, 2)

    def test_reference_roundtrip_is_bitwise_identical(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ReferenceCache(temp_dir, "abc")
            reference = {
                "edge_pts": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                "edge_w": torch.tensor([0.75], dtype=torch.float32),
                "D_E": torch.arange(9, dtype=torch.float32).reshape(3, 3),
                "e_hat": torch.arange(18, dtype=torch.float32).reshape(3, 3, 2),
                "edge_mask": np.eye(3, dtype=bool),
                "size": (3, 3),
            }
            cache.save_reference(reference)
            loaded = cache.load_reference("cpu")
            for key in ("edge_pts", "edge_w", "D_E", "e_hat"):
                self.assertTrue(torch.equal(reference[key], loaded[key]))
            self.assertTrue(np.array_equal(reference["edge_mask"], loaded["edge_mask"]))
            self.assertEqual(reference["size"], loaded["size"])

    def test_hash_uses_image_contents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "image.bin"
            path.write_bytes(b"first")
            first = compute_cache_hash(path, "prompt", "model", 8, 0, 0.3)
            path.write_bytes(b"second")
            second = compute_cache_hash(path, "prompt", "model", 8, 0, 0.3)
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
