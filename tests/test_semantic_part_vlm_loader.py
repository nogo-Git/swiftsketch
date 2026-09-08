import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CONTROL_SKETCH = ROOT / "ControlSketch"
sys.path.insert(0, str(CONTROL_SKETCH))

import semantic_part_enumerator as enumerator


class _FakeModel:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        model = cls()
        model.model_id = model_id
        model.kwargs = kwargs
        model.moved_to = None
        cls.calls.append(model)
        return model

    def to(self, device):
        self.moved_to = device
        return self


class _FakeProcessor:
    calls = []

    def __init__(self):
        self.tokenizer = types.SimpleNamespace(padding_side="right")

    @classmethod
    def from_pretrained(cls, model_id):
        processor = cls()
        processor.model_id = model_id
        cls.calls.append(processor)
        return processor


class SemanticPartVLMLoaderTests(unittest.TestCase):
    def setUp(self):
        _FakeModel.calls.clear()
        _FakeProcessor.calls.clear()

    def _transformers_module(self, model_attribute):
        module = types.ModuleType("transformers")
        module.AutoProcessor = _FakeProcessor
        setattr(module, model_attribute, _FakeModel)
        return module

    def test_loads_qwen25_without_changing_legacy_arguments(self):
        transformers = self._transformers_module(
            "Qwen2_5_VLForConditionalGeneration"
        )
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            model, processor, family = enumerator._load_qwen_vl(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "cpu",
            )

        self.assertEqual(family, enumerator.QWEN25_VL_FAMILY)
        self.assertEqual(model.kwargs["torch_dtype"], "auto")
        self.assertIsNone(model.kwargs["device_map"])
        self.assertEqual(model.moved_to, "cpu")
        self.assertEqual(processor.model_id, model.model_id)
        self.assertEqual(processor.tokenizer.padding_side, "left")

    def test_loads_qwen3_with_current_transformers_arguments(self):
        transformers = self._transformers_module(
            "Qwen3VLForConditionalGeneration"
        )
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            model, processor, family = enumerator._load_qwen_vl(
                "Qwen/Qwen3-VL-8B-Instruct",
                "cuda:0",
            )

        self.assertEqual(family, enumerator.QWEN3_VL_FAMILY)
        self.assertEqual(model.kwargs["dtype"], "auto")
        self.assertEqual(model.kwargs["device_map"], "auto")
        self.assertIsNone(model.moved_to)
        self.assertEqual(processor.model_id, model.model_id)
        self.assertEqual(processor.tokenizer.padding_side, "left")

    def test_rejects_unrelated_model_families(self):
        with self.assertRaisesRegex(ValueError, "Qwen2.5-VL or Qwen3-VL"):
            enumerator._qwen_vl_family("example/unrelated-vlm")

    def test_qwen3_preparation_uses_integrated_chat_template(self):
        class Inputs(dict):
            pass

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return Inputs(token_type_ids="unused", input_ids="tokens")

        processor = Processor()
        messages = [{"role": "user", "content": []}]
        inputs = enumerator._prepare_qwen_vl_inputs(
            processor,
            messages,
            enumerator.QWEN3_VL_FAMILY,
        )

        self.assertNotIn("token_type_ids", inputs)
        self.assertEqual(inputs["input_ids"], "tokens")
        self.assertTrue(processor.kwargs["tokenize"])
        self.assertTrue(processor.kwargs["return_dict"])
        self.assertEqual(processor.messages, messages)


if __name__ == "__main__":
    unittest.main()
