import gc
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List

import torch
from PIL import Image


DEFAULT_QWEN25_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN25_VL_FAMILY = "qwen2.5-vl"
QWEN3_VL_FAMILY = "qwen3-vl"


@dataclass
class SemanticComponent:
    name: str
    phrase: str
    count: int
    weight: float
    importance: float = None
    confidence: float = None

    @property
    def grounding_phrase(self):
        """Legacy alias retained for old callers and caches."""
        return self.phrase

    def public_dict(self):
        return {
            "name": self.name,
            "phrase": self.phrase,
            "count": self.count,
            "weight": self.weight,
        }


@dataclass
class SemanticPartEnumeration:
    components: List[SemanticComponent]
    parts_text: str
    weights_text: str
    part_queries: Dict[str, str]
    raw_text: str
    fallback: bool = False
    parse_failed: bool = False
    attempts: int = 1

    @classmethod
    def from_components(
        cls,
        components,
        raw_text="",
        fallback=False,
        parse_failed=False,
        attempts=1,
    ):
        parts = ["outline"] + [component.name for component in components]
        weights = ["outline=1.0"] + [
            f"{component.name}={component.weight:.8f}"
            for component in components
        ]
        return cls(
            components=list(components),
            parts_text=",".join(parts),
            weights_text=",".join(weights),
            part_queries={
                component.name: component.phrase
                for component in components
            },
            raw_text=raw_text,
            fallback=fallback,
            parse_failed=parse_failed,
            attempts=attempts,
        )

    @classmethod
    def from_cache(cls, payload):
        cached_items = payload.get("parts", payload.get("components", []))
        uses_new_schema = any(
            isinstance(item, dict) and "weight" in item
            for item in cached_items
        )
        components = _normalize_components(
            {"parts" if uses_new_schema else "components": cached_items},
            max_parts=len(cached_items),
        )
        return cls.from_components(
            components,
            raw_text=payload.get("raw_output", ""),
            fallback=bool(payload.get("vlm_fallback", False)),
            parse_failed=bool(payload.get("parse_failed", False)),
            attempts=int(payload.get("attempts", 1)),
        )

    def filtered(self, names):
        wanted = set(names)
        return self.from_components(
            _renormalize([
                component
                for component in self.components
                if component.name in wanted
            ]),
            raw_text=self.raw_text,
            fallback=self.fallback,
            parse_failed=self.parse_failed,
            attempts=self.attempts,
        )


def enumerate_semantic_parts(
    image,
    object_name="",
    caption="",
    model_id=DEFAULT_QWEN25_VL_MODEL,
    device=None,
    max_parts=8,
    weight_min=None,
    weight_max=None,
    seed=0,
    temperature=0.3,
    prompt_kind="new",
    runner=None,
):
    del weight_min, weight_max  # Legacy arguments; weights are now normalized.
    prompt_builder = _build_prompt if prompt_kind == "new" else _build_old_prompt
    prompt = prompt_builder(object_name, caption, max_parts)
    raw_text = ""
    last_error = None
    parse_failed = False

    for attempt in range(1, 3):
        try:
            if runner is None:
                raw_text = _run_qwen_vl(
                    image=image,
                    prompt=prompt,
                    model_id=model_id,
                    device=device,
                    seed=seed,
                    temperature=temperature,
                )
            else:
                raw_text = runner(
                    image=image,
                    prompt=prompt,
                    model_id=model_id,
                    device=device,
                    seed=seed,
                    temperature=temperature,
                )
            components = parse_components(raw_text, max_parts=max_parts)
            if not components:
                raise ValueError("VLM returned no usable parts")
            return SemanticPartEnumeration.from_components(
                components,
                raw_text=raw_text,
                attempts=attempt,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            last_error = error
            parse_failed = True

    print(
        f"[semantic_vlm] warning: using outline-only fallback: {last_error}",
        file=sys.stderr,
        flush=True,
    )
    return SemanticPartEnumeration.from_components(
        [],
        raw_text=raw_text,
        fallback=True,
        parse_failed=parse_failed,
        attempts=2,
    )


def _build_prompt(object_name, caption, max_parts):
    del caption  # The requested schema includes only the object-name context.
    ctx = f"\nObject: {object_name}" if object_name else ""
    return f"""You are labeling parts of an object so that a line drawing can be initialized.

List the drawable parts of the main foreground object. Use as many or as few
as the object actually needs, up to {max_parts}. A simple object may need only
two or three; a complex one may need more. Do not pad the list.

Granularity rule - follow this strictly:
  A part must be a named structural component that a person would draw as a
  separate closed or open contour. Use the level of "wheel", "handlebar",
  "ear", "tail", "backrest".
  Too coarse (do NOT use): the whole object, "body" alone, "upper half".
  Too fine (do NOT use): "spoke", "eyelash", "screw", "grain of wood",
  surface texture, material, colour region.
  If a part appears more than once (two ears, four legs), emit it ONCE with
  a plural name and set count.

Exclude: background, shadow, reflection, lighting, anything not visible.

Rank the parts from most to least important for recognizing the object in a
minimal line drawing. Output them in that order, most important first.
Assign each an integer weight so that the weights sum to EXACTLY 100.

Return only JSON, no prose:
{{"parts":[{{"name":"wheels","phrase":"wheel of the bicycle","count":2,"weight":35}}]}}
{ctx}"""


def _build_old_prompt(object_name, caption, max_parts):
    context = ""
    if object_name:
        context += f"\nKnown object name: {object_name}"
    if caption:
        context += f"\nCaption: {caption}"

    return f"""
You are helping initialize vector sketch strokes.

Given the image, list visually and semantically important drawable components
of the main foreground object. The object may be an animal, person, vehicle,
furniture, building, tool, plant, food, logo, or another non-living object.

List components that help recognize the object, define its visible structure,
or should receive sketch strokes.

Do not list the whole object, background, lighting, shadow, reflection,
vague color regions, or invisible components.

For each component, return:
- name: short lowercase component name
- grounding_phrase: short Grounding-DINO-friendly phrase
- importance: integer 1-5, where 5 means many initial strokes should be assigned
- confidence: number 0-1

Return only valid JSON:
{{
  "components": [
    {{
      "name": "wheel",
      "grounding_phrase": "wheel",
      "importance": 5,
      "confidence": 0.95
    }}
  ]
}}

Return at most {max_parts} components.
{context}
""".strip()


def _qwen_vl_family(model_id):
    normalized = model_id.lower().replace("_", "-")
    if "qwen3-vl" in normalized:
        return QWEN3_VL_FAMILY
    if "qwen2.5-vl" in normalized:
        return QWEN25_VL_FAMILY
    raise ValueError(
        "Unsupported semantic VLM model. Expected a Qwen2.5-VL or "
        f"Qwen3-VL model ID, got: {model_id}"
    )


def _load_qwen_vl(model_id, device_name):
    from transformers import AutoProcessor

    family = _qwen_vl_family(model_id)
    try:
        if family == QWEN3_VL_FAMILY:
            from transformers import Qwen3VLForConditionalGeneration
            model_class = Qwen3VLForConditionalGeneration
            dtype_kwargs = {"dtype": "auto"}
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_class = Qwen2_5_VLForConditionalGeneration
            dtype_kwargs = {"torch_dtype": "auto"}
    except ImportError as exc:
        requirement = (
            "transformers>=4.57.0"
            if family == QWEN3_VL_FAMILY
            else "a Transformers release with Qwen2.5-VL support"
        )
        raise ImportError(
            f"{model_id} requires {requirement}. Install it in the dedicated "
            "Qwen environment; the ControlSketch environment was not modified."
        ) from exc

    model = model_class.from_pretrained(
        model_id,
        device_map="auto" if device_name.startswith("cuda") else None,
        **dtype_kwargs,
    )
    if not device_name.startswith("cuda"):
        model = model.to(device_name)

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    return model, processor, family


def _prepare_qwen_vl_inputs(processor, messages, family):
    if family == QWEN3_VL_FAMILY:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        return inputs

    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def _run_qwen_vl(image, prompt, model_id, device, seed=0, temperature=0.3):
    image = _as_rgb_image(image)
    device_name = str(device) if device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, processor, family = _load_qwen_vl(model_id, device_name)
    try:
        return _generate_qwen_vl(
            model, processor, family, image, prompt, seed, temperature
        )
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _generate_qwen_vl(
    model, processor, family, image, prompt, seed=0, temperature=0.3
):
    from transformers import set_seed

    image = _as_rgb_image(image)
    model.generation_config.temperature = float(temperature)
    model.generation_config.do_sample = True
    model.generation_config.num_beams = 1
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = _prepare_qwen_vl_inputs(processor, messages, family)
    inputs = inputs.to(next(model.parameters()).device)

    set_seed(int(seed))
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            num_beams=1,
            temperature=float(temperature),
        )
    generated_ids = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _as_rgb_image(image):
    return (
        image.convert("RGB")
        if isinstance(image, Image.Image)
        else Image.fromarray(image).convert("RGB")
    )


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise json.JSONDecodeError("No JSON value found", text, 0)
    start = min(starts)
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    return data


def parse_components(text, max_parts=8, warning_fn=None):
    return _normalize_components(
        _extract_json(text),
        max_parts=max_parts,
        warning_fn=warning_fn,
    )


def _normalize_components(
    data,
    max_parts,
    weight_min=None,
    weight_max=None,
    warning_fn=None,
):
    del weight_min, weight_max
    warning_fn = warning_fn or (lambda message: warnings.warn(message))
    if isinstance(data, list):
        is_new_list = any(
            isinstance(item, dict)
            and ("phrase" in item or "weight" in item)
            for item in data
        )
        data = {"parts" if is_new_list else "components": data}
    if not isinstance(data, dict):
        raise TypeError("VLM JSON root must be an object or array")
    banned = {
        "object", "whole object", "main object", "foreground object",
        "background", "shadow", "lighting", "reflection", "detail",
        "texture", "color", "shape", "area", "outline", "silhouette",
        "body", "upper half",
    }
    is_new = isinstance(data.get("parts"), list)
    items = data.get("parts", []) if is_new else data.get("components", [])
    normalized = []

    for item in items[:max_parts]:
        if not isinstance(item, dict):
            continue
        name = _clean_phrase(item.get("name", ""))
        phrase_key = "phrase" if is_new else "grounding_phrase"
        phrase = _clean_phrase(item.get(phrase_key, "") or name)
        if not name or not phrase or name in banned:
            continue

        try:
            count = int(item.get("count", 1))
        except (TypeError, ValueError):
            count = 1
        count = max(count, 1)

        if is_new:
            try:
                if "weight" not in item:
                    raise ValueError
                raw_weight = float(item["weight"])
                if not math.isfinite(raw_weight) or raw_weight < 0:
                    raise ValueError
            except (TypeError, ValueError):
                raw_weight = 0.0
                warning_fn(f"Invalid weight for semantic part '{name}'; using 0")
            importance = None
            confidence = None
        else:
            try:
                importance = float(item.get("importance", 1.0))
            except (TypeError, ValueError):
                importance = 1.0
            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            importance = _clamp(importance, 1.0, 5.0)
            confidence = _clamp(confidence, 0.0, 1.0)
            raw_weight = importance * max(confidence, 0.05)

        normalized.append(SemanticComponent(
            name=name,
            phrase=phrase,
            count=count,
            weight=raw_weight,
            importance=importance,
            confidence=confidence,
        ))

    unique = []
    seen = set()
    for component in normalized:
        if component.name not in seen:
            unique.append(component)
            seen.add(component.name)
    return _renormalize(unique)


def _renormalize(components):
    components = list(components)
    if not components:
        return []
    total = sum(max(float(component.weight), 0.0) for component in components)
    if total <= 0:
        value = 1.0 / len(components)
        return [
            SemanticComponent(
                name=component.name,
                phrase=component.phrase,
                count=component.count,
                weight=value,
                importance=component.importance,
                confidence=component.confidence,
            )
            for component in components
        ]
    return [
        SemanticComponent(
            name=component.name,
            phrase=component.phrase,
            count=component.count,
            weight=max(float(component.weight), 0.0) / total,
            importance=component.importance,
            confidence=component.confidence,
        )
        for component in components
    ]


def _clean_phrase(text):
    text = str(text).lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9 -]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip(" -")


def _clamp(value, low, high):
    return max(low, min(high, value))
