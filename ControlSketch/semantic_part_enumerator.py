# ControlSketch/semantic_part_enumerator.py

import gc
import json
import re
from dataclasses import dataclass
from typing import Dict, List

import torch
from PIL import Image


DEFAULT_QWEN25_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


@dataclass
class SemanticComponent:
    name: str
    grounding_phrase: str
    importance: float
    confidence: float
    weight: float


@dataclass
class SemanticPartEnumeration:
    components: List[SemanticComponent]
    parts_text: str
    weights_text: str
    part_queries: Dict[str, str]
    raw_text: str


def enumerate_semantic_parts(
    image,
    object_name="",
    caption="",
    model_id=DEFAULT_QWEN25_VL_MODEL,
    device=None,
    max_parts=8,
    weight_min=0.5,
    weight_max=4.0,
):
    raw_text = _run_qwen25_vl(
        image=image,
        prompt=_build_prompt(object_name, caption, max_parts),
        model_id=model_id,
        device=device,
    )

    data = _extract_json(raw_text)
    components = _normalize_components(data, max_parts, weight_min, weight_max)

    parts = ["outline"] + [c.name for c in components]
    weights = ["outline=1.0"] + [f"{c.name}={c.weight:.3f}" for c in components]
    part_queries = {c.name: c.grounding_phrase for c in components}

    return SemanticPartEnumeration(
        components=components,
        parts_text=",".join(parts),
        weights_text=",".join(weights),
        part_queries=part_queries,
        raw_text=raw_text,
    )


def _build_prompt(object_name, caption, max_parts):
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


def _run_qwen25_vl(image, prompt, model_id, device):
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    device_name = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto" if device_name.startswith("cuda") else None,
    )
    if not device_name.startswith("cuda"):
        model = model.to(device_name)

    processor = AutoProcessor.from_pretrained(model_id)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated_ids = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    raw_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw_text


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def _normalize_components(data, max_parts, weight_min, weight_max):
    banned = {
        "object", "whole object", "main object", "foreground object",
        "background", "shadow", "lighting", "reflection",
        "detail", "texture", "color", "shape", "area",
        "outline", "silhouette",
    }

    items = data.get("components", [])
    normalized = []

    for item in items:
        name = _clean_phrase(item.get("name", ""))
        phrase = _clean_phrase(item.get("grounding_phrase", name))

        if not name or name in banned:
            continue

        importance = _clamp(float(item.get("importance", 1.0)), 1.0, 5.0)
        confidence = _clamp(float(item.get("confidence", 1.0)), 0.0, 1.0)
        raw_score = importance * max(confidence, 0.05)

        normalized.append({
            "name": name,
            "grounding_phrase": phrase or name,
            "importance": importance,
            "confidence": confidence,
            "raw_score": raw_score,
        })

    normalized = sorted(normalized, key=lambda x: x["raw_score"], reverse=True)[:max_parts]

    if not normalized:
        return []

    mean_score = sum(x["raw_score"] for x in normalized) / len(normalized)

    return [
        SemanticComponent(
            name=x["name"],
            grounding_phrase=x["grounding_phrase"],
            importance=x["importance"],
            confidence=x["confidence"],
            weight=_clamp(x["raw_score"] / max(mean_score, 1e-6), weight_min, weight_max),
        )
        for x in normalized
    ]


def _clean_phrase(text):
    text = str(text).lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9 -]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip(" -")


def _clamp(value, low, high):
    return max(low, min(high, value))