import os
import sys
from pathlib import Path

from semantic_part_enumerator import (
    DEFAULT_QWEN25_VL_MODEL,
    SemanticPartEnumeration,
    _build_prompt,
    enumerate_semantic_parts,
)
from semantic_reference_cache import ReferenceCache, compute_cache_hash


def prepare_auto_semantic_data(args, image, foreground_mask, device):
    model_id = getattr(args, "semantic_vlm_model", DEFAULT_QWEN25_VL_MODEL)
    max_parts = int(getattr(args, "max_parts", 8))
    seed = int(getattr(args, "vlm_seed", 0))
    temperature = float(getattr(args, "vlm_temperature", 0.3))
    prompt = _build_prompt(
        getattr(args, "object_name", ""),
        getattr(args, "caption", ""),
        max_parts,
    )
    cache_hash = compute_cache_hash(
        args.target,
        prompt,
        model_id,
        max_parts,
        seed,
        temperature,
    )
    cache = ReferenceCache(args.cache_dir, cache_hash)
    rebuild = bool(getattr(args, "rebuild_reference", False))

    if cache.semantic_ready and not rebuild:
        payload = cache.load_parts()
        enumeration = SemanticPartEnumeration.from_cache(payload)
        masks = cache.load_masks()
        info = _cache_info(payload, cache, enumeration, cache_hit=True)
        _write_run_debug(args, enumeration)
        print(f"[semantic_cache] hit={cache_hash}", flush=True)
        return enumeration, masks, cache, info

    enumeration = enumerate_semantic_parts(
        image=image,
        object_name=getattr(args, "object_name", ""),
        caption=getattr(args, "caption", ""),
        model_id=model_id,
        device=device,
        max_parts=max_parts,
        seed=seed,
        temperature=temperature,
    )
    original_components = list(enumeration.components)
    requested_parts = [component.name for component in original_components]
    masks = {}

    if requested_parts and not enumeration.fallback:
        try:
            segmenter = _make_segmenter(args, device)
            if segmenter is not None:
                masks = segmenter.segment_parts(
                    image=image,
                    parts=requested_parts,
                    foreground_mask=foreground_mask,
                    object_name=getattr(args, "object_name", ""),
                    part_queries=enumeration.part_queries,
                )
        except Exception as error:
            print(
                f"[semantic_segmenter] warning: outline-only after failure: {error}",
                file=sys.stderr,
                flush=True,
            )
            masks = {}

    retained = [
        name for name in requested_parts
        if name in masks and _mask_has_foreground(masks[name])
    ]
    dropped = [name for name in requested_parts if name not in retained]
    masks = {name: masks[name] for name in retained}
    enumeration = enumeration.filtered(retained)
    if dropped:
        print(
            f"[semantic_segmenter] dropped parts: {','.join(dropped)}",
            flush=True,
        )

    payload = {
        "schema_version": 2,
        "raw_output": enumeration.raw_text,
        "vlm_parts": [component.public_dict() for component in original_components],
        "parts": [component.public_dict() for component in enumeration.components],
        "sam_dropped_parts": dropped,
        "vlm_fallback": bool(enumeration.fallback),
        "parse_failed": bool(enumeration.parse_failed),
        "attempts": int(enumeration.attempts),
        "model_id": model_id,
        "max_parts": max_parts,
        "vlm_seed": seed,
        "vlm_temperature": temperature,
    }
    cache.save_parts(payload)
    cache.save_masks(masks)
    info = _cache_info(payload, cache, enumeration, cache_hit=False)
    _write_run_debug(args, enumeration)
    print(f"[semantic_cache] wrote={cache_hash}", flush=True)
    return enumeration, masks, cache, info


def _make_segmenter(args, device):
    import semantic_segmenter

    name = getattr(args, "semantic_segmenter", "sam3")
    if name == "none":
        return None
    debug_dir = os.path.join(args.output_dir, "semantic_debug")
    if name == "sam3":
        sam3_python = getattr(args, "sam3_python", "")
        if not sam3_python:
            raise ValueError("--sam3_python must point to the sam3 environment")
        return semantic_segmenter.SAM3SubprocessSegmenter(
            sam3_python=sam3_python,
            checkpoint_path=getattr(args, "sam3_checkpoint_path", ""),
            confidence_threshold=getattr(args, "sam3_confidence_threshold", 0.5),
            min_area_ratio=getattr(args, "semantic_min_area_ratio", 0.0002),
            max_masks_per_part=getattr(args, "semantic_max_masks_per_part", 4),
            debug_dir=debug_dir,
            max_part_area_ratio=getattr(args, "semantic_max_part_area_ratio", 0.90),
        )
    if name == "grounded_sam":
        return semantic_segmenter.GroundedSAMSegmenter(
            device=device,
            grounding_model_id=getattr(args, "grounding_dino_model"),
            sam_model_id=getattr(args, "sam_model"),
            box_threshold=getattr(args, "grounding_box_threshold", 0.25),
            text_threshold=getattr(args, "grounding_text_threshold", 0.20),
            min_area_ratio=getattr(args, "semantic_min_area_ratio", 0.0002),
            max_masks_per_part=getattr(args, "semantic_max_masks_per_part", 4),
            debug_dir=debug_dir,
            max_box_object_area_ratio=getattr(
                args, "grounding_max_box_object_area_ratio", 0.90
            ),
            max_part_area_ratio=getattr(args, "semantic_max_part_area_ratio", 0.90),
            prefer_small_boxes=bool(getattr(args, "grounding_prefer_small_boxes", 1)),
        )
    raise ValueError(f"Unknown semantic segmenter: {name}")


def _cache_info(payload, cache, enumeration, cache_hit):
    return {
        "vlm_model": payload.get("model_id", DEFAULT_QWEN25_VL_MODEL),
        "max_parts": int(payload.get("max_parts", 8)),
        "vlm_seed": int(payload.get("vlm_seed", 0)),
        "vlm_temperature": float(payload.get("vlm_temperature", 0.3)),
        "n_parts_returned": len(
            payload.get("vlm_parts", payload.get("parts", []))
        ),
        "cache_hash": cache.cache_hash,
        "parts": [component.public_dict() for component in enumeration.components],
        "sam_dropped_parts": list(payload.get("sam_dropped_parts", [])),
        "vlm_fallback": bool(payload.get("vlm_fallback", False)),
        "cache_hit": bool(cache_hit),
    }


def _write_run_debug(args, enumeration):
    debug_dir = Path(args.output_dir) / "semantic_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "vlm_raw_output.txt").write_text(
        enumeration.raw_text,
        encoding="utf-8",
    )


def _mask_has_foreground(mask):
    try:
        import torch
        if isinstance(mask, torch.Tensor):
            return bool(torch.any(mask > 0).item())
    except ImportError:
        pass
    import numpy as np
    return bool(np.any(np.asarray(mask) > 0))
