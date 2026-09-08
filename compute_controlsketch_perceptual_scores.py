#!/usr/bin/env python3
"""Compute DreamSIM and MS-SSIM scores for ControlSketch outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from compute_controlsketch_clip_scores import (
    DEFAULT_SKETCH_GLOB,
    SketchPair,
    as_json_path,
    build_input_index,
    discover_pairs,
)
from image_similarity_metrics import (
    DreamSimEvaluator,
    XDoGConfig,
    compute_ms_ssim_scores,
)


METRIC_DREAMSIM = "dreamsim"
METRIC_MS_SSIM = "ms-ssim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DreamSIM distance and/or MS-SSIM fidelity between source "
            "images and ControlSketch final sketches, then emit JSON."
        )
    )
    parser.add_argument("input_root", type=Path, help="Directory of reference images.")
    parser.add_argument("sketch_root", type=Path, help="Directory of sketch outputs.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[METRIC_DREAMSIM, METRIC_MS_SSIM],
        default=[METRIC_DREAMSIM, METRIC_MS_SSIM],
        help="Metrics to compute. Default: dreamsim ms-ssim.",
    )
    parser.add_argument(
        "--sketch-glob",
        default=DEFAULT_SKETCH_GLOB,
        help=f"Sketch glob relative to sketch_root. Default: {DEFAULT_SKETCH_GLOB}",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. Default: cuda when available, otherwise cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Default: 16.")
    parser.add_argument(
        "--dreamsim-type",
        default="ensemble",
        help="DreamSIM model variant. Default: ensemble.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square MS-SSIM evaluation size; must be >160. Default: 256.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="stretch",
        help="MS-SSIM resize policy. Default: stretch (paper images are square).",
    )
    parser.add_argument("--xdog-sigma", type=float, default=0.5)
    parser.add_argument("--xdog-k", type=float, default=10.0)
    parser.add_argument("--xdog-gamma", type=float, default=0.98)
    parser.add_argument("--xdog-epsilon", type=float, default=-0.1)
    parser.add_argument("--xdog-phi", type=float, default=200.0)
    parser.add_argument(
        "--fail-on-missing-input",
        action="store_true",
        help="Fail instead of recording sketches without matching reference images.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable DreamSIM and MS-SSIM progress bars.",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    return parser.parse_args()


def _mean_or_none(values: list[float]) -> Optional[float]:
    return mean(values) if values else None


def summarize_results(
    results: list[dict[str, Any]], metrics: list[str]
) -> dict[str, Any]:
    metric_keys = {
        METRIC_DREAMSIM: "dreamsim_distance",
        METRIC_MS_SSIM: "ms_ssim",
    }
    summary: dict[str, Any] = {"num_scored": len(results)}
    for metric in metrics:
        key = metric_keys[metric]
        summary[f"mean_{key}"] = _mean_or_none([item[key] for item in results])

    by_category: dict[str, list[dict[str, Any]]] = {}
    by_num_strokes: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_category.setdefault(result["category"], []).append(result)
        stroke_key = (
            str(result["num_strokes"])
            if result["num_strokes"] is not None
            else "unknown"
        )
        by_num_strokes.setdefault(stroke_key, []).append(result)

    def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        group: dict[str, Any] = {"count": len(items)}
        for metric in metrics:
            key = metric_keys[metric]
            group[f"mean_{key}"] = _mean_or_none([item[key] for item in items])
        return group

    summary["by_category"] = {
        key: summarize_group(items) for key, items in sorted(by_category.items())
    }
    summary["by_num_strokes"] = {
        key: summarize_group(items)
        for key, items in sorted(
            by_num_strokes.items(),
            key=lambda item: (
                item[0] == "unknown",
                int(item[0]) if item[0].isdigit() else 0,
            ),
        )
    }
    return summary


def _base_result(pair: SketchPair) -> dict[str, Any]:
    return {
        "category": pair.category,
        "run": pair.run,
        "num_strokes": pair.num_strokes,
        "input_image": as_json_path(pair.input_image),
        "sketch_image": as_json_path(pair.sketch_image),
    }


def compute_scores(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    metrics = list(dict.fromkeys(args.metrics))
    input_index = build_input_index(args.input_root)
    pairs, discovery_errors = discover_pairs(
        sketch_root=args.sketch_root,
        input_index=input_index,
        sketch_glob=args.sketch_glob,
        fail_on_missing_input=args.fail_on_missing_input,
    )
    path_pairs = [(pair.input_image, pair.sketch_image) for pair in pairs]
    results = [_base_result(pair) for pair in pairs]
    metadata: dict[str, Any] = {
        "input_root": as_json_path(args.input_root),
        "sketch_root": as_json_path(args.sketch_root),
        "sketch_glob": args.sketch_glob,
        "metrics": metrics,
        "requested_device": args.device,
        "batch_size": args.batch_size,
        "progress_enabled": not args.no_progress,
        "score_directions": {
            "dreamsim_distance": "lower_is_better",
            "ms_ssim": "higher_is_better",
        },
    }

    resolved_devices: dict[str, str] = {}
    if pairs and METRIC_DREAMSIM in metrics:
        evaluator = DreamSimEvaluator(args.device, args.dreamsim_type)
        distances = evaluator.distances(
            path_pairs, args.batch_size, show_progress=not args.no_progress
        )
        for result, distance in zip(results, distances):
            result["dreamsim_distance"] = distance
        resolved_devices[METRIC_DREAMSIM] = evaluator.device
        metadata["dreamsim"] = {
            "model_type": args.dreamsim_type,
            "score_definition": "1 - cosine_similarity(normalized_embeddings)",
            "reference_preprocessing": "DreamSIM official RGB preprocessing",
            "sketch_preprocessing": (
                "RGB composited on white, then DreamSIM preprocessing"
            ),
        }

    xdog_config = XDoGConfig(
        sigma=args.xdog_sigma,
        k=args.xdog_k,
        gamma=args.xdog_gamma,
        epsilon=args.xdog_epsilon,
        phi=args.xdog_phi,
    )
    if pairs and METRIC_MS_SSIM in metrics:
        scores, resolved_device = compute_ms_ssim_scores(
            path_pairs,
            device=args.device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            resize_mode=args.resize_mode,
            xdog_config=xdog_config,
            show_progress=not args.no_progress,
        )
        for result, score in zip(results, scores):
            result["ms_ssim"] = score
        resolved_devices[METRIC_MS_SSIM] = resolved_device
        metadata["ms_ssim"] = {
            "implementation": "pytorch-msssim",
            "data_range": 1.0,
            "channels": 1,
            "image_size": args.image_size,
            "resize_mode": args.resize_mode,
            "reference_preprocessing": "grayscale XDoG edge map",
            "sketch_preprocessing": "grayscale composited on white",
            "xdog": xdog_config.to_dict(),
        }

    metadata["resolved_devices"] = resolved_devices
    summary = summarize_results(results, metrics)
    summary["num_discovery_errors"] = len(discovery_errors)
    summary["num_sketches_found"] = len(pairs) + len(discovery_errors)
    return {
        "metadata": metadata,
        "summary": summary,
        "results": results,
        "errors": discovery_errors,
    }


def main() -> int:
    args = parse_args()
    try:
        payload = compute_scores(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    json_text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=None if args.compact else 2,
    )
    if args.output is None:
        print(json_text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
