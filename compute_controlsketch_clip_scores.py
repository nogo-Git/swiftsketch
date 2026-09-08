#!/usr/bin/env python3
"""Compute CLIP image-image scores for ControlSketch outputs.

This scans:
  <sketch_root>/*/*_strokes/final_sketch.png

Each sketch is paired with:
  <input_root>/<category>.png

The JSON output contains one score per sketch plus summary statistics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from compute_clip_score import import_clip_module, load_image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SKETCH_GLOB = "*/*_strokes/final_sketch.png"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
}
STROKES_RE = re.compile(r"_(\d+)_strokes$")


@dataclass(frozen=True)
class SketchPair:
    category: str
    run: str
    num_strokes: Optional[int]
    input_image: Path
    sketch_image: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CLIP similarities between ControlSketch final sketches and "
            "their source images, then emit JSON."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Root directory containing input/reference images.",
    )
    parser.add_argument(
        "sketch_root",
        type=Path,
        help="Root directory containing sketch outputs.",
    )
    parser.add_argument(
        "--sketch-glob",
        default=DEFAULT_SKETCH_GLOB,
        help=(
            "Glob pattern, relative to sketch_root, for sketch images. "
            f"Default: {DEFAULT_SKETCH_GLOB}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this file. If omitted, JSON is printed to stdout.",
    )
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        help="CLIP model name or path to a local CLIP .pt checkpoint. Default: ViT-B/32.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for CLIP inference. Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images encoded per CLIP batch. Default: 32.",
    )
    parser.add_argument(
        "--jit",
        action="store_true",
        help="Load CLIP as a JIT model. By default the non-JIT model is used.",
    )
    parser.add_argument(
        "--fail-on-missing-input",
        action="store_true",
        help="Return an error if a sketch category has no matching input image.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON instead of pretty-printed JSON.",
    )
    return parser.parse_args()


def as_json_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_input_index(input_root: Path) -> dict[str, Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(input_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        key = path.stem
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path)
            continue
        index[key] = path

    if duplicates:
        details = "; ".join(
            f"{stem}: {', '.join(as_json_path(path) for path in paths)}"
            for stem, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate input image stems found under {input_root}: {details}")

    return index


def parse_num_strokes(run_name: str) -> Optional[int]:
    match = STROKES_RE.search(run_name)
    return int(match.group(1)) if match else None


def discover_pairs(
    sketch_root: Path,
    input_index: dict[str, Path],
    sketch_glob: str,
    fail_on_missing_input: bool,
) -> tuple[list[SketchPair], list[dict[str, str]]]:
    if not sketch_root.exists():
        raise FileNotFoundError(f"Sketch root not found: {sketch_root}")

    pairs: list[SketchPair] = []
    errors: list[dict[str, str]] = []
    sketch_paths = sorted(path for path in sketch_root.glob(sketch_glob) if path.is_file())

    for sketch_path in sketch_paths:
        rel_parts = sketch_path.relative_to(sketch_root).parts
        if not rel_parts:
            errors.append(
                {
                    "sketch_image": as_json_path(sketch_path),
                    "error": "Could not infer category from sketch path.",
                }
            )
            continue

        category = rel_parts[0]
        input_image = sketch_path.parent / "input.png"
        if not input_image.is_file():
            input_image = input_index.get(category)
        if input_image is None:
            errors.append(
                {
                    "category": category,
                    "sketch_image": as_json_path(sketch_path),
                    "error": f"No adjacent input.png or matching input image named {category}.* under input root.",
                }
            )
            continue

        pairs.append(
            SketchPair(
                category=category,
                run=sketch_path.parent.name,
                num_strokes=parse_num_strokes(sketch_path.parent.name),
                input_image=input_image,
                sketch_image=sketch_path,
            )
        )

    if fail_on_missing_input and errors:
        missing = "\n".join(f"- {item['sketch_image']}: {item['error']}" for item in errors)
        raise FileNotFoundError(f"Some sketches have no matching input image:\n{missing}")

    return pairs, errors


def resolve_device(device: str, torch: Any) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def encode_images(
    paths: list[Path],
    model: Any,
    preprocess: Any,
    device: str,
    batch_size: int,
    torch: Any,
) -> Any:
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack([preprocess(load_image(path)) for path in batch_paths]).to(device)
        batch_features = model.encode_image(batch).float()
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        features.append(batch_features)

    if not features:
        return torch.empty((0, 0))
    return torch.cat(features, dim=0)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_scored": 0,
            "mean_cosine_similarity": None,
            "mean_clip_score_x100": None,
            "by_category": {},
        }

    by_category: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_category.setdefault(result["category"], []).append(result)

    return {
        "num_scored": len(results),
        "mean_cosine_similarity": mean(item["cosine_similarity"] for item in results),
        "mean_clip_score_x100": mean(item["clip_score_x100"] for item in results),
        "by_category": {
            category: {
                "count": len(items),
                "mean_cosine_similarity": mean(item["cosine_similarity"] for item in items),
                "mean_clip_score_x100": mean(item["clip_score_x100"] for item in items),
            }
            for category, items in sorted(by_category.items())
        },
    }


def compute_scores(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to compute CLIP scores. Install the project "
            "requirements or run this script in the project's Python environment."
        ) from exc

    input_index = build_input_index(args.input_root)
    pairs, discovery_errors = discover_pairs(
        sketch_root=args.sketch_root,
        input_index=input_index,
        sketch_glob=args.sketch_glob,
        fail_on_missing_input=args.fail_on_missing_input,
    )

    device = resolve_device(args.device, torch)
    results: list[dict[str, Any]] = []

    if pairs:
        clip = import_clip_module()
        model, preprocess = clip.load(args.model, device=device, jit=args.jit)
        model.eval()

        for pair in pairs:
            pair_features = encode_images(
                [pair.input_image, pair.sketch_image],
                model,
                preprocess,
                device,
                2,
                torch,
            )
            cosine_similarity = (pair_features[0] @ pair_features[1]).item()
            results.append(
                {
                    "category": pair.category,
                    "run": pair.run,
                    "num_strokes": pair.num_strokes,
                    "input_image": as_json_path(pair.input_image),
                    "sketch_image": as_json_path(pair.sketch_image),
                    "cosine_similarity": cosine_similarity,
                    "clip_score_x100": cosine_similarity * 100.0,
                }
            )

    summary = summarize_results(results)
    summary["num_discovery_errors"] = len(discovery_errors)
    summary["num_sketches_found"] = len(pairs) + len(discovery_errors)

    return {
        "metadata": {
            "sketch_root": as_json_path(args.sketch_root),
            "input_root": as_json_path(args.input_root),
            "sketch_glob": args.sketch_glob,
            "model": args.model,
            "device": device,
            "pair_batch_size": 2,
            "score_definition": "clip_score_x100 = cosine_similarity(image_features, sketch_features) * 100",
        },
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

    indent = None if args.compact else 2
    json_text = json.dumps(payload, ensure_ascii=False, indent=indent)

    if args.output is None:
        print(json_text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
