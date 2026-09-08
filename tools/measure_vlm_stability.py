#!/usr/bin/env python3
"""Measure semantic-part output stability without reading or writing caches."""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONTROL_SKETCH = ROOT / "ControlSketch"
sys.path.insert(0, str(CONTROL_SKETCH))

from semantic_part_enumerator import (  # noqa: E402
    DEFAULT_QWEN25_VL_MODEL,
    _generate_qwen_vl,
    _load_qwen_vl,
    enumerate_semantic_parts,
)


class ReusableQwenRunner:
    def __init__(self, model_id, device):
        self.model_id = model_id
        self.device = device
        self.model, self.processor, self.family = _load_qwen_vl(model_id, device)
        self.default_temperature = float(
            getattr(self.model.generation_config, "temperature", 1.0) or 1.0
        )

    def __call__(self, image, prompt, model_id, device, seed, temperature):
        if model_id != self.model_id:
            raise ValueError("Reusable runner received a different model ID")
        return _generate_qwen_vl(
            self.model,
            self.processor,
            self.family,
            image,
            prompt,
            seed,
            temperature,
        )


def _pair_metrics(runs):
    if not runs:
        return 0.0, 0.0
    pairs = list(itertools.combinations(runs, 2))
    if not pairs:
        return 1.0, 0.0
    jaccards = []
    l1_distances = []
    for left, right in pairs:
        left_weights = {part.name: part.weight for part in left.components}
        right_weights = {part.name: part.weight for part in right.components}
        left_names = set(left_weights)
        right_names = set(right_weights)
        union = left_names | right_names
        jaccards.append(len(left_names & right_names) / len(union) if union else 0.0)
        l1_distances.append(sum(
            abs(left_weights.get(name, 0.0) - right_weights.get(name, 0.0))
            for name in union
        ))
    return float(np.mean(jaccards)), float(np.mean(l1_distances))


def _measure(image, runner, repeat, prompt_kind, max_parts, temperature):
    runs = [
        enumerate_semantic_parts(
            image=image,
            model_id=runner.model_id,
            device=runner.device,
            max_parts=max_parts,
            seed=seed,
            temperature=temperature,
            prompt_kind=prompt_kind,
            runner=runner,
        )
        for seed in range(repeat)
    ]
    valid_runs = [run for run in runs if not run.parse_failed]
    jaccard, weight_l1 = _pair_metrics(valid_runs)
    part_counts = [len(run.components) for run in valid_runs]
    return {
        "jaccard": jaccard,
        "part_count_variance": float(np.var(part_counts)) if part_counts else 0.0,
        "weight_l1": weight_l1,
        "parse_failure_rate": sum(run.parse_failed for run in runs) / repeat,
        "fallback_rate": sum(run.fallback for run in runs) / repeat,
    }


def _images(image_dir):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--prompt", choices=["old", "new"], default="new")
    parser.add_argument("--model", default=DEFAULT_QWEN25_VL_MODEL)
    parser.add_argument("--max_parts", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="stability.csv")
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("--repeat must be at least 2")

    paths = _images(args.image_dir)
    if not paths:
        parser.error("image_dir contains no supported images")
    runner = ReusableQwenRunner(args.model, args.device)
    temperatures = (("t03", 0.3), ("default", runner.default_temperature))
    rows = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        row = {"image": str(path), "prompt": args.prompt}
        for label, temperature in temperatures:
            metrics = _measure(
                image, runner, args.repeat, args.prompt, args.max_parts, temperature
            )
            row[f"temperature_{label}"] = temperature
            for name, value in metrics.items():
                row[f"{name}_{label}"] = value
        rows.append(row)

    fieldnames = list(rows[0])
    with Path(args.output).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"prompt={args.prompt} images={len(rows)} repeat={args.repeat}")
    for label, _ in temperatures:
        summary = ", ".join(
            f"{metric}={np.mean([row[f'{metric}_{label}'] for row in rows]):.4f}"
            for metric in (
                "jaccard", "part_count_variance", "weight_l1",
                "parse_failure_rate", "fallback_rate"
            )
        )
        print(f"{label}: {summary}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
