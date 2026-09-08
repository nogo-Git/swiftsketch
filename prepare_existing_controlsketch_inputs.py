#!/usr/bin/env python3
"""Align existing ControlSketch input images with their final sketches."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


DEFAULT_SKETCH_GLOB = "*/*_strokes/final_sketch.png"


@dataclass
class PreparationResult:
    run_dir: str
    status: str
    input_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None
    transformed: bool = False
    message: str | None = None


def load_config(path: Path) -> dict[str, Any]:
    config = np.load(path, allow_pickle=True).item()
    if not isinstance(config, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return config


def _transform_parameters(
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    config: dict[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    required = (
        "scale_w",
        "scale_h",
        "original_center_x",
        "original_center_y",
    )
    if not all(key in config for key in required):
        return None

    source_width, source_height = source_size
    output_width, output_height = output_size
    scale_w = float(config["scale_w"])
    scale_h = float(config["scale_h"])
    center_x = float(config["original_center_x"])
    center_y = float(config["original_center_y"])
    if scale_w <= 0 or scale_h <= 0:
        raise ValueError("scale_w and scale_h must be positive")

    # PIL expects an inverse affine mapping from each output coordinate to the
    # corresponding source coordinate. This is the inverse of the transform
    # used by increase_object_size() when the final sketch is saved.
    a = scale_w * source_width / output_width
    c = source_width / 2.0 - scale_w * center_x * source_width
    e = scale_h * source_height / output_height
    f = source_height / 2.0 - scale_h * center_y * source_height
    return a, 0.0, c, 0.0, e, f


def align_input_image(
    source: Image.Image,
    output_size: tuple[int, int],
    config: dict[str, Any],
) -> tuple[Image.Image, bool]:
    source = source.convert("RGB")
    parameters = _transform_parameters(source.size, output_size, config)
    if parameters is None:
        return (
            source.resize(output_size, Image.Resampling.LANCZOS),
            False,
        )

    aligned = source.transform(
        output_size,
        Image.Transform.AFFINE,
        parameters,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )
    return aligned, True


def prepare_run(run_dir: Path, *, dry_run: bool, force: bool) -> PreparationResult:
    final_path = run_dir / "final_sketch.png"
    input_path = run_dir / "input.png"
    backup_path = run_dir / "optimization_input.png"
    config_path = run_dir / "config.npy"

    if not final_path.is_file():
        return PreparationResult(str(run_dir), "skipped", message="missing final_sketch.png")
    if not input_path.is_file() and not backup_path.is_file():
        return PreparationResult(str(run_dir), "error", message="missing input.png")
    if backup_path.is_file() and not force:
        return PreparationResult(
            str(run_dir),
            "skipped",
            message="optimization_input.png already exists; use --force to regenerate",
        )

    source_path = backup_path if backup_path.is_file() else input_path
    try:
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        with Image.open(final_path) as image:
            output_size = image.size
        config = load_config(config_path) if config_path.is_file() else {}
        aligned, transformed = align_input_image(source, output_size, config)
    except Exception as error:
        return PreparationResult(str(run_dir), "error", message=str(error))

    result = PreparationResult(
        str(run_dir),
        "prepared" if not dry_run else "would_prepare",
        input_size=source.size,
        output_size=output_size,
        transformed=transformed,
    )
    if dry_run:
        return result

    if not backup_path.exists():
        source.save(backup_path)
    temporary_path = run_dir / ".input.aligned.tmp.png"
    try:
        aligned.save(temporary_path)
        temporary_path.replace(input_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return result


def prepare_tree(
    sketch_root: Path,
    *,
    sketch_glob: str = DEFAULT_SKETCH_GLOB,
    dry_run: bool = False,
    force: bool = False,
    show_progress: bool = True,
) -> list[PreparationResult]:
    final_paths = sorted(sketch_root.glob(sketch_glob))
    iterator = tqdm(
        final_paths,
        desc="Preparing evaluation inputs",
        unit="run",
        disable=not show_progress,
    )
    return [
        prepare_run(path.parent, dry_run=dry_run, force=force)
        for path in iterator
    ]


def summarize(results: list[PreparationResult]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return {
        "counts": counts,
        "results": [asdict(result) for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild input.png for existing ControlSketch runs so it has the "
            "same coordinate frame and resolution as final_sketch.png. The old "
            "optimization input is preserved as optimization_input.png."
        )
    )
    parser.add_argument("sketch_root", type=Path, help="ControlSketch output root")
    parser.add_argument(
        "--sketch-glob",
        default=DEFAULT_SKETCH_GLOB,
        help=f"Glob relative to sketch_root. Default: {DEFAULT_SKETCH_GLOB}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate input.png from an existing optimization_input.png.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.sketch_root.is_dir():
        raise SystemExit(f"Sketch root does not exist: {args.sketch_root}")

    results = prepare_tree(
        args.sketch_root,
        sketch_glob=args.sketch_glob,
        dry_run=args.dry_run,
        force=args.force,
        show_progress=not args.no_progress,
    )
    report = summarize(results)
    report["sketch_root"] = str(args.sketch_root)
    report["sketch_glob"] = args.sketch_glob
    report["dry_run"] = args.dry_run
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if report["counts"].get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
