import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CATEGORIES = ("bird", "cat", "dog", "fish")
DEFAULT_STROKES = (32, 24, 16, 8)
_NUMPY_UNAVAILABLE_WARNED = False


def has_svg_key(npz_path, stroke_count):
    global _NUMPY_UNAVAILABLE_WARNED

    key = f"svg_{stroke_count}s"
    try:
        import numpy as np

        with np.load(npz_path, allow_pickle=True) as data:
            return key in data.files
    except ModuleNotFoundError:
        if not _NUMPY_UNAVAILABLE_WARNED:
            print(
                "[warn] numpy is not available, so existing svg_<N>s keys "
                "cannot be inspected. Activate the SwiftSketch environment "
                "before running real generation.",
                flush=True,
            )
            _NUMPY_UNAVAILABLE_WARNED = True
        return False
    except Exception as exc:
        print(f"[warn] could not inspect {npz_path}: {exc}", flush=True)
        return False


def parse_min_seeds(values):
    min_seeds = {}
    for value in values:
        try:
            category, seed = value.split("=", 1)
            min_seeds[category] = int(seed)
        except ValueError as exc:
            raise ValueError(
                f"invalid --min_seed_by_category value {value!r}; "
                "expected CATEGORY=SEED"
            ) from exc
    return min_seeds


def sample_seed(npz_path):
    try:
        return int(npz_path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def collect_targets(input_root, categories, min_seeds):
    targets = []
    for category in categories:
        category_dir = input_root / category
        if not category_dir.is_dir():
            print(f"[warn] missing category directory: {category_dir}", flush=True)
            continue
        for npz_path in sorted(category_dir.glob("*.npz")):
            minimum_seed = min_seeds.get(category)
            if minimum_seed is not None:
                seed = sample_seed(npz_path)
                if seed is None:
                    print(f"[warn] could not parse seed from {npz_path}", flush=True)
                    continue
                if seed < minimum_seed:
                    continue
            targets.append((category, npz_path))
    return targets


def has_completed_output(output_dir, target_path, stroke_count):
    stem = target_path.stem
    final_svg = output_dir / stem / f"{stem}_{stroke_count}_strokes" / "final_svg.svg"
    return final_svg.is_file()


def build_command(args, target_path, category, stroke_count):
    command = [
        sys.executable,
        str(args.repo_root / "ControlSketch" / "object_sketching.py"),
        "--target",
        str(target_path),
        "--num_strokes",
        str(stroke_count),
        "--output_dir",
        str(args.output_dir),
        "--wandb_name",
        f"{target_path.stem}_{stroke_count}_strokes",
        "--use_wandb",
        "0",
        "--save_svg_in_dict",
        "1",
        "--num_iter",
        str(args.num_iter),
        "--save_interval",
        str(args.save_interval),
        "--condition",
        args.condition,
        "--conditioning_scale",
        str(args.conditioning_scale),
        "--diffusion_guidance_scale",
        str(args.diffusion_guidance_scale),
        "--seed",
        str(args.seed),
    ]
    if args.use_cpu:
        command.extend(["--use_cpu", "1"])
    if args.object_name_from_category:
        command.extend(["--object_name", category])
    command.extend(args.extra_args)
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Generate ControlSketch SVGs for controlsketch_var_data/train."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("controlsketch_var_data/train"),
        help="Root directory containing category subdirectories with .npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("controlsketch_var_data/controlsketch_outputs"),
        help="Directory where ControlSketch run artifacts are saved.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Category subdirectories to process.",
    )
    parser.add_argument(
        "--strokes",
        nargs="+",
        type=int,
        default=DEFAULT_STROKES,
        help="Stroke counts to generate for each sample.",
    )
    parser.add_argument(
        "--min_seed_by_category",
        nargs="*",
        default=(),
        metavar="CATEGORY=SEED",
        help="Only process samples at or above each category's minimum seed.",
    )
    parser.add_argument("--num_iter", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--condition", default="depth")
    parser.add_argument("--conditioning_scale", type=float, default=0.15)
    parser.add_argument("--diffusion_guidance_scale", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument(
        "--object_name_from_category",
        action="store_true",
        help="Pass the category name as --object_name to object_sketching.py.",
    )
    parser.add_argument(
        "--rerun_existing",
        action="store_true",
        help="Regenerate sketches even when svg_<N>s already exists in the .npz.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned jobs without running ControlSketch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run at most this many ControlSketch jobs. Useful for smoke tests.",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments appended to every object_sketching.py invocation.",
    )
    args = parser.parse_args()

    args.repo_root = Path(__file__).resolve().parent
    args.input_root = (args.repo_root / args.input_root).resolve()
    args.output_dir = (args.repo_root / args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        min_seeds = parse_min_seeds(args.min_seed_by_category)
    except ValueError as exc:
        parser.error(str(exc))

    targets = collect_targets(args.input_root, args.categories, min_seeds)
    if not targets:
        print(f"No .npz files found under {args.input_root}", flush=True)
        return 1

    jobs = []
    skipped = 0
    for category, target_path in targets:
        for stroke_count in args.strokes:
            if not args.rerun_existing and (
                has_svg_key(target_path, stroke_count)
                or has_completed_output(args.output_dir, target_path, stroke_count)
            ):
                skipped += 1
                continue
            jobs.append((category, target_path, stroke_count))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    total_possible = len(targets) * len(args.strokes)
    print(
        f"targets={len(targets)} total_possible_jobs={total_possible} "
        f"skipped_existing={skipped} jobs_to_run={len(jobs)}",
        flush=True,
    )

    if args.dry_run:
        for category, target_path, stroke_count in jobs:
            print(f"[dry-run] {category} {target_path} {stroke_count} strokes", flush=True)
        return 0

    for index, (category, target_path, stroke_count) in enumerate(jobs, start=1):
        print(
            f"[{index}/{len(jobs)}] {category} {target_path.name} "
            f"{stroke_count} strokes",
            flush=True,
        )
        command = build_command(args, target_path, category, stroke_count)
        result = subprocess.run(command, cwd=args.repo_root)
        if result.returncode != 0:
            print(
                f"[error] failed: {target_path} ({stroke_count} strokes), "
                f"exit_code={result.returncode}",
                flush=True,
            )
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
