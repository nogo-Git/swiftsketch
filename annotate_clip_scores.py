#!/usr/bin/env python3
"""Annotate images with CLIP scores from a JSON file.

The default input format matches compute_controlsketch_clip_scores.py:

  {
    "results": [
      {
        "sketch_image": "path/to/final_sketch.png",
        "clip_score_x100": 79.43
      }
    ]
  }

Each image is copied to a new file with the CLIP score drawn in the lower-right
corner. By default, final_sketch.png becomes final_sketch_clip_score.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read CLIP scores from JSON and write annotated copies of the "
            "corresponding images."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="JSON file containing image paths and CLIP scores.",
    )
    parser.add_argument(
        "--entries-key",
        default="results",
        help=(
            "Dot-separated key for the list or mapping of score entries. "
            "Use an empty string or '.' for a top-level list/mapping. "
            "Default: results."
        ),
    )
    parser.add_argument(
        "--image-key",
        default="sketch_image",
        help="Key containing the image path in each entry. Default: sketch_image.",
    )
    parser.add_argument(
        "--score-key",
        default="clip_score_x100",
        help="Key containing the score in each entry. Default: clip_score_x100.",
    )
    parser.add_argument(
        "--path-root",
        type=Path,
        default=None,
        help=(
            "Root used to resolve relative image paths. If omitted, paths are "
            "tried relative to the current directory, the JSON directory, and "
            "the project root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for annotated images. When omitted, each image "
            "is written next to its source image."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="_clip_score",
        help="Suffix added before the image extension. Default: _clip_score.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help=(
            "Text template. Available fields include score, raw_score, image, "
            "index, and any entry keys. Default: CLIP: {score:.2f}."
        ),
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places used by the default template. Default: 2.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=None,
        help="Font size in pixels. Default: proportional to image size.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=None,
        help="Margin from the image edge in pixels. Default: proportional to image size.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=None,
        help="Padding around the text in pixels. Default: proportional to font size.",
    )
    parser.add_argument(
        "--text-fill",
        default="#111111",
        help="Text color accepted by Pillow. Default: #111111.",
    )
    parser.add_argument(
        "--box-fill",
        default="#ffffff",
        help="Background box color accepted by Pillow. Default: #ffffff.",
    )
    parser.add_argument(
        "--box-alpha",
        type=int,
        default=230,
        help="Background box opacity from 0 to 255. Default: 230.",
    )
    parser.add_argument(
        "--box-outline",
        default="#111111",
        help="Background box outline color accepted by Pillow. Default: #111111.",
    )
    parser.add_argument(
        "--box-outline-alpha",
        type=int,
        default=40,
        help="Background box outline opacity from 0 to 255. Default: 40.",
    )
    parser.add_argument(
        "--no-box",
        action="store_true",
        help="Draw only the score text, without a background box.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing images.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return an error if an entry is missing a path, score, or image file.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def split_key_path(raw_key: str | None) -> list[str]:
    if raw_key is None or raw_key in {"", "."}:
        return []
    return [part for part in raw_key.split(".") if part]


def lookup_key_path(payload: Any, raw_key: str | None) -> Any:
    current = payload
    for key in split_key_path(raw_key):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(raw_key)
        current = current[key]
    return current


def is_score_map(payload: dict[str, Any], score_key: str) -> bool:
    if not payload:
        return False

    for value in payload.values():
        if isinstance(value, (int, float, str)):
            continue
        if isinstance(value, dict) and score_key in value:
            continue
        return False
    return True


def iter_score_records(
    payload: Any,
    entries_key: str | None,
    score_key: str,
) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        entries = payload
    else:
        try:
            entries = lookup_key_path(payload, entries_key)
        except KeyError:
            if isinstance(payload, dict) and is_score_map(payload, score_key):
                entries = payload
            else:
                raise

    if isinstance(entries, dict):
        for image_path, value in entries.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("_image_path", image_path)
            else:
                record = {"_image_path": image_path, "_score": value}
            yield record
        return

    if not isinstance(entries, list):
        raise TypeError("--entries-key must point to a list or mapping.")

    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Each list entry must be a JSON object.")
        yield entry


def extract_image_path(entry: dict[str, Any], image_key: str) -> str | None:
    value = entry.get(image_key, entry.get("_image_path"))
    if value is None:
        return None
    return str(value)


def extract_score(entry: dict[str, Any], score_key: str) -> float | None:
    value = entry.get(score_key, entry.get("_score"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_image_path(
    raw_path: str,
    path_root: Path | None,
    json_path: Path,
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    if path_root is not None:
        return (path_root / path).resolve()

    candidates = [
        Path.cwd() / path,
        json_path.parent / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def relative_output_path(
    image_path: Path,
    json_path: Path,
    path_root: Path | None,
) -> Path:
    roots = []
    if path_root is not None:
        roots.append(path_root.resolve())
    roots.extend([Path.cwd().resolve(), PROJECT_ROOT.resolve(), json_path.parent.resolve()])

    for root in roots:
        try:
            return image_path.resolve().relative_to(root)
        except ValueError:
            continue

    return Path(image_path.name)


def build_output_path(
    image_path: Path,
    json_path: Path,
    path_root: Path | None,
    output_dir: Path | None,
    suffix: str,
) -> Path:
    output_name = f"{image_path.stem}{suffix}{image_path.suffix}"
    if output_dir is None:
        return image_path.with_name(output_name)

    rel_path = relative_output_path(image_path, json_path, path_root)
    return output_dir / rel_path.with_name(output_name)


def clamp_alpha(value: int) -> int:
    return max(0, min(255, value))


def parse_rgba(color: str, alpha: int | None = None) -> tuple[int, int, int, int]:
    from PIL import ImageColor

    rgba = ImageColor.getcolor(color, "RGBA")
    if alpha is None:
        return rgba
    return rgba[:3] + (clamp_alpha(alpha),)


def load_font(font_size: int) -> Any:
    from PIL import ImageFont

    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_bbox(draw: Any, text: str, font: Any) -> tuple[int, int, int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def fit_font(draw: Any, text: str, image_width: int, margin: int, padding: int, font_size: int) -> Any:
    max_width = max(1, image_width - (margin * 2) - (padding * 2))
    size = font_size
    font = load_font(size)

    while size > 8:
        left, _, right, _ = text_bbox(draw, text, font)
        if right - left <= max_width:
            return font
        size -= 1
        font = load_font(size)

    return font


def annotate_image(
    image_path: Path,
    output_path: Path,
    text: str,
    args: argparse.Namespace,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    with Image.open(image_path) as source:
        source = ImageOps.exif_transpose(source)
        base = source.convert("RGBA")

    width, height = base.size
    font_size = args.font_size or max(14, round(min(width, height) * 0.045))
    margin = args.margin if args.margin is not None else max(8, round(min(width, height) * 0.025))
    padding = args.padding if args.padding is not None else max(6, round(font_size * 0.35))

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = fit_font(draw, text, width, margin, padding, font_size)

    left, top, right, bottom = text_bbox(draw, text, font)
    text_width = right - left
    text_height = bottom - top
    box_right = width - margin
    box_bottom = height - margin
    box_left = box_right - text_width - (padding * 2)
    box_top = box_bottom - text_height - (padding * 2)
    text_x = box_left + padding - left
    text_y = box_top + padding - top

    if not args.no_box:
        draw.rectangle(
            (box_left, box_top, box_right, box_bottom),
            fill=parse_rgba(args.box_fill, args.box_alpha),
            outline=parse_rgba(args.box_outline, args.box_outline_alpha),
        )

    draw.text((text_x, text_y), text, fill=parse_rgba(args.text_fill), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = Image.alpha_composite(base, overlay)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        annotated.convert("RGB").save(output_path, quality=95)
    else:
        annotated.save(output_path)


def format_text(
    template: str,
    entry: dict[str, Any],
    image_path: Path,
    score: float,
    index: int,
) -> str:
    values = dict(entry)
    values.update(
        {
            "score": score,
            "raw_score": score,
            "image": image_path.as_posix(),
            "index": index,
        }
    )
    return template.format(**values)


def annotate_from_json(args: argparse.Namespace) -> tuple[int, int, list[str]]:
    json_path = args.json_path.resolve()
    payload = load_json(json_path)
    template = args.template or f"CLIP: {{score:.{args.decimals}f}}"

    num_written = 0
    num_skipped = 0
    errors: list[str] = []

    for index, entry in enumerate(
        iter_score_records(payload, args.entries_key, args.score_key),
        start=1,
    ):
        raw_image_path = extract_image_path(entry, args.image_key)
        score = extract_score(entry, args.score_key)

        if raw_image_path is None:
            num_skipped += 1
            errors.append(f"entry {index}: missing image key '{args.image_key}'")
            continue
        if score is None:
            num_skipped += 1
            errors.append(f"entry {index}: missing or invalid score key '{args.score_key}'")
            continue

        image_path = resolve_image_path(raw_image_path, args.path_root, json_path)
        if not image_path.exists():
            num_skipped += 1
            errors.append(f"entry {index}: image not found: {raw_image_path}")
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            num_skipped += 1
            errors.append(f"entry {index}: unsupported image extension: {image_path}")
            continue

        output_path = build_output_path(
            image_path=image_path,
            json_path=json_path,
            path_root=args.path_root,
            output_dir=args.output_dir,
            suffix=args.suffix,
        )
        text = format_text(template, entry, image_path, score, index)

        if args.dry_run:
            print(f"{image_path} -> {output_path} [{text}]")
        else:
            annotate_image(image_path, output_path, text, args)
        num_written += 1

    return num_written, num_skipped, errors


def main() -> int:
    args = parse_args()

    try:
        num_written, num_skipped, errors = annotate_from_json(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for error in errors:
        print(f"warning: {error}", file=sys.stderr)

    action = "would write" if args.dry_run else "wrote"
    print(f"{action} {num_written} annotated image(s); skipped {num_skipped}.")

    if args.strict and errors:
        return 1
    if num_written == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
