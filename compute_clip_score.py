#!/usr/bin/env python3
"""Compute a CLIP image-image score between an input image and a sketch."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CLIP_ROOT = PROJECT_ROOT / "SwiftSketch" / "CLIP_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the CLIP similarity score between a reference image and a "
            "sketch image."
        )
    )
    parser.add_argument("image", type=Path, help="Path to the input/reference image.")
    parser.add_argument("sketch", type=Path, help="Path to the sketch image.")
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        help=(
            "CLIP model name or path to a local CLIP .pt checkpoint. "
            "Default: ViT-B/32."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for CLIP inference. Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--jit",
        action="store_true",
        help="Load CLIP as a JIT model. By default the non-JIT model is used.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def image_to_rgb_on_white(image: Any) -> Any:
    """Convert PIL image to RGB, compositing transparent pixels on white."""
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image)
    has_alpha = image.mode in {"RGBA", "LA"} or "transparency" in image.info
    if not has_alpha:
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    return white.convert("RGB")


def load_image(path: Path) -> Any:
    from PIL import Image

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if path.suffix.lower() == ".svg":
        try:
            import cairosvg
        except ImportError as exc:
            raise ImportError(
                "SVG input requires cairosvg. Install it or convert the sketch to PNG."
            ) from exc

        png_bytes = cairosvg.svg2png(url=str(path))
        return image_to_rgb_on_white(Image.open(io.BytesIO(png_bytes)))

    return image_to_rgb_on_white(Image.open(path))


def import_clip_module() -> Any:
    if LOCAL_CLIP_ROOT.exists() and str(LOCAL_CLIP_ROOT) not in sys.path:
        sys.path.insert(0, str(LOCAL_CLIP_ROOT))

    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            "CLIP dependencies are not available. Install this project's "
            "requirements, including torch, torchvision, and ftfy."
        ) from exc

    return clip


def compute_clip_score(
    image_path: Path,
    sketch_path: Path,
    model_name: str,
    device: str,
    jit: bool,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to compute CLIP scores. Install the project "
            "requirements or run this script in the project's Python environment."
        ) from exc

    clip = import_clip_module()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    model, preprocess = clip.load(model_name, device=device, jit=jit)
    model.eval()

    images = torch.stack(
        [
            preprocess(load_image(image_path)),
            preprocess(load_image(sketch_path)),
        ],
        dim=0,
    ).to(device)

    features = model.encode_image(images).float()
    features = features / features.norm(dim=-1, keepdim=True)

    cosine_similarity = (features[0] @ features[1]).item()

    clip_score = cosine_similarity * 100.0

    return {
        "image": str(image_path),
        "sketch": str(sketch_path),
        "model": model_name,
        "device": device,
        "cosine_similarity": cosine_similarity,
        "clip_score_x100": clip_score,
    }


def main() -> int:
    args = parse_args()
    try:
        result = compute_clip_score(
            image_path=args.image,
            sketch_path=args.sketch,
            model_name=args.model,
            device=args.device,
            jit=args.jit,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"image: {result['image']}")
    print(f"sketch: {result['sketch']}")
    print(f"model: {result['model']}")
    print(f"device: {result['device']}")
    print(f"cosine_similarity: {result['cosine_similarity']:.6f}")
    print(f"clip_score_x100: {result['clip_score_x100']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
