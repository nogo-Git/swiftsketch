import argparse
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image


def save_npz_preview(npz_path, output_root):
    data = np.load(npz_path, allow_pickle=True)

    sample_name = npz_path.stem
    output_dir = output_root / sample_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if "image" in data:
        image = Image.open(io.BytesIO(data["image"].tobytes())).convert("RGB")
        image.save(output_dir / "image.png")

    if "mask" in data:
        mask = (data["mask"] * 255).astype("uint8")
        Image.fromarray(mask).save(output_dir / "mask.png")

    if "attn_map" in data:
        attn = data["attn_map"]
        attn = attn - attn.min()
        if attn.max() > 0:
            attn = attn / attn.max()
        attn = (attn * 255).astype("uint8")
        Image.fromarray(attn).save(output_dir / "attn_map.png")

    print(f"saved: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing .npz files")
    parser.add_argument(
        "--output_dir",
        default="preview",
        help="Directory to save preview images",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return

    for npz_path in npz_files:
        save_npz_preview(npz_path, output_root)

    print(f"finished: {len(npz_files)} files")


if __name__ == "__main__":
    main()