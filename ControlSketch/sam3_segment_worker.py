import argparse
import json
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parts_json", required=True)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_sam3_repo = os.path.join(script_dir, "sam3")
    if os.path.isdir(os.path.join(local_sam3_repo, "sam3")):
        sys.path.insert(0, local_sam3_repo)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    with open(args.parts_json, "r", encoding="utf-8") as f:
        parts = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_context = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    checkpoint_path = args.checkpoint_path or None

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is None,
    )
    model.eval()

    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    image = Image.open(args.image).convert("RGB")
    with inference_context():
        state = processor.set_image(image)

    save_dict = {}
    meta = {}

    for item in parts:
        part = item["part"]
        prompt = item["prompt"]

        with inference_context():
            output = processor.set_text_prompt(state=state, prompt=prompt)

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            meta[part] = {"prompt": prompt, "num_masks": 0}
            continue

        masks = masks.detach().cpu()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]

        if torch.is_floating_point(masks):
            masks_np = masks.to(torch.float32).numpy()
        else:
            masks_np = masks.numpy()
        boxes_np = boxes.detach().to(torch.float32).cpu().numpy()
        scores_cpu = scores.detach().to(torch.float32).cpu()
        save_dict[f"{part}__masks"] = (masks_np > 0).astype(np.uint8)
        save_dict[f"{part}__boxes"] = boxes_np
        save_dict[f"{part}__scores"] = scores_cpu.numpy()

        meta[part] = {
            "prompt": prompt,
            "num_masks": int(masks.shape[0]),
            "scores": [float(s) for s in scores_cpu],
        }

    save_dict["__meta__"] = np.array(json.dumps(meta), dtype=object)
    np.savez_compressed(args.output, **save_dict)


if __name__ == "__main__":
    main()