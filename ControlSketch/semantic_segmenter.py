import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
import os


def _to_binary_mask(mask, target_size):
    if isinstance(mask, torch.Tensor):
        tensor = mask.detach().float().cpu()
        while tensor.ndim > 2:
            tensor = tensor.squeeze(0)
        if tuple(tensor.shape[-2:]) != tuple(target_size):
            tensor = F.interpolate(
                tensor[None, None],
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        arr = tensor.numpy()
    else:
        arr = np.asarray(mask, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if tuple(arr.shape[:2]) != tuple(target_size):
            arr = cv2.resize(arr, (target_size[1], target_size[0]))

    if arr.max() > 1.0:
        arr = arr / 255.0
    return (arr >= 0.5).astype(np.uint8)


class GroundedSAMSegmenter:
    def __init__(
        self,
        device,
        grounding_model_id="IDEA-Research/grounding-dino-base",
        sam_model_id="facebook/sam-vit-base",
        box_threshold=0.25,
        text_threshold=0.20,
        min_area_ratio=0.0002,
        max_masks_per_part=4,
        debug_dir=None,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_area_ratio = min_area_ratio
        self.max_masks_per_part = max_masks_per_part

        self.grounding_processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model_id
        ).to(device)
        self.grounding_model.eval()

        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(device)
        self.sam_model.eval()
        
        self.debug_dir = debug_dir
        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

    def segment_parts(self, image, parts, foreground_mask=None, object_name=""):
        image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
        h, w = image.height, image.width

        if foreground_mask is not None:
            fg = _to_binary_mask(foreground_mask, (h, w))
        else:
            fg = np.ones((h, w), dtype=np.uint8)

        part_masks = {}

        for part in parts:
            boxes, scores = self._detect_boxes(image, part, object_name)

            print(
                f"[grounded_sam] part={part} raw_boxes={len(boxes)} "
                f"scores={[round(float(s), 3) for s in scores]}",
                flush=True,
            )

            self._save_grounding_boxes(image, part, boxes, scores, "raw")

            if boxes.numel() == 0:
                warnings.warn(f"Grounding DINO found no boxes for part: {part}")
                continue

            boxes, scores = self._keep_top_boxes(boxes, scores)
            self._save_grounding_boxes(image, part, boxes, scores, "used")
    
            masks = self._segment_boxes(image, boxes)

            merged = np.zeros((h, w), dtype=np.uint8)
            min_area = int(h * w * self.min_area_ratio)

            for mask in masks:
                mask_np = mask.astype(np.uint8)
                mask_np = np.logical_and(mask_np > 0, fg > 0).astype(np.uint8)

                if int(mask_np.sum()) < min_area:
                    continue

                merged = np.logical_or(merged > 0, mask_np > 0).astype(np.uint8)

            if merged.sum() > 0:
                part_masks[part] = torch.from_numpy(merged).float()
            else:
                warnings.warn(f"SAM masks for part '{part}' were filtered out.")

        return part_masks

    def _detect_boxes(self, image, part, object_name):
        query = self._make_query(part, object_name)
        text = f"{query}."

        inputs = self.grounding_processor(
            images=image,
            text=text,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        target_sizes = [(image.height, image.width)]

        try:
            results = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            results = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )

        result = results[0]
        boxes = result.get("boxes", torch.empty((0, 4))).detach().cpu()
        scores = result.get("scores", torch.empty((0,))).detach().cpu()
        return boxes, scores

    def _segment_boxes(self, image, boxes):
        input_boxes = [boxes.tolist()]

        inputs = self.sam_processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam_model(**inputs, multimask_output=False)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]

        # Common shapes:
        # [num_boxes, 1, H, W] or [num_boxes, H, W]
        if masks.ndim == 4:
            masks = masks[:, 0]
        elif masks.ndim == 2:
            masks = masks.unsqueeze(0)

        return [(mask > 0).numpy().astype(np.uint8) for mask in masks]

    def _keep_top_boxes(self, boxes, scores):
        if len(boxes) <= self.max_masks_per_part:
            return boxes, scores

        order = torch.argsort(scores, descending=True)[:self.max_masks_per_part]
        return boxes[order], scores[order]

    def _make_query(self, part, object_name):
        part = part.replace("_", " ").lower().strip()
        object_name = object_name.lower().strip()

        if object_name:
            return f"{object_name} {part}"
        return part

    def _save_grounding_boxes(self, image, part, boxes, scores, suffix):
        if self.debug_dir is None:
            return

        image_np = np.array(image.convert("RGB")).copy()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
            score = float(scores[i]) if i < len(scores) else 0.0

            cv2.rectangle(image_np, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(
                image_np,
                f"{part}:{score:.2f}",
                (x0, max(0, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        save_path = os.path.join(self.debug_dir, f"grounding_{part}_{suffix}.jpg")
        cv2.imwrite(save_path, image_np)
        print(f"[grounded_sam] saved boxes: {save_path}", flush=True)