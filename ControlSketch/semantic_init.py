import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_SEMANTIC_WEIGHTS = {
    "outline": 1.0,
    "eyes": 3.0,
    "eye": 3.0,
    "nose": 2.0,
    "mouth": 2.0,
    "ears": 1.5,
    "ear": 1.5,
}


def parse_semantic_parts(parts_text: str) -> List[str]:
    if not parts_text or not parts_text.strip():
        return ["outline"]

    parts = [part.strip().lower() for part in parts_text.split(",") if part.strip()]
    return parts or ["outline"]


def parse_semantic_weights(weights_text: str, parts: List[str]) -> Dict[str, float]:
    weights = {
        part: DEFAULT_SEMANTIC_WEIGHTS.get(part, 1.0)
        for part in parts
    }

    if not weights_text:
        return weights

    for item in weights_text.split(","):
        if "=" not in item:
            continue

        key, value = item.split("=", 1)
        key = key.strip().lower()

        try:
            weights[key] = float(value.strip())
        except ValueError:
            warnings.warn(f"Invalid semantic weight ignored: {item}")

    return weights


def allocate_points_by_score(scores: np.ndarray, total_points: int) -> np.ndarray:
    if total_points <= 0:
        return np.zeros(len(scores), dtype=np.int32)

    if len(scores) == 0:
        return np.zeros(0, dtype=np.int32)

    scores = np.asarray(scores, dtype=np.float64)
    if scores.sum() <= 0:
        scores = np.ones_like(scores)

    raw = scores / scores.sum() * total_points
    allocation = np.floor(raw).astype(np.int32)

    remaining = total_points - int(allocation.sum())
    if remaining > 0:
        fractions = raw - allocation
        order = np.argsort(-fractions)
        for index in order[:remaining]:
            allocation[index] += 1

    return allocation


def _mask_to_numpy(mask, target_size: Tuple[int, int]) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        tensor = mask.detach().float().cpu()

        while tensor.ndim > 2:
            tensor = tensor.squeeze(0)

        if tuple(tensor.shape[-2:]) != tuple(target_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
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
            arr = cv2.resize(
                arr,
                (target_size[1], target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )

    if arr.max() > 1.0:
        arr = arr / 255.0

    return (arr >= 0.5).astype(np.uint8)


def _contour_perimeter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0

    closed = np.vstack([points, points[0]])
    distances = np.linalg.norm(closed[1:] - closed[:-1], axis=1)
    return float(distances.sum())


def _find_contours(binary_mask: np.ndarray, min_perimeter: float) -> List[np.ndarray]:
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    valid_contours = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float32)
        if _contour_perimeter(points) >= min_perimeter:
            valid_contours.append(points)

    valid_contours.sort(key=_contour_perimeter, reverse=True)
    return valid_contours


def _sample_closed_contour(points: np.ndarray, num_points: int) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if len(points) == 1:
        return np.repeat(points.astype(np.float32), num_points, axis=0)

    closed = np.vstack([points, points[0]])
    starts = closed[:-1]
    vectors = closed[1:] - closed[:-1]
    lengths = np.linalg.norm(vectors, axis=1)

    valid = lengths > 1e-6
    starts = starts[valid]
    vectors = vectors[valid]
    lengths = lengths[valid]

    total_length = lengths.sum()
    if total_length <= 1e-6:
        return np.repeat(points[:1].astype(np.float32), num_points, axis=0)

    cumulative = np.cumsum(lengths)
    samples = np.linspace(0.0, total_length, num_points, endpoint=False)

    segment_indices = np.searchsorted(cumulative, samples, side="right")
    segment_indices = np.clip(segment_indices, 0, len(lengths) - 1)

    previous = np.concatenate([[0.0], cumulative[:-1]])
    local = samples - previous[segment_indices]
    ratio = local / lengths[segment_indices]

    sampled = starts[segment_indices] + vectors[segment_indices] * ratio[:, None]
    return sampled.astype(np.float32)


def _make_visualization(
    foreground_mask: np.ndarray,
    regions: List[Dict],
    allocations: np.ndarray,
) -> np.ndarray:
    h, w = foreground_mask.shape
    vis = np.full((h, w, 3), 255, dtype=np.uint8)
    vis[foreground_mask > 0] = np.array([245, 245, 245], dtype=np.uint8)

    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (255, 190, 11),
        (0, 119, 182),
    ]

    for index, region in enumerate(regions):
        color = palette[index % len(palette)]
        contour = np.round(region["contour"]).astype(np.int32).reshape(-1, 1, 2)
        thickness = 2 if allocations[index] > 0 else 1
        cv2.drawContours(vis, [contour], -1, color, thickness)

    return vis


def build_semantic_initial_points(
    mask,
    total_points: int,
    canvas_width: int,
    canvas_height: int,
    parts_text: str,
    weights_text: str,
    part_masks: Optional[Dict[str, object]] = None,
    min_perimeter: float = 8.0,
    curvature_sampling: bool = False,
    curvature_weight: float = 2.0,
    curvature_window: int = 6,
    min_sampling_density: float = 0.20,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    target_size = (canvas_height, canvas_width)
    foreground_mask = _mask_to_numpy(mask, target_size)

    parts = parse_semantic_parts(parts_text)
    weights = parse_semantic_weights(weights_text, parts)

    regions = []

    for part in parts:
        if part == "outline":
            part_binary = foreground_mask
        elif part_masks and part in part_masks:
            part_binary = _mask_to_numpy(part_masks[part], target_size)
            part_binary = np.logical_and(part_binary > 0, foreground_mask > 0).astype(np.uint8)
        else:
            warnings.warn(
                f"Semantic part '{part}' has no mask yet. "
                "Only 'outline' is supported before Grounded-SAM integration."
            )
            continue

        contours = _find_contours(part_binary, min_perimeter)
        for contour in contours:
            perimeter = _contour_perimeter(contour)
            weight = weights.get(part, 1.0)
            regions.append({
                "part": part,
                "contour": contour,
                "perimeter": perimeter,
                "weight": weight,
                "score": perimeter * weight,
            })

    if not regions:
        return None

    scores = np.array([region["score"] for region in regions], dtype=np.float64)
    allocations = allocate_points_by_score(scores, total_points)

    sampled_points = []
    for region, num_region_points in zip(regions, allocations):
        if curvature_sampling:
            sampled = _sample_closed_contour_curvature_weighted(
                region["contour"],
                int(num_region_points),
                curvature_weight=curvature_weight,
                curvature_window=curvature_window,
                min_density=min_sampling_density,
            )
        else:
            sampled = _sample_closed_contour(region["contour"], int(num_region_points))
        if len(sampled) > 0:
            sampled_points.append(sampled)

    if not sampled_points:
        return None

    points = np.concatenate(sampled_points, axis=0).astype(np.float32)

    if len(points) != total_points:
        points = points[:total_points]

    vis = _make_visualization(foreground_mask, regions, allocations)
    return points, vis

def _discrete_curvature(points: np.ndarray, window: int = 6) -> np.ndarray:
    n = len(points)
    if n < 3:
        return np.zeros(n, dtype=np.float32)

    window = max(1, min(window, n // 3))

    prev_points = np.roll(points, window, axis=0)
    next_points = np.roll(points, -window, axis=0)

    v1 = points - prev_points
    v2 = next_points - points

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-6) & (n2 > 1e-6)

    curvature = np.zeros(n, dtype=np.float32)
    dots = np.sum(v1[valid] * v2[valid], axis=1) / (n1[valid] * n2[valid])
    dots = np.clip(dots, -1.0, 1.0)

    curvature[valid] = np.arccos(dots).astype(np.float32)
    return curvature


def _sample_closed_contour_curvature_weighted(
    points: np.ndarray,
    num_points: int,
    curvature_weight: float = 2.0,
    curvature_window: int = 6,
    min_density: float = 0.20,
) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    if len(points) < 3:
        return _sample_closed_contour(points, num_points)

    closed = np.vstack([points, points[0]])
    starts = closed[:-1]
    vectors = closed[1:] - closed[:-1]
    lengths = np.linalg.norm(vectors, axis=1)

    valid = lengths > 1e-6
    if not np.any(valid):
        return np.repeat(points[:1].astype(np.float32), num_points, axis=0)

    curvature = _discrete_curvature(points, window=curvature_window)

    scale = np.percentile(curvature, 90)
    if scale > 1e-6:
        curvature_norm = np.clip(curvature / scale, 0.0, 1.0)
    else:
        curvature_norm = np.zeros_like(curvature)

    point_density = min_density + curvature_weight * curvature_norm
    segment_density = 0.5 * (point_density + np.roll(point_density, -1))

    starts = starts[valid]
    vectors = vectors[valid]
    lengths = lengths[valid]
    segment_density = segment_density[valid]

    weighted_lengths = lengths * segment_density
    total_weighted_length = weighted_lengths.sum()

    if total_weighted_length <= 1e-6:
        return _sample_closed_contour(points, num_points)

    cumulative = np.cumsum(weighted_lengths)
    samples = np.linspace(0.0, total_weighted_length, num_points, endpoint=False)

    segment_indices = np.searchsorted(cumulative, samples, side="right")
    segment_indices = np.clip(segment_indices, 0, len(weighted_lengths) - 1)

    previous = np.concatenate([[0.0], cumulative[:-1]])
    local = samples - previous[segment_indices]
    ratio = local / weighted_lengths[segment_indices]

    sampled = starts[segment_indices] + vectors[segment_indices] * ratio[:, None]
    return sampled.astype(np.float32)