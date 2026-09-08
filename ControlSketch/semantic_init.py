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


def allocate(
    weights,
    n_total: int,
    min_per_part: int = 1,
    w_thresh: float = 0.05,
) -> np.ndarray:
    """Allocate an exact integer budget with lower bounds and largest remainders."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if n_total < 0 or min_per_part < 0:
        raise ValueError("n_total and min_per_part must be non-negative")
    if len(weights) == 0:
        if n_total:
            raise ValueError("cannot allocate a positive budget to no parts")
        return np.zeros(0, dtype=np.int32)

    weights = np.where(np.isfinite(weights) & (weights >= 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    allocation = np.zeros(len(weights), dtype=np.int32)

    eligible = np.flatnonzero(weights >= w_thresh)
    eligible = eligible[np.argsort(-weights[eligible], kind="stable")]
    remaining = int(n_total)
    for index in eligible:
        if remaining < min_per_part:
            break
        allocation[index] += min_per_part
        remaining -= min_per_part

    if remaining:
        raw = weights * remaining
        extra = np.floor(raw).astype(np.int32)
        allocation += extra
        remainder = remaining - int(extra.sum())
        if remainder:
            fractions = raw - extra
            order = np.argsort(-fractions, kind="stable")
            allocation[order[:remainder]] += 1

    assert int(allocation.sum()) == n_total
    return allocation


def allocate_points_by_score(scores: np.ndarray, total_points: int) -> np.ndarray:
    """Backward-compatible unconstrained largest-remainder allocation."""
    return allocate(scores, total_points, min_per_part=0, w_thresh=np.inf)


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


def _polyline_length(points: np.ndarray, closed: bool) -> float:
    if len(points) < 2:
        return 0.0

    if closed:
        points = np.vstack([points, points[0]])

    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def _sample_open_contour(points: np.ndarray, num_points: int) -> np.ndarray:
    """開いた輪郭を弧長に沿って等間隔にサンプリングする。"""
    if num_points <= 0 or len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if len(points) == 1:
        return np.repeat(points.astype(np.float32), num_points, axis=0)

    starts = points[:-1]
    vectors = points[1:] - points[:-1]
    lengths = np.linalg.norm(vectors, axis=1)

    valid = lengths > 1e-6
    starts = starts[valid]
    vectors = vectors[valid]
    lengths = lengths[valid]

    if len(lengths) == 0:
        return np.repeat(points[:1].astype(np.float32), num_points, axis=0)

    total_length = float(lengths.sum())

    if num_points == 1:
        samples = np.array([total_length / 2.0], dtype=np.float32)
    else:
        samples = np.linspace(
            0.0,
            total_length,
            num_points,
            endpoint=True,
            dtype=np.float32,
        )

    cumulative = np.cumsum(lengths)
    segment_indices = np.searchsorted(cumulative, samples, side="right")
    segment_indices = np.clip(segment_indices, 0, len(lengths) - 1)

    previous = np.concatenate([[0.0], cumulative[:-1]])
    local = samples - previous[segment_indices]
    ratio = local / lengths[segment_indices]

    sampled = (
        starts[segment_indices]
        + vectors[segment_indices] * ratio[:, None]
    )
    return sampled.astype(np.float32)


def _split_contour_by_keep_mask(
    points: np.ndarray,
    keep_mask: np.ndarray,
) -> List[np.ndarray]:
    """
    閉じた輪郭からkeep_mask=Trueの連続区間を取り出す。

    最初と最後が同じ区間に属する場合も、輪郭が循環していることを
    考慮して正しくまとめる。
    """
    points = np.asarray(points, dtype=np.float32)
    keep_mask = np.asarray(keep_mask, dtype=bool)

    if len(points) == 0 or not np.any(keep_mask):
        return []

    if np.all(keep_mask):
        return [points]

    # Falseの直後から始めることで、配列の先頭と末尾をまたぐ
    # True区間が分断されることを防ぐ。
    false_index = int(np.flatnonzero(~keep_mask)[0])
    start_index = (false_index + 1) % len(points)

    points = np.roll(points, -start_index, axis=0)
    keep_mask = np.roll(keep_mask, -start_index)

    runs = []
    run_start = None

    for index, keep in enumerate(keep_mask):
        if keep and run_start is None:
            run_start = index
        elif not keep and run_start is not None:
            runs.append(points[run_start:index])
            run_start = None

    if run_start is not None:
        runs.append(points[run_start:])

    return [run for run in runs if len(run) >= 2]


def _make_outline_distance_map(foreground_mask: np.ndarray) -> np.ndarray:
    """各画素から概形輪郭までの距離を計算する。"""
    kernel = np.ones((3, 3), dtype=np.uint8)

    eroded = cv2.erode(
        foreground_mask.astype(np.uint8),
        kernel,
        iterations=1,
    )

    # オブジェクトの内側にある1画素幅の概形輪郭
    outline_boundary = (
        foreground_mask.astype(np.uint8) - eroded
    ) > 0

    # distanceTransformは0画素までの距離を返す。
    distance_input = (~outline_boundary).astype(np.uint8)

    return cv2.distanceTransform(
        distance_input,
        cv2.DIST_L2,
        5,
    )


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
        cv2.polylines(
            vis,
            [contour],
            isClosed=region.get("closed", True),
            color=color,
            thickness=thickness,
        )

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
    outline_overlap_tolerance: float = 4.0,
    curvature_sampling: bool = False,
    curvature_weight: float = 2.0,
    curvature_window: int = 6,
    min_sampling_density: float = 0.20,
    return_metadata: bool = False,
    return_analysis_contours: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    target_size = (canvas_height, canvas_width)
    foreground_mask = _mask_to_numpy(mask, target_size)
    
    outline_distance = _make_outline_distance_map(foreground_mask)

    parts = parse_semantic_parts(parts_text)
    weights = parse_semantic_weights(weights_text, parts)

    regions = []
    part_masks_for_loss = {}

    for part in parts:
        if part == "outline":
            part_binary = foreground_mask
        elif part_masks and part in part_masks:
            part_binary = _mask_to_numpy(part_masks[part], target_size)
            part_binary = np.logical_and(part_binary > 0, foreground_mask > 0).astype(np.uint8)
        else:
            warnings.warn(
                f"Semantic part '{part}' has no mask yet. "
                "Only 'outline' is supported before semantic segmenter integration."
            )
            continue
        
        part_masks_for_loss[part] = part_binary
        contours = _find_contours(part_binary, min_perimeter)
        weight = weights.get(part, 1.0)

        for contour in contours:
            if part == "outline" or outline_overlap_tolerance <= 0:
                # 概形輪郭はそのまま閉曲線として登録する。
                candidate_contours = [(contour, True)]
            else:
                rounded = np.round(contour).astype(np.int32)

                xs = np.clip(
                    rounded[:, 0],
                    0,
                    canvas_width - 1,
                )
                ys = np.clip(
                    rounded[:, 1],
                    0,
                    canvas_height - 1,
                )

                distances = outline_distance[ys, xs]

                # 概形輪郭から指定距離より離れた部分だけを残す。
                keep_mask = distances > outline_overlap_tolerance

                open_contours = _split_contour_by_keep_mask(
                    contour,
                    keep_mask,
                )
                candidate_contours = [
                    (open_contour, False)
                    for open_contour in open_contours
                ]

            for candidate, is_closed in candidate_contours:
                length = _polyline_length(candidate, closed=is_closed)

                # 重複排除によって生じた短い断片を除外する。
                if length < min_perimeter:
                    continue

                regions.append({
                    "part": part,
                    "contour": candidate,
                    "closed": is_closed,
                    "perimeter": length,
                    "weight": weight,
                    "score": length * weight,
                })

    if not regions:
        return None

    part_order = list(dict.fromkeys(region["part"] for region in regions))
    part_weights = np.array([
        next(region["weight"] for region in regions if region["part"] == part)
        for part in part_order
    ], dtype=np.float64)
    if "outline" in part_order and len(part_order) > 1:
        outline_index = part_order.index("outline")
        semantic_indices = [
            index for index in range(len(part_order))
            if index != outline_index
        ]
        grouped = allocate(
            [part_weights[outline_index], part_weights[semantic_indices].sum()],
            total_points, min_per_part=0, w_thresh=np.inf,
        )
        part_allocations = np.zeros(len(part_order), dtype=np.int32)
        part_allocations[outline_index] = grouped[0]
        part_allocations[semantic_indices] = allocate(
            part_weights[semantic_indices], int(grouped[1]),
            min_per_part=1, w_thresh=0.05,
        )
    else:
        part_allocations = allocate(
            part_weights, total_points, min_per_part=1, w_thresh=0.05
        )
    allocations = np.zeros(len(regions), dtype=np.int32)
    for part, part_count in zip(part_order, part_allocations):
        region_indices = [
            index for index, region in enumerate(regions)
            if region["part"] == part
        ]
        perimeters = np.array([
            regions[index]["perimeter"] for index in region_indices
        ], dtype=np.float64)
        instance_allocations = allocate(
            perimeters, int(part_count), min_per_part=0, w_thresh=np.inf
        )
        allocations[region_indices] = instance_allocations
    assert int(allocations.sum()) == total_points

    sampled_points = []
    sampled_parts = []

    for region, num_region_points in zip(regions, allocations):
        num_region_points = int(num_region_points)

        if region["closed"]:
            if curvature_sampling:
                sampled = _sample_closed_contour_curvature_weighted(
                    region["contour"],
                    num_region_points,
                    curvature_weight=curvature_weight,
                    curvature_window=curvature_window,
                    min_density=min_sampling_density,
                )
            else:
                sampled = _sample_closed_contour(
                    region["contour"],
                    num_region_points,
                )
        else:
            # 重複排除後の部位輪郭は開曲線になる。
            sampled = _sample_open_contour(
                region["contour"],
                num_region_points,
            )

        if len(sampled) > 0:
            sampled_points.append(sampled)
            sampled_parts.extend(
                [region["part"]] * len(sampled)
            )

    if not sampled_points:
        return None

    points = np.concatenate(sampled_points, axis=0).astype(np.float32)

    if len(points) != total_points:
        points = points[:total_points]

    vis = _make_visualization(foreground_mask, regions, allocations)

    if return_metadata:
        if return_analysis_contours:
            analysis_contours = {}
            for region in regions:
                analysis_contours.setdefault(region["part"], []).append({
                    "points": region["contour"].copy(),
                    "closed": region["closed"],
                })
            return (
                points,
                vis,
                sampled_parts,
                part_masks_for_loss,
                analysis_contours,
            )
        return points, vis, sampled_parts, part_masks_for_loss

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
