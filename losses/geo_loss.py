"""Differentiable contour geometry losses for ControlSketch.

Reference fields are built once on the CPU.  All operations in ``GeoLoss``
remain differentiable with respect to the cubic Bezier control points.
"""

import math
from typing import Dict, Iterable, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    label,
)


TensorOrArray = Union[torch.Tensor, np.ndarray]


def bezier_polyline(P: torch.Tensor, M: int = 16) -> torch.Tensor:
    """Approximate cubic Beziers ``P:[n,4,2]`` by ``M`` line segments."""
    if P.ndim != 3 or P.shape[1:] != (4, 2):
        raise ValueError(f"P must have shape [n,4,2], got {tuple(P.shape)}")
    if M < 1:
        raise ValueError("M must be at least 1")

    t = torch.linspace(0.0, 1.0, M + 1, device=P.device, dtype=P.dtype)
    t = t.view(1, M + 1, 1)
    one_minus_t = 1.0 - t
    return (
        one_minus_t.pow(3) * P[:, 0:1]
        + 3.0 * one_minus_t.pow(2) * t * P[:, 1:2]
        + 3.0 * one_minus_t * t.pow(2) * P[:, 2:3]
        + t.pow(3) * P[:, 3:4]
    )


def pseudo_huber(d: torch.Tensor, tau: float) -> torch.Tensor:
    """Return ``tau^2 * (sqrt(1 + d^2/tau^2) - 1)``."""
    tau_tensor = torch.as_tensor(tau, dtype=d.dtype, device=d.device)
    if torch.any(tau_tensor <= 0):
        raise ValueError("tau must be positive")
    return tau_tensor.square() * (
        torch.sqrt(1.0 + (d / tau_tensor).square()) - 1.0
    )


def sample_field(
    field: torch.Tensor,
    pts: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """Bilinearly sample an ``[H,W]`` or ``[H,W,C]`` field at pixel points."""
    height, width = size
    if tuple(field.shape[:2]) != (height, width):
        raise ValueError(
            f"field spatial shape {tuple(field.shape[:2])} != size {size}"
        )
    if pts.shape[-1] != 2:
        raise ValueError("pts must end in an (x, y) coordinate dimension")
    field = field.to(device=pts.device, dtype=pts.dtype)

    if field.ndim == 2:
        image = field.unsqueeze(0).unsqueeze(0)
        channels = 1
    elif field.ndim == 3:
        channels = field.shape[-1]
        image = field.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError("field must have shape [H,W] or [H,W,C]")

    original_shape = pts.shape[:-1]
    flat_pts = pts.reshape(-1, 2)
    grid_x = 2.0 * flat_pts[:, 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * flat_pts[:, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).view(channels, -1).transpose(0, 1)
    if field.ndim == 2:
        return sampled[:, 0].reshape(original_shape)
    return sampled.reshape(*original_shape, channels)


def point_to_segments_dist(
    pts: torch.Tensor,
    V: torch.Tensor,
    chunk: int = 4096,
) -> torch.Tensor:
    """Distance from each point in ``pts`` to the closest segment in ``V``."""
    if pts.ndim != 2 or pts.shape[-1] != 2:
        raise ValueError("pts must have shape [Q,2]")
    if V.ndim != 3 or V.shape[-1] != 2 or V.shape[1] < 2:
        raise ValueError("V must have shape [n,M+1,2]")
    if chunk < 1:
        raise ValueError("chunk must be at least 1")
    pts = pts.to(device=V.device, dtype=V.dtype)
    if V.shape[0] == 0:
        return torch.full(
            (pts.shape[0],), float("inf"), dtype=pts.dtype, device=pts.device
        )

    A = V[:, :-1].reshape(-1, 2)
    B = V[:, 1:].reshape(-1, 2)
    segment = B - A
    length_sq = segment.square().sum(dim=-1).clamp_min(1e-12)
    distances = []
    for start in range(0, pts.shape[0], chunk):
        point_chunk = pts[start:start + chunk]
        offset = point_chunk[:, None, :] - A[None, :, :]
        projection = (offset * segment[None, :, :]).sum(dim=-1)
        projection = (projection / length_sq[None, :]).clamp(0.0, 1.0)
        closest = A[None, :, :] + projection[..., None] * segment[None, :, :]
        distance_sq = (point_chunk[:, None, :] - closest).square().sum(dim=-1)
        distances.append(torch.sqrt(distance_sq.min(dim=1).values.clamp_min(1e-12)))
    return torch.cat(distances, dim=0)


def mask_contour(mask: TensorOrArray) -> np.ndarray:
    """Return the one-pixel-wide inner boundary of a binary mask."""
    binary = np.asarray(mask, dtype=bool)
    eroded = binary_erosion(binary, structure=np.ones((3, 3), dtype=bool))
    return binary & ~eroded


def remove_short_components(edge_mask: TensorOrArray, min_len: int = 15) -> np.ndarray:
    """Remove 8-connected components shorter than ``min_len`` pixels."""
    edge = np.asarray(edge_mask, dtype=bool)
    if min_len <= 1:
        return edge.copy()
    labels, count = label(edge, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(edge)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_len)
    keep[0] = False
    return keep[labels]


def _layer_edge(layer: TensorOrArray, size: Tuple[int, int]) -> np.ndarray:
    array = np.asarray(layer, dtype=bool)
    if array.shape != size:
        raise ValueError(f"reference layer shape {array.shape} != size {size}")
    # Filled masks retain pixels after erosion; already-thin edge maps do not.
    eroded = binary_erosion(array, structure=np.ones((3, 3), dtype=bool))
    return mask_contour(array) if eroded.any() else array


def build_reference(
    layers: Iterable[Tuple[TensorOrArray, float]],
    size: Tuple[int, int],
    min_len: int = 15,
    blur: float = 1.0,
    rho: float = 2.0,
    device: Union[str, torch.device] = "cuda",
) -> Dict[str, object]:
    """Build weighted contour points, distance field, and tangent field once."""
    height, width = size
    wmap = np.zeros((height, width), dtype=np.float32)
    for layer_mask, weight in layers:
        if weight < 0:
            raise ValueError("reference weights must be non-negative")
        edge = _layer_edge(layer_mask, size)
        wmap = np.maximum(wmap, edge.astype(np.float32) * float(weight))

    scaled_min_len = max(1, int(round(min_len * width / 224.0)))
    edge_mask = remove_short_components(wmap > 0, scaled_min_len)
    wmap *= edge_mask
    if not edge_mask.any():
        raise ValueError("reference contains no contour component after filtering")

    ys, xs = np.nonzero(edge_mask)
    edge_pts = np.stack((xs, ys), axis=-1).astype(np.float32)
    edge_w = wmap[ys, xs].astype(np.float32)

    distance, nearest = distance_transform_edt(
        ~edge_mask, return_indices=True
    )
    distance = gaussian_filter(distance.astype(np.float32), float(blur))

    image = gaussian_filter(edge_mask.astype(np.float32), 1.0)
    grad_y, grad_x = np.gradient(image)
    jxx = gaussian_filter(grad_x * grad_x, float(rho))
    jyy = gaussian_filter(grad_y * grad_y, float(rho))
    jxy = gaussian_filter(grad_x * grad_y, float(rho))
    normal_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    normal_x = np.cos(normal_angle)
    normal_y = np.sin(normal_angle)
    tangent = np.stack((-normal_y, normal_x), axis=-1).astype(np.float32)
    nearest_y, nearest_x = nearest
    tangent = tangent[nearest_y, nearest_x]

    target_device = torch.device(device)
    return {
        "edge_pts": torch.from_numpy(edge_pts).to(target_device),
        "edge_w": torch.from_numpy(edge_w).to(target_device),
        "D_E": torch.from_numpy(distance).to(target_device),
        "e_hat": torch.from_numpy(tangent).to(target_device),
        "size": (height, width),
        "edge_mask": edge_mask,
    }


class GeoLoss(torch.nn.Module):
    """Precision, coverage, tangent, and initialization-anchor losses."""

    def __init__(
        self,
        ref: Dict[str, object],
        T: int = 2000,
        M: int = 16,
        w_pos: float = 1.0,
        w_cov: float = 1.0,
        w_tan: float = 0.0,
        w_anchor: float = 10.0,
        tau0: float = 16.0,
        tau1: float = 2.0,
        t_hold: float = 0.3,
        t_decay: float = 0.1,
        geo_hold: float = 0.5,
        geo_decay: float = 0.2,
        cov_subsample: int = 4096,
        chunk: int = 4096,
    ):
        super().__init__()
        if T <= 0 or M <= 0 or tau0 <= 0 or tau1 <= 0:
            raise ValueError("T, M, tau0, and tau1 must be positive")
        if t_decay <= 0 or geo_decay <= 0:
            raise ValueError("schedule decay constants must be positive")
        self.T = int(T)
        self.M = int(M)
        self.w_pos = float(w_pos)
        self.w_cov = float(w_cov)
        self.w_tan = float(w_tan)
        self.w_anchor = float(w_anchor)
        scale = float(ref["size"][1]) / 224.0
        self.tau0 = float(tau0) * scale
        self.tau1 = float(tau1) * scale
        self.t_hold = float(t_hold)
        self.t_decay = float(t_decay)
        self.geo_hold = float(geo_hold)
        self.geo_decay = float(geo_decay)
        self.cov_subsample = int(cov_subsample)
        self.chunk = int(chunk)
        self.size = tuple(ref["size"])
        self.edge_mask = np.asarray(ref["edge_mask"], dtype=bool)
        self.register_buffer("edge_pts", ref["edge_pts"].detach().clone())
        self.register_buffer("edge_w", ref["edge_w"].detach().clone())
        self.register_buffer("D_E", ref["D_E"].detach().clone())
        self.register_buffer("e_hat", ref["e_hat"].detach().clone())

    def tau(self, step: int) -> float:
        """Log-linearly anneal tau from tau0 to tau1."""
        progress = min(max(float(step) / self.T, 0.0), 1.0)
        return math.exp(
            math.log(self.tau0) * (1.0 - progress)
            + math.log(self.tau1) * progress
        )

    @staticmethod
    def _float(value: torch.Tensor) -> float:
        return float(value.detach().cpu())

    def forward(
        self,
        P: torch.Tensor,
        step: int = 0,
        P_init: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        V = bezier_polyline(P, self.M)
        delta = V[:, 1:] - V[:, :-1]
        lengths = torch.linalg.vector_norm(delta, dim=-1)
        midpoints = 0.5 * (V[:, 1:] + V[:, :-1])
        total_length = lengths.sum().clamp_min(1e-12)
        tau = self.tau(step)
        zero = P.sum() * 0.0

        pos = zero
        if self.w_pos != 0.0:
            distances = sample_field(self.D_E, midpoints, self.size)
            pos = (lengths * pseudo_huber(distances, tau)).sum() / total_length

        cov = zero
        if self.w_cov != 0.0:
            points = self.edge_pts
            weights = self.edge_w
            if self.cov_subsample > 0 and points.shape[0] > self.cov_subsample:
                indices = torch.randperm(points.shape[0], device=points.device)[
                    :self.cov_subsample
                ]
                points = points[indices]
                weights = weights[indices]
            distances = point_to_segments_dist(points, V, self.chunk)
            cov = (weights * pseudo_huber(distances, tau)).sum() / (
                weights.sum().clamp_min(1e-12)
            )

        tan = zero
        if self.w_tan != 0.0:
            tangents = sample_field(self.e_hat, midpoints, self.size)
            normals = torch.stack((-tangents[..., 1], tangents[..., 0]), dim=-1)
            normal_component = (delta * normals).sum(dim=-1)
            tan = normal_component.square().sum() / (
                delta.square().sum().clamp_min(1e-12)
            )

        anc = zero
        if self.w_anchor != 0.0 and P_init is not None:
            anc = (P - P_init).square().mean()

        progress = min(max(float(step) / self.T, 0.0), 1.0)
        w_geo = math.exp(-max(progress - self.geo_hold, 0.0) / self.geo_decay)
        w_anc = self.w_anchor * math.exp(
            -max(progress - self.t_hold, 0.0) / self.t_decay
        )
        total = w_geo * (
            self.w_pos * pos + self.w_cov * cov + self.w_tan * tan
        ) + w_anc * anc
        logs = {
            "tau": float(tau),
            "w_geo": float(w_geo),
            "w_anc": float(w_anc),
            "pos": self._float(pos),
            "cov": self._float(cov),
            "tan": self._float(tan),
            "anc": self._float(anc),
            "total": self._float(total),
        }
        return total, logs


@torch.no_grad()
def contour_prf(
    P: torch.Tensor,
    ref: Dict[str, object],
    sigma: float = 4.0,
    M: int = 64,
) -> Dict[str, float]:
    """Return debug-only soft contour precision, recall, and F-score."""
    if P.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}
    V = bezier_polyline(P, M)
    delta = V[:, 1:] - V[:, :-1]
    lengths = torch.linalg.vector_norm(delta, dim=-1)
    total_length = lengths.sum()
    if total_length <= 1e-12:
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}
    midpoints = 0.5 * (V[:, 1:] + V[:, :-1])
    distance_edge = sample_field(ref["D_E"], midpoints, tuple(ref["size"]))
    kernel_edge = torch.exp(-distance_edge.square() / (2.0 * sigma * sigma))
    precision = (lengths * kernel_edge).sum() / total_length

    distance_strokes = point_to_segments_dist(ref["edge_pts"], V)
    kernel_strokes = torch.exp(-distance_strokes.square() / (2.0 * sigma * sigma))
    recall = (ref["edge_w"] * kernel_strokes).sum() / ref["edge_w"].sum().clamp_min(1e-12)
    f_score = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "precision": float(precision.cpu()),
        "recall": float(recall.cpu()),
        "f": float(f_score.cpu()),
    }


def grad_norm_ratio(
    P: Union[torch.Tensor, Sequence[torch.Tensor]],
    loss_sds: torch.Tensor,
    loss_geo: torch.Tensor,
) -> Dict[str, float]:
    """Return norms, ratio, and cosine of SDS and geometry gradients."""
    parameters = (P,) if isinstance(P, torch.Tensor) else tuple(P)

    def gradient_vector(loss: torch.Tensor) -> torch.Tensor:
        if not parameters:
            return loss.detach().new_zeros((0,))
        if not loss.requires_grad:
            return torch.cat(
                [torch.zeros_like(param).reshape(-1) for param in parameters]
            )
        grads = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        flattened = [
            (
                grad.reshape(-1)
                if grad is not None
                else torch.zeros_like(param).reshape(-1)
            )
            for param, grad in zip(parameters, grads)
        ]
        return torch.cat(flattened)

    sds_gradient = gradient_vector(loss_sds)
    geo_gradient = gradient_vector(loss_geo)
    g_sds = torch.linalg.vector_norm(sds_gradient)
    g_geo = torch.linalg.vector_norm(geo_gradient)
    ratio = g_geo / (g_sds + 1e-12)
    cosine = torch.dot(sds_gradient, geo_gradient) / (
        g_sds * g_geo + 1e-12
    )
    cosine = cosine.clamp(-1.0, 1.0)
    return {
        "g_sds": float(g_sds.detach().cpu()),
        "g_geo": float(g_geo.detach().cpu()),
        "ratio": float(ratio.detach().cpu()),
        "cos": float(cosine.detach().cpu()),
    }
