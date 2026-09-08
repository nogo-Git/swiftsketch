#!/usr/bin/env python3
"""Reusable DreamSIM and MS-SSIM metrics for image-to-sketch evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageOps

from compute_clip_score import load_image


RESAMPLING = Image.Resampling if hasattr(Image, "Resampling") else Image


@dataclass(frozen=True)
class XDoGConfig:
    """Parameters for the extended Difference-of-Gaussians edge map."""

    sigma: float = 0.5
    k: float = 10.0
    gamma: float = 0.98
    epsilon: float = -0.1
    phi: float = 200.0

    def validate(self) -> None:
        if self.sigma <= 0:
            raise ValueError("XDoG sigma must be > 0.")
        if self.k <= 1:
            raise ValueError("XDoG k must be > 1.")
        if self.gamma <= 0:
            raise ValueError("XDoG gamma must be > 0.")
        if self.phi <= 0:
            raise ValueError("XDoG phi must be > 0.")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def resolve_device(device: str, torch: Any) -> str:
    """Resolve an auto/cpu/cuda device argument and validate CUDA requests."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def resize_image(image: Image.Image, size: int, mode: str = "stretch") -> Image.Image:
    """Resize an image to a square using stretch or aspect-preserving letterbox."""
    if size < 1:
        raise ValueError("Image size must be >= 1.")
    if mode == "stretch":
        return image.resize((size, size), resample=RESAMPLING.LANCZOS)
    if mode == "letterbox":
        return ImageOps.pad(
            image,
            (size, size),
            method=RESAMPLING.LANCZOS,
            color="white",
            centering=(0.5, 0.5),
        )
    raise ValueError(f"Unsupported resize mode: {mode}")


def make_xdog_edge_map(image: Image.Image, config: XDoGConfig) -> Image.Image:
    """Convert a PIL image to a white-background XDoG edge map."""
    config.validate()
    try:
        import numpy as np
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise ImportError(
            "XDoG preprocessing requires numpy and scipy. Install the project "
            "requirements before running MS-SSIM evaluation."
        ) from exc

    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    narrow = gaussian_filter(gray, sigma=config.sigma)
    wide = gaussian_filter(gray, sigma=config.sigma * config.k)
    difference = narrow - config.gamma * wide

    edge_map = np.ones_like(difference, dtype=np.float32)
    below_threshold = difference < config.epsilon
    edge_map[below_threshold] = 1.0 + np.tanh(
        config.phi * (difference[below_threshold] - config.epsilon)
    )
    edge_map = np.clip(edge_map, 0.0, 1.0)
    return Image.fromarray((edge_map * 255.0).round().astype(np.uint8), mode="L")


def _pil_gray_to_tensor(image: Image.Image, torch: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for image metric preprocessing.") from exc

    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def prepare_ms_ssim_pair(
    input_image: Image.Image,
    sketch_image: Image.Image,
    *,
    image_size: int,
    resize_mode: str,
    xdog_config: XDoGConfig,
    show_progress: bool = True,
    torch: Any,
) -> tuple[Any, Any]:
    """Prepare an XDoG reference and grayscale sketch as 1xHxW tensors."""
    input_resized = resize_image(input_image.convert("RGB"), image_size, resize_mode)
    sketch_resized = resize_image(sketch_image.convert("RGB"), image_size, resize_mode)
    input_edge_map = make_xdog_edge_map(input_resized, xdog_config)
    return (
        _pil_gray_to_tensor(input_edge_map, torch),
        _pil_gray_to_tensor(sketch_resized, torch),
    )


def _chunks(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size < 1:
        raise ValueError("Batch size must be >= 1.")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class DreamSimEvaluator:
    """Load DreamSIM once and compute cached image embeddings in batches."""

    def __init__(self, device: str, model_type: str = "ensemble") -> None:
        try:
            import torch
            from dreamsim import dreamsim
        except ImportError as exc:
            raise ImportError(
                "DreamSIM evaluation requires dreamsim and torch. Install the "
                "project requirements before running this metric."
            ) from exc

        self.torch = torch
        self.device = resolve_device(device, torch)
        self.model_type = model_type
        kwargs: dict[str, Any] = {
            "pretrained": True,
            "device": self.device,
        }
        if model_type != "ensemble":
            kwargs["dreamsim_type"] = model_type
        self.model, self.preprocess = dreamsim(**kwargs)
        self.model.eval()

    def _preprocess_image(self, path: Path) -> Any:
        tensor = self.preprocess(load_image(path))
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(
                f"Unexpected DreamSIM preprocess output for {path}: {tuple(tensor.shape)}"
            )
        return tensor.squeeze(0)

    def embed_paths(
        self,
        paths: Sequence[Path],
        batch_size: int,
        show_progress: bool = True,
    ) -> dict[Path, Any]:
        """Embed unique paths, returning normalized CPU vectors keyed by path."""
        unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
        embeddings: dict[Path, Any] = {}
        batches: Iterable[Sequence[Path]] = _chunks(unique_paths, batch_size)
        if show_progress:
            from tqdm.auto import tqdm

            batches = tqdm(
                batches,
                total=(len(unique_paths) + batch_size - 1) // batch_size,
                desc="DreamSIM embeddings",
                unit="batch",
            )
        with self.torch.inference_mode():
            for batch_paths in batches:
                batch = self.torch.stack(
                    [self._preprocess_image(path) for path in batch_paths]
                ).to(self.device)
                batch_embeddings = self.model.embed(batch).float()
                batch_embeddings = self.torch.nn.functional.normalize(
                    batch_embeddings, dim=-1
                ).cpu()
                for path, embedding in zip(batch_paths, batch_embeddings):
                    embeddings[path] = embedding
        return embeddings

    def distances(
        self,
        pairs: Sequence[tuple[Path, Path]],
        batch_size: int,
        show_progress: bool = True,
    ) -> list[float]:
        """Return cosine distances for input/sketch path pairs."""
        all_paths = [path for pair in pairs for path in pair]
        embeddings = self.embed_paths(all_paths, batch_size, show_progress)
        distances: list[float] = []
        for input_path, sketch_path in pairs:
            input_embedding = embeddings[input_path.resolve()]
            sketch_embedding = embeddings[sketch_path.resolve()]
            cosine_similarity = self.torch.dot(input_embedding, sketch_embedding).item()
            distances.append(1.0 - cosine_similarity)
        return distances


def compute_ms_ssim_scores(
    pairs: Sequence[tuple[Path, Path]],
    *,
    device: str,
    batch_size: int,
    image_size: int,
    resize_mode: str,
    xdog_config: XDoGConfig,
    show_progress: bool = True,
) -> tuple[list[float], str]:
    """Compute one MS-SSIM score per input/sketch path pair."""
    try:
        import torch
        from pytorch_msssim import ms_ssim
    except ImportError as exc:
        raise ImportError(
            "MS-SSIM evaluation requires pytorch-msssim and torch. Install the "
            "project requirements before running this metric."
        ) from exc

    # Five default MS-SSIM scales with an 11-pixel window require >160 pixels.
    if image_size <= 160:
        raise ValueError("--image-size must be > 160 for five-scale MS-SSIM.")

    resolved_device = resolve_device(device, torch)
    scores: list[float] = []
    batches: Iterable[Sequence[tuple[Path, Path]]] = _chunks(pairs, batch_size)
    if show_progress:
        from tqdm.auto import tqdm

        batches = tqdm(
            batches,
            total=(len(pairs) + batch_size - 1) // batch_size,
            desc="MS-SSIM pairs",
            unit="batch",
        )
    with torch.inference_mode():
        for batch_pairs in batches:
            references = []
            sketches = []
            for input_path, sketch_path in batch_pairs:
                reference, sketch = prepare_ms_ssim_pair(
                    load_image(input_path),
                    load_image(sketch_path),
                    image_size=image_size,
                    resize_mode=resize_mode,
                    xdog_config=xdog_config,
                    torch=torch,
                )
                references.append(reference)
                sketches.append(sketch)

            reference_batch = torch.stack(references).to(resolved_device)
            sketch_batch = torch.stack(sketches).to(resolved_device)
            batch_scores = ms_ssim(
                reference_batch,
                sketch_batch,
                data_range=1.0,
                size_average=False,
            )
            scores.extend(float(score) for score in batch_scores.detach().cpu().tolist())

    return scores, resolved_device
