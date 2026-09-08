import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


class ReferenceCache:
    def __init__(self, root, cache_hash):
        self.root = Path(root)
        self.cache_hash = cache_hash
        self.entry_dir = self.root / cache_hash
        self.parts_path = self.entry_dir / "parts.json"
        self.masks_path = self.entry_dir / "masks.npz"
        self.reference_path = self.entry_dir / "reference.npz"

    @property
    def semantic_ready(self):
        return self.parts_path.is_file() and self.masks_path.is_file()

    @property
    def reference_ready(self):
        return self.reference_path.is_file()

    def load_parts(self):
        with self.parts_path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def save_parts(self, payload):
        self.entry_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.parts_path, payload)

    def load_masks(self):
        with np.load(self.masks_path, allow_pickle=False) as archive:
            names = [str(name) for name in archive["names"].tolist()]
            return {
                name: archive[f"mask_{index:04d}"].astype(np.uint8)
                for index, name in enumerate(names)
            }

    def save_masks(self, masks):
        self.entry_dir.mkdir(parents=True, exist_ok=True)
        names = list(masks)
        arrays = {"names": np.asarray(names, dtype=np.str_)}
        arrays.update({
            f"mask_{index:04d}": _mask_numpy(masks[name])
            for index, name in enumerate(names)
        })
        _atomic_npz(self.masks_path, arrays)

    def load_reference(self, device):
        with np.load(self.reference_path, allow_pickle=False) as archive:
            size = tuple(int(value) for value in archive["size"].tolist())
            return {
                "edge_pts": torch.from_numpy(archive["edge_pts"].copy()).to(device),
                "edge_w": torch.from_numpy(archive["edge_w"].copy()).to(device),
                "D_E": torch.from_numpy(archive["D_E"].copy()).to(device),
                "e_hat": torch.from_numpy(archive["e_hat"].copy()).to(device),
                "edge_mask": archive["edge_mask"].astype(bool),
                "size": size,
            }

    def save_reference(self, reference):
        self.entry_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "edge_pts": _tensor_numpy(reference["edge_pts"]),
            "edge_w": _tensor_numpy(reference["edge_w"]),
            "D_E": _tensor_numpy(reference["D_E"]),
            "e_hat": _tensor_numpy(reference["e_hat"]),
            "edge_mask": np.asarray(reference["edge_mask"], dtype=bool),
            "size": np.asarray(reference["size"], dtype=np.int64),
        }
        _atomic_npz(self.reference_path, arrays)


def compute_cache_hash(
    image_path,
    prompt,
    model_id,
    max_parts,
    vlm_seed,
    vlm_temperature,
):
    digest = hashlib.sha256()
    with Path(image_path).open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    settings = {
        "prompt": prompt,
        "model_id": model_id,
        "max_parts": int(max_parts),
        "vlm_seed": int(vlm_seed),
        "vlm_temperature": float(vlm_temperature),
    }
    digest.update(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _mask_numpy(mask):
    array = _tensor_numpy(mask)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return (array > 0.5).astype(np.uint8)


def _tensor_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _atomic_json(path, payload):
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temp_file:
        json.dump(payload, temp_file, indent=2, ensure_ascii=False)
        temp_name = temp_file.name
    os.replace(temp_name, path)


def _atomic_npz(path, arrays):
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temp_file:
        np.savez_compressed(temp_file, **arrays)
        temp_name = temp_file.name
    os.replace(temp_name, path)
