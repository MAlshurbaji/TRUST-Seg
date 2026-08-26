from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from trustseg.io import case_id_from_name, list_nifti, normalize_minmax, read_volume, require_same_shape


class VolumeDataset(Dataset[dict[str, Any]]):
    """Load MRI, slice-wise bbox priors, and either pseudo or true 3D targets."""

    def __init__(
        self,
        volume_root: str | Path,
        target_dir: str | Path,
        target_suffix: str,
        confidence_dir: str | Path | None = None,
        confidence_suffix: str = "_image_mask_conf.nii.gz",
        with_bbox: bool = True,
    ) -> None:
        self.volume_root = Path(volume_root)
        self.image_dir = self.volume_root / "images"
        self.bbox_dir = self.volume_root / "bboxes"
        self.target_dir = Path(target_dir)
        self.target_suffix = target_suffix
        self.confidence_dir = Path(confidence_dir) if confidence_dir is not None else None
        self.confidence_suffix = confidence_suffix
        self.with_bbox = with_bbox

        image_paths = list_nifti(self.image_dir)
        self.cases = [(case_id_from_name(path.name), path) for path in image_paths]
        if not self.cases:
            raise RuntimeError(f"No NIfTI image volumes found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_id, image_path = self.cases[index]
        image, reference = read_volume(image_path)

        if self.with_bbox:
            bbox_path = self.bbox_dir / f"{case_id}_bbox.nii.gz"
            bbox, _ = read_volume(bbox_path)
            bbox = (bbox > 0).astype(np.float32)
        else:
            bbox = np.zeros_like(image, dtype=np.float32)

        target_path = self.target_dir / f"{case_id}{self.target_suffix}"
        target, _ = read_volume(target_path)
        target = np.clip(target, 0.0, 1.0).astype(np.float32)

        if self.confidence_dir is None:
            confidence = np.ones_like(target, dtype=np.float32)
        else:
            confidence_path = self.confidence_dir / f"{case_id}{self.confidence_suffix}"
            confidence, _ = read_volume(confidence_path)
            confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)

        require_same_shape(
            case_id,
            (("image", image), ("bbox", bbox), ("target", target), ("confidence", confidence)),
        )

        channels = [normalize_minmax(image)]
        if self.with_bbox:
            channels.append(bbox)

        spacing_xyz = reference.GetSpacing()
        return {
            "case_id": case_id,
            "inputs": torch.from_numpy(np.stack(channels, axis=0)),
            "target": torch.from_numpy(target[None, ...]),
            "weight": torch.from_numpy(confidence[None, ...]),
            "shape": tuple(int(value) for value in image.shape),
            "spacing": tuple(float(value) for value in reversed(spacing_xyz)),
            "reference_path": str(image_path),
        }


def collate_volumes(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    max_depth = max(sample["inputs"].shape[-3] for sample in samples)
    max_height = max(sample["inputs"].shape[-2] for sample in samples)
    max_width = max(sample["inputs"].shape[-1] for sample in samples)

    def pad(tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        depth, height, width = tensor.shape[-3:]
        return F.pad(
            tensor,
            (0, max_width - width, 0, max_height - height, 0, max_depth - depth),
            value=value,
        )

    return {
        "case_id": [sample["case_id"] for sample in samples],
        "inputs": torch.stack([pad(sample["inputs"]) for sample in samples]),
        "target": torch.stack([pad(sample["target"]) for sample in samples]),
        "weight": torch.stack([pad(sample["weight"]) for sample in samples]),
        "shape": [sample["shape"] for sample in samples],
        "spacing": [sample["spacing"] for sample in samples],
        "reference_path": [sample["reference_path"] for sample in samples],
    }


def pad_to_factor(tensor: torch.Tensor, factor: int = 16) -> tuple[torch.Tensor, tuple[int, int, int]]:
    depth, height, width = tensor.shape[-3:]
    padding = (
        (factor - depth % factor) % factor,
        (factor - height % factor) % factor,
        (factor - width % factor) % factor,
    )
    pad_depth, pad_height, pad_width = padding
    padded = F.pad(tensor, (0, pad_width, 0, pad_height, 0, pad_depth))
    return padded, padding


def unpad(tensor: torch.Tensor, padding: tuple[int, int, int]) -> torch.Tensor:
    pad_depth, pad_height, pad_width = padding
    depth_end = tensor.shape[-3] - pad_depth if pad_depth else tensor.shape[-3]
    height_end = tensor.shape[-2] - pad_height if pad_height else tensor.shape[-2]
    width_end = tensor.shape[-1] - pad_width if pad_width else tensor.shape[-1]
    return tensor[..., :depth_end, :height_end, :width_end]


def steps_per_epoch(dataset_size: int, batch_size: int) -> int:
    return max(1, math.ceil(dataset_size / batch_size))

