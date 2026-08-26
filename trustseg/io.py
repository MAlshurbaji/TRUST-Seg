from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


NIFTI_SUFFIXES = (".nii.gz", ".nii")
CASE_SUFFIXES = (
    "_image_mask_conf",
    "_image_mask_unc",
    "_image_mask",
    "_pred_mask",
    "_image",
    "_label",
    "_bbox",
    "_mask",
    "_pred",
)


def strip_nifti_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in NIFTI_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def case_id_from_name(name: str) -> str:
    stem = strip_nifti_suffix(Path(name).name)
    for suffix in CASE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def list_nifti(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name.lower().endswith(NIFTI_SUFFIXES)
    )


def read_volume(path: str | Path, dtype: np.dtype = np.float32) -> tuple[np.ndarray, sitk.Image]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(dtype, copy=False)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {array.shape} from {path}")
    return array, image


def write_volume(array: np.ndarray, path: str | Path, reference: sitk.Image) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(array))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(destination))


def require_same_shape(case_id: str, arrays: Iterable[tuple[str, np.ndarray]]) -> None:
    named_arrays = list(arrays)
    if not named_arrays:
        return
    reference_name, reference = named_arrays[0]
    for name, array in named_arrays[1:]:
        if array.shape != reference.shape:
            raise ValueError(
                f"Shape mismatch for {case_id}: {reference_name}={reference.shape}, "
                f"{name}={array.shape}"
            )


def normalize_minmax(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32, copy=False))
    minimum = float(volume.min())
    maximum = float(volume.max())
    if maximum <= minimum:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - minimum) / (maximum - minimum)).astype(np.float32)

