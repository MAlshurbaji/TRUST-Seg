from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def parse_slice_name(path: Path) -> tuple[str, int]:
    case_id, slice_text = path.stem.rsplit("_", 1)
    return case_id, int(slice_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stack 2D pseudo-label PNGs into 3D NIfTI masks.")
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reference-images",
        help="Optional NIfTI image directory used to copy spacing, origin, and direction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_images) if args.reference_images else None
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for mask_path in sorted(masks_dir.glob("*.png")):
        case_id, slice_index = parse_slice_name(mask_path)
        grouped[case_id].append((slice_index, mask_path))

    for case_id, slices in tqdm(sorted(grouped.items()), desc="Stacking masks"):
        arrays: list[np.ndarray] = []
        expected_shape: tuple[int, int] | None = None
        for _, mask_path in sorted(slices):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Unable to read {mask_path}")
            if expected_shape is None:
                expected_shape = mask.shape
            elif mask.shape != expected_shape:
                raise ValueError(f"Inconsistent slice shape in case {case_id}: {mask_path}")
            arrays.append((mask > 0).astype(np.uint8))

        volume = np.stack(arrays)
        image = sitk.GetImageFromArray(volume)
        if reference_dir is not None:
            reference_path = reference_dir / f"{case_id}_image.nii.gz"
            reference = sitk.ReadImage(str(reference_path))
            if sitk.GetArrayFromImage(reference).shape != volume.shape:
                raise ValueError(f"Reference shape mismatch for case {case_id}")
            image.CopyInformation(reference)

        output_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(output_dir / f"{case_id}_image_mask.nii.gz"))

    print(f"Saved {len(grouped)} pseudo-label volumes to {output_dir}")


if __name__ == "__main__":
    main()

