from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def parse_slice_name(path: Path) -> tuple[str, int]:
    try:
        case_id, slice_text = path.stem.rsplit("_", 1)
        return case_id, int(slice_text)
    except ValueError as error:
        raise ValueError(
            f"Expected '<case>_<slice>.png', got {path.name}. Example: 001_004.png"
        ) from error


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def write_nifti(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(array), str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D image, label, and slice-wise bbox NIfTI volumes."
    )
    parser.add_argument("--images-dir", required=True, help="Input image PNG slices.")
    parser.add_argument("--labels-dir", required=True, help="Input label PNG slices.")
    parser.add_argument("--bbox-json", required=True, help="Slice-wise bbox JSON.")
    parser.add_argument("--output-dir", required=True, help="Output split directory.")
    parser.add_argument(
        "--strict-labels",
        action="store_true",
        help="Fail if a label slice is absent. By default, absent labels are treated as empty.",
    )
    parser.add_argument("--npz-dir", help="Optional directory for image/label/bbox NPZ files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    npz_dir = Path(args.npz_dir) if args.npz_dir else None
    with Path(args.bbox_json).open("r", encoding="utf-8") as stream:
        bbox_records = json.load(stream)

    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for image_path in sorted(images_dir.glob("*.png")):
        case_id, slice_index = parse_slice_name(image_path)
        grouped[case_id].append((slice_index, image_path))
    if not grouped:
        raise RuntimeError(f"No PNG slices found in {images_dir}")

    for case_id, slices in tqdm(sorted(grouped.items()), desc="Building volumes"):
        image_slices: list[np.ndarray] = []
        label_slices: list[np.ndarray] = []
        bbox_slices: list[np.ndarray] = []

        for _, image_path in sorted(slices):
            image = read_grayscale(image_path).astype(np.float32)
            label_path = labels_dir / image_path.name
            if label_path.exists():
                label = (read_grayscale(label_path) > 0).astype(np.uint8)
            elif args.strict_labels:
                raise FileNotFoundError(label_path)
            else:
                label = np.zeros_like(image, dtype=np.uint8)

            bbox_mask = np.zeros_like(label, dtype=np.uint8)
            boxes = bbox_records.get(image_path.name, bbox_records.get(image_path.stem, []))
            for x, y, width, height in boxes:
                x0, y0 = max(0, int(x)), max(0, int(y))
                x1 = min(image.shape[1], x0 + int(width))
                y1 = min(image.shape[0], y0 + int(height))
                bbox_mask[y0:y1, x0:x1] = 1

            image_slices.append(image)
            label_slices.append(label)
            bbox_slices.append(bbox_mask)

        image_volume = np.stack(image_slices).astype(np.float32)
        label_volume = np.stack(label_slices).astype(np.uint8)
        bbox_volume = np.stack(bbox_slices).astype(np.uint8)

        write_nifti(image_volume, output_dir / "images" / f"{case_id}_image.nii.gz")
        write_nifti(label_volume, output_dir / "labels" / f"{case_id}_label.nii.gz")
        write_nifti(bbox_volume, output_dir / "bboxes" / f"{case_id}_bbox.nii.gz")

        if npz_dir is not None:
            npz_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_dir / f"{case_id}.npz",
                image=image_volume,
                label_true=label_volume,
                bbox_2d=bbox_volume,
            )

    print(f"Saved {len(grouped)} reconstructed cases to {output_dir}")


if __name__ == "__main__":
    main()

