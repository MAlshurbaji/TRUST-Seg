from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trustseg.io import case_id_from_name, list_nifti, normalize_minmax, read_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert paired 3D NIfTI volumes to 2D PNG slices.")
    parser.add_argument("--images-dir", required=True, help="Directory containing MRI volumes.")
    parser.add_argument("--labels-dir", required=True, help="Directory containing label volumes.")
    parser.add_argument("--output-images", required=True, help="Destination for image PNG files.")
    parser.add_argument("--output-labels", required=True, help="Destination for binary label PNG files.")
    parser.add_argument("--size", type=int, default=128, help="Output height and width.")
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=0, help="Array slicing axis.")
    parser.add_argument("--index-base", type=int, default=0, choices=(0, 1))
    return parser.parse_args()


def find_label(labels_dir: Path, case_id: str) -> Path:
    candidates = (
        labels_dir / f"{case_id}_label.nii.gz",
        labels_dir / f"{case_id}_mask.nii.gz",
        labels_dir / f"{case_id}.nii.gz",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No label volume found for case {case_id} in {labels_dir}")


def take_slice(volume: np.ndarray, index: int, axis: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    output_images = Path(args.output_images)
    output_labels = Path(args.output_labels)
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(list_nifti(args.images_dir), desc="Slicing volumes"):
        case_id = case_id_from_name(image_path.name)
        image, _ = read_volume(image_path)
        label, _ = read_volume(find_label(labels_dir, case_id))
        if image.shape != label.shape:
            raise ValueError(f"Shape mismatch for {case_id}: image={image.shape}, label={label.shape}")

        image = normalize_minmax(image)
        for index in range(image.shape[args.axis]):
            image_slice = (take_slice(image, index, args.axis) * 255.0).astype(np.uint8)
            label_slice = (take_slice(label, index, args.axis) > 0).astype(np.uint8) * 255
            slice_number = index + args.index_base
            filename = f"{case_id}_{slice_number:03d}.png"

            Image.fromarray(image_slice).resize(
                (args.size, args.size), Image.Resampling.BILINEAR
            ).save(output_images / filename)
            Image.fromarray(label_slice).resize(
                (args.size, args.size), Image.Resampling.NEAREST
            ).save(output_labels / filename)

    print(f"Saved image slices to {output_images}")
    print(f"Saved label slices to {output_labels}")


if __name__ == "__main__":
    main()
