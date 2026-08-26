from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from skimage import measure
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D SAM2/MedSAM2 pseudo-labels using per-slice bbox prompts."
    )
    parser.add_argument("--external-repo", required=True, help="Local SAM2 or MedSAM2 checkout.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True, help="Model config expected by the external repo.")
    parser.add_argument("--images-dir", required=True, help="Input NIfTI image directory.")
    parser.add_argument("--bboxes-dir", required=True, help="Input NIfTI bbox directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--keep-all-components",
        action="store_true",
        help="Do not apply the largest-connected-component postprocessing used in the paper scripts.",
    )
    return parser.parse_args()


def normalize_uint8(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    lower, upper = np.percentile(volume, (0.5, 99.5))
    if upper <= lower:
        return np.zeros_like(volume, dtype=np.uint8)
    normalized = np.clip(volume, lower, upper)
    normalized = (normalized - lower) / (upper - lower)
    return (normalized * 255.0).astype(np.uint8)


def resize_rgb(volume: np.ndarray, image_size: int) -> np.ndarray:
    result = np.empty((volume.shape[0], 3, image_size, image_size), dtype=np.float32)
    for index, image_slice in enumerate(volume):
        image = Image.fromarray(image_slice).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        result[index] = np.asarray(image).transpose(2, 0, 1)
    return result


def slice_boxes(bbox_volume: np.ndarray) -> list[tuple[int, np.ndarray]]:
    prompts: list[tuple[int, np.ndarray]] = []
    for slice_index, bbox_slice in enumerate(bbox_volume > 0):
        if not bbox_slice.any():
            continue
        rows, columns = np.where(bbox_slice)
        prompts.append(
            (
                slice_index,
                np.asarray(
                    [columns.min(), rows.min(), columns.max(), rows.max()],
                    dtype=np.int32,
                ),
            )
        )
    return prompts


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def case_id(filename: str) -> str:
    for suffix in ("_image.nii.gz", ".nii.gz", ".nii"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def main() -> None:
    args = parse_args()
    external_repo = Path(args.external_repo).expanduser().resolve()
    sys.path.insert(0, str(external_repo))
    try:
        from sam2.build_sam import build_sam2_video_predictor_npz
    except ImportError as error:
        raise RuntimeError(
            "The selected checkout does not expose build_sam2_video_predictor_npz. "
            "Use the SAM2/MedSAM2 version described in its README, then run this adapter again."
        ) from error

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor_npz(args.model_config, args.checkpoint)
    image_size = int(getattr(predictor, "image_size", getattr(predictor, "sam_image_size", 512)))
    images_dir = Path(args.images_dir)
    bboxes_dir = Path(args.bboxes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.nii*"))
    for image_path in tqdm(image_paths, desc="Generating SAM2/MedSAM2 masks"):
        identifier = case_id(image_path.name)
        image_reference = sitk.ReadImage(str(image_path))
        volume = sitk.GetArrayFromImage(image_reference)
        bbox = sitk.GetArrayFromImage(sitk.ReadImage(str(bboxes_dir / f"{identifier}_bbox.nii.gz")))
        if volume.shape != bbox.shape:
            raise ValueError(f"Image/bbox shape mismatch for {identifier}")

        prompts = slice_boxes(bbox)
        prepared = resize_rgb(normalize_uint8(volume), image_size) / 255.0
        prepared_tensor = torch.from_numpy(prepared).to(device)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), device=device)[:, None, None]
        prepared_tensor = (prepared_tensor - mean) / std
        segmentation = np.zeros_like(volume, dtype=np.uint8)

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            state = predictor.init_state(prepared_tensor, volume.shape[1], volume.shape[2])
            for frame_index, box in prompts:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_index,
                    obj_id=1,
                    box=box,
                )
            for frame_index, _, logits in predictor.propagate_in_video(state):
                segmentation[frame_index, (logits[0] > 0).cpu().numpy()[0]] = 1
            predictor.reset_state(state)

            for frame_index, box in prompts:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_index,
                    obj_id=1,
                    box=box,
                )
            for frame_index, _, logits in predictor.propagate_in_video(state, reverse=True):
                segmentation[frame_index, (logits[0] > 0).cpu().numpy()[0]] = 1
            predictor.reset_state(state)

        if segmentation.any() and not args.keep_all_components:
            segmentation = largest_component(segmentation).astype(np.uint8)
        output = sitk.GetImageFromArray(segmentation)
        output.CopyInformation(image_reference)
        sitk.WriteImage(output, str(output_dir / f"{identifier}_image_mask.nii.gz"))

    print(f"Saved pseudo-label volumes to {output_dir}")


if __name__ == "__main__":
    main()
