from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D SAM or MedSAM pseudo-labels from slice-wise bbox prompts."
    )
    parser.add_argument("--external-repo", required=True, help="Local SAM or MedSAM checkout.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="vit_l", choices=("vit_b", "vit_l", "vit_h"))
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--bbox-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_repo = Path(args.external_repo).expanduser().resolve()
    sys.path.insert(0, str(external_repo))
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as error:
        raise RuntimeError(
            "Could not import segment_anything. Install the selected SAM/MedSAM repository "
            "or pass its local checkout with --external-repo."
        ) from error

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    predictor = SamPredictor(model)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.bbox_json).open("r", encoding="utf-8") as stream:
        records = json.load(stream)

    for filename, boxes in tqdm(records.items(), desc="Generating SAM masks"):
        image = cv2.imread(str(images_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping missing image: {images_dir / filename}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        combined = np.zeros_like(image, dtype=np.uint8)
        if boxes:
            predictor.set_image(image_rgb)
            for x, y, width, height in boxes:
                box_xyxy = np.asarray([x, y, x + width, y + height], dtype=np.float32)
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_xyxy,
                    multimask_output=True,
                )
                combined = np.maximum(combined, masks[int(np.argmax(scores))].astype(np.uint8))
        cv2.imwrite(str(output_dir / filename), combined * 255)

    print(f"Saved pseudo-label slices to {output_dir}")


if __name__ == "__main__":
    main()

