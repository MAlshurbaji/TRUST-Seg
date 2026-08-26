from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def apply_grabcut(image_rgb: np.ndarray, box: list[int], iterations: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x, y, box_width, box_height = (int(value) for value in box)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    box_width = max(1, min(box_width, width - x))
    box_height = max(1, min(box_height, height - y))

    state = np.zeros((height, width), dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image_rgb,
        state,
        (x, y, box_width, box_height),
        background,
        foreground,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )
    return np.isin(state, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GrabCut pseudo-labels from MRI slices and bboxes.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--bbox-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.bbox_json).open("r", encoding="utf-8") as stream:
        records = json.load(stream)

    for filename, boxes in tqdm(records.items(), desc="Running GrabCut"):
        image = cv2.imread(str(images_dir / filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping missing image: {images_dir / filename}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        combined = np.zeros_like(image, dtype=np.uint8)
        for box in boxes:
            try:
                combined = np.maximum(combined, apply_grabcut(image_rgb, box, args.iterations))
            except cv2.error as error:
                print(f"GrabCut failed for {filename}, box {box}: {error}")
        cv2.imwrite(str(output_dir / filename), combined * 255)

    print(f"Saved GrabCut masks to {output_dir}")


if __name__ == "__main__":
    main()

