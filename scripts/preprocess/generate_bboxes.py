from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


Box = tuple[int, int, int, int]


def boxes_overlap(first: Box, second: Box) -> bool:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2


def merge_overlapping_boxes(boxes: list[Box]) -> list[Box]:
    merged = list(boxes)
    while True:
        used: set[int] = set()
        next_boxes: list[Box] = []
        for first_index, second_index in combinations(range(len(merged)), 2):
            if first_index in used or second_index in used:
                continue
            first, second = merged[first_index], merged[second_index]
            if not boxes_overlap(first, second):
                continue
            x1, y1, w1, h1 = first
            x2, y2, w2, h2 = second
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
            next_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
            used.update((first_index, second_index))

        next_boxes.extend(box for index, box in enumerate(merged) if index not in used)
        if len(next_boxes) == len(merged):
            return sorted(next_boxes)
        merged = next_boxes


def boxes_from_mask(mask: np.ndarray, expansion: int) -> list[Box]:
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape
    boxes: list[Box] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        x0, y0 = max(0, x - expansion), max(0, y - expansion)
        x1 = min(width, x + box_width + expansion)
        y1 = min(height, y + box_height + expansion)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return merge_overlapping_boxes(boxes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tight slice-wise lesion bounding boxes.")
    parser.add_argument("--labels-dir", required=True, help="Directory containing binary label PNGs.")
    parser.add_argument("--output-json", required=True, help="Output JSON file.")
    parser.add_argument("--bbox-masks-dir", help="Optional output directory for filled bbox masks.")
    parser.add_argument("--expansion", type=int, default=0, help="Pixels added on each box side.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    output_json = Path(args.output_json)
    bbox_masks_dir = Path(args.bbox_masks_dir) if args.bbox_masks_dir else None
    if args.expansion < 0:
        raise ValueError("--expansion must be non-negative")
    if bbox_masks_dir is not None:
        bbox_masks_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, list[list[int]]] = {}
    image_paths = sorted(path for path in labels_dir.iterdir() if path.suffix.lower() == ".png")
    for label_path in tqdm(image_paths, desc="Generating bboxes"):
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read {label_path}")
        boxes = boxes_from_mask(mask, args.expansion)
        records[label_path.name] = [list(box) for box in boxes]

        if bbox_masks_dir is not None:
            bbox_mask = np.zeros_like(mask, dtype=np.uint8)
            for x, y, width, height in boxes:
                bbox_mask[y : y + height, x : x + width] = 255
            cv2.imwrite(str(bbox_masks_dir / label_path.name), bbox_mask)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2)
    print(f"Saved {len(records)} slice records to {output_json}")


if __name__ == "__main__":
    main()

