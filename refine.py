from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trustseg.config import load_config, nested, resolve_path
from trustseg.data import VolumeDataset, collate_volumes, pad_to_factor, unpad
from trustseg.io import read_volume, write_volume
from trustseg.refinement import smooth_delta_attention
from trustseg.training import build_model, choose_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply TRUST-Seg Smooth Delta Attention refinement.")
    parser.add_argument("--config", required=True, help="Dataset YAML configuration.")
    parser.add_argument("--checkpoint", required=True, help="Initial-student checkpoint.")
    parser.add_argument("--output", help="Override the refined pseudo-label directory.")
    parser.add_argument("--save-attention", action="store_true", help="Also save SDA attention maps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_root = resolve_path(config, nested(config, "dataset", "train_volumes"))
    teacher_dir = resolve_path(config, nested(config, "paths", "initial_pseudo_labels"))
    output_dir = resolve_path(
        config,
        args.output or nested(config, "paths", "refined_pseudo_labels"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VolumeDataset(
        volume_root=train_root,
        target_dir=teacher_dir,
        target_suffix="_image_mask.nii.gz",
        with_bbox=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(nested(config, "training").get("workers", 4)),
        collate_fn=collate_volumes,
    )
    device = choose_device(str(nested(config, "training").get("device", "cuda")))
    model = build_model(nested(config, "model"), with_bbox=True).to(device)
    load_checkpoint(model, resolve_path(config, args.checkpoint), device)
    model.eval()

    refinement = nested(config, "refinement")
    factor = int(nested(config, "training").get("padding_factor", 16))
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Refining pseudo-labels"):
            case_id = batch["case_id"][0]
            depth, height, width = batch["shape"][0]
            inputs = batch["inputs"].to(device)
            padded_inputs, padding = pad_to_factor(inputs, factor=factor)
            student = torch.sigmoid(unpad(model(padded_inputs), padding))[0, 0]
            student_np = student[:depth, :height, :width].cpu().numpy()

            teacher = batch["target"][0, 0, :depth, :height, :width].numpy()
            bbox = batch["inputs"][0, 1, :depth, :height, :width].numpy()
            uncertainty_path = teacher_dir / f"{case_id}_image_mask_unc.nii.gz"
            uncertainty, reference = read_volume(uncertainty_path)

            refined, refined_uncertainty, confidence, attention = smooth_delta_attention(
                student_probability=student_np,
                teacher_consensus=teacher,
                teacher_uncertainty=uncertainty,
                bbox=bbox,
                uncertainty_weight=float(refinement["uncertainty_weight"]),
                student_threshold=float(refinement["student_threshold"]),
                confidence_threshold=float(refinement["confidence_threshold"]),
                attention_steepness=float(refinement["attention_steepness"]),
                confidence_alpha=float(refinement["confidence_alpha"]),
            )

            prefix = output_dir / f"{case_id}_image_mask"
            write_volume(refined, f"{prefix}.nii.gz", reference)
            write_volume(refined_uncertainty, f"{prefix}_unc.nii.gz", reference)
            write_volume(confidence, f"{prefix}_conf.nii.gz", reference)
            if args.save_attention:
                write_volume(attention, output_dir / f"{case_id}_attention.nii.gz", reference)

    print(f"Saved SDA-refined pseudo-labels to {output_dir}")


if __name__ == "__main__":
    main()

