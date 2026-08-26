from __future__ import annotations

import argparse

from trustseg.config import load_config, nested, resolve_path
from trustseg.data import VolumeDataset
from trustseg.training import build_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TRUST-Seg 3D U-Net student.")
    parser.add_argument("--config", required=True, help="Dataset YAML configuration.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("initial", "final", "supervised"),
        help="Training target: initial consensus, SDA-refined labels, or true labels.",
    )
    parser.add_argument("--output", help="Override the run output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_root = resolve_path(config, nested(config, "dataset", "train_volumes"))
    validation_root = resolve_path(config, nested(config, "dataset", "validation_volumes"))

    if args.stage == "initial":
        target_dir = resolve_path(config, nested(config, "paths", "initial_pseudo_labels"))
        target_suffix = "_image_mask.nii.gz"
        confidence_dir = target_dir
        with_bbox = bool(nested(config, "model", "with_bbox"))
    elif args.stage == "final":
        target_dir = resolve_path(config, nested(config, "paths", "refined_pseudo_labels"))
        target_suffix = "_image_mask.nii.gz"
        confidence_dir = target_dir
        with_bbox = bool(nested(config, "model", "with_bbox"))
    else:
        target_dir = train_root / "labels"
        target_suffix = "_label.nii.gz"
        confidence_dir = None
        with_bbox = bool(config["model"].get("supervised_with_bbox", True))

    train_dataset = VolumeDataset(
        volume_root=train_root,
        target_dir=target_dir,
        target_suffix=target_suffix,
        confidence_dir=confidence_dir,
        with_bbox=with_bbox,
    )
    validation_dataset = VolumeDataset(
        volume_root=validation_root,
        target_dir=validation_root / "labels",
        target_suffix="_label.nii.gz",
        with_bbox=with_bbox,
    )

    training_config = dict(nested(config, "training"))
    training_config["epochs"] = int(training_config[f"{args.stage}_epochs"])
    output_dir = resolve_path(
        config,
        args.output or (resolve_path(config, nested(config, "paths", "runs")) / args.stage),
    )
    model = build_model(nested(config, "model"), with_bbox=with_bbox)
    best_path = train_model(
        model,
        train_dataset,
        validation_dataset,
        output_dir,
        training_config,
        stage=args.stage,
    )
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
