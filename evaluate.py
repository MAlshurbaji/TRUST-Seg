from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from trustseg.config import load_config, nested, resolve_path
from trustseg.data import VolumeDataset, collate_volumes
from trustseg.training import (
    build_model,
    choose_device,
    evaluate_model,
    load_checkpoint,
    write_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TRUST-Seg checkpoint in 3D.")
    parser.add_argument("--config", required=True, help="Dataset YAML configuration.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint.")
    parser.add_argument("--output", required=True, help="Directory for metrics and predictions.")
    parser.add_argument("--without-bbox", action="store_true", help="Evaluate a one-channel model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    test_root = resolve_path(config, nested(config, "dataset", "test_volumes"))
    output_dir = resolve_path(config, args.output)
    with_bbox = not args.without_bbox

    dataset = VolumeDataset(
        volume_root=test_root,
        target_dir=test_root / "labels",
        target_suffix="_label.nii.gz",
        with_bbox=with_bbox,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(nested(config, "training").get("workers", 4)),
        collate_fn=collate_volumes,
    )
    device = choose_device(str(nested(config, "training").get("device", "cuda")))
    model = build_model(nested(config, "model"), with_bbox=with_bbox).to(device)
    load_checkpoint(model, resolve_path(config, args.checkpoint), device)

    aggregate, cases = evaluate_model(
        model,
        loader,
        device,
        factor=int(nested(config, "training").get("padding_factor", 16)),
        threshold=float(nested(config, "training").get("threshold", 0.5)),
        prediction_dir=output_dir / "predictions",
    )
    write_evaluation(aggregate, cases, output_dir)
    print(
        f"DSC={aggregate.dice * 100:.2f}% | IoU={aggregate.iou * 100:.2f}% | "
        f"HD95={aggregate.hd95:.2f} | ASD={aggregate.asd:.2f}"
    )
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()

