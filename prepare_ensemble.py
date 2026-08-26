from __future__ import annotations

import argparse

from trustseg.config import load_config, nested, resolve_path
from trustseg.ensemble import build_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reliability-weighted TRUST-Seg teacher consensus volumes."
    )
    parser.add_argument("--config", required=True, help="Dataset YAML configuration.")
    parser.add_argument("--output", help="Override the ensemble output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    teachers = nested(config, "teachers")
    teacher_dirs = {
        name: resolve_path(config, values["path"]) for name, values in teachers.items()
    }
    reliabilities = {name: float(values["reliability"]) for name, values in teachers.items()}
    train_root = resolve_path(config, nested(config, "dataset", "train_volumes"))
    output_dir = resolve_path(
        config,
        args.output or nested(config, "paths", "initial_pseudo_labels"),
    )
    alpha = float(nested(config, "ensemble", "confidence_alpha"))

    weights = build_ensemble(
        teacher_dirs=teacher_dirs,
        reliability_scores=reliabilities,
        bbox_dir=train_root / "bboxes",
        output_dir=output_dir,
        confidence_alpha=alpha,
    )
    print(f"Saved ensemble pseudo-labels to {output_dir}")
    print("Normalized teacher weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")


if __name__ == "__main__":
    main()

