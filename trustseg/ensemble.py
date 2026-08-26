from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trustseg.io import case_id_from_name, list_nifti, read_volume, require_same_shape, write_volume


def normalized_reliability(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        raise ValueError("At least one teacher reliability score is required")
    if any(score < 0 for score in scores.values()):
        raise ValueError("Teacher reliability scores must be non-negative")
    total = float(sum(scores.values()))
    if total <= 0:
        raise ValueError("The sum of teacher reliability scores must be positive")
    return {name: float(score / total) for name, score in scores.items()}


def aggregate_teacher_masks(
    masks: dict[str, np.ndarray],
    weights: dict[str, float],
    bbox: np.ndarray,
    confidence_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = set(weights) - set(masks)
    if missing:
        raise KeyError(f"Teacher masks are missing for: {sorted(missing)}")

    bbox_gate = (bbox > 0).astype(np.float32)
    stacked = np.stack(
        [(masks[name] > 0).astype(np.float32) * bbox_gate for name in weights],
        axis=0,
    )
    weight_array = np.asarray([weights[name] for name in weights], dtype=np.float32)
    weight_array = weight_array.reshape((-1,) + (1,) * bbox.ndim)

    consensus = np.sum(weight_array * stacked, axis=0)
    uncertainty = np.sum(weight_array * (stacked - consensus[None, ...]) ** 2, axis=0)
    uncertainty = np.clip(uncertainty / 0.25, 0.0, 1.0)
    confidence = np.exp(-float(confidence_alpha) * uncertainty)
    return (
        consensus.astype(np.float32),
        uncertainty.astype(np.float32),
        confidence.astype(np.float32),
    )


def build_ensemble(
    teacher_dirs: dict[str, Path],
    reliability_scores: dict[str, float],
    bbox_dir: Path,
    output_dir: Path,
    confidence_alpha: float = 1.0,
) -> dict[str, float]:
    if set(teacher_dirs) != set(reliability_scores):
        raise ValueError("Teacher directories and reliability scores must use identical names")

    weights = normalized_reliability(reliability_scores)
    anchor_name = next(iter(teacher_dirs))
    anchor_files = list_nifti(teacher_dirs[anchor_name])
    if not anchor_files:
        raise RuntimeError(f"No teacher masks found in {teacher_dirs[anchor_name]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for anchor_path in anchor_files:
        case_id = case_id_from_name(anchor_path.name)
        bbox, bbox_reference = read_volume(bbox_dir / f"{case_id}_bbox.nii.gz")

        masks: dict[str, np.ndarray] = {}
        for teacher_name, teacher_dir in teacher_dirs.items():
            teacher_path = teacher_dir / f"{case_id}_image_mask.nii.gz"
            mask, _ = read_volume(teacher_path)
            masks[teacher_name] = mask

        require_same_shape(case_id, [("bbox", bbox), *masks.items()])
        consensus, uncertainty, confidence = aggregate_teacher_masks(
            masks,
            weights,
            bbox,
            confidence_alpha,
        )
        prefix = output_dir / f"{case_id}_image_mask"
        write_volume(consensus, f"{prefix}.nii.gz", bbox_reference)
        write_volume(uncertainty, f"{prefix}_unc.nii.gz", bbox_reference)
        write_volume(confidence, f"{prefix}_conf.nii.gz", bbox_reference)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "confidence_alpha": confidence_alpha,
        "reliability_scores": reliability_scores,
        "normalized_weights": weights,
        "teacher_directories": {name: str(path) for name, path in teacher_dirs.items()},
    }
    with (output_dir / "ensemble_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return weights

