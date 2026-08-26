from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from medpy import metric


@dataclass(frozen=True)
class SegmentationMetrics:
    dice: float
    iou: float
    hd95: float
    asd: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float] | None = None,
) -> SegmentationMetrics:
    prediction = np.asarray(prediction) > 0
    target = np.asarray(target) > 0

    pred_nonempty = bool(prediction.any())
    target_nonempty = bool(target.any())
    if not pred_nonempty and not target_nonempty:
        return SegmentationMetrics(1.0, 1.0, 0.0, 0.0)
    if pred_nonempty != target_nonempty:
        # This matches the experimental scripts used for the reported results.
        return SegmentationMetrics(0.0, 0.0, 0.0, 0.0)

    return SegmentationMetrics(
        dice=float(metric.binary.dc(prediction, target)),
        iou=float(metric.binary.jc(prediction, target)),
        hd95=float(metric.binary.hd95(prediction, target, voxelspacing=spacing)),
        asd=float(metric.binary.asd(prediction, target, voxelspacing=spacing)),
    )


def mean_metrics(items: list[SegmentationMetrics]) -> SegmentationMetrics:
    if not items:
        raise ValueError("No metrics were provided")
    values = np.asarray([[item.dice, item.iou, item.hd95, item.asd] for item in items])
    means = values.mean(axis=0)
    return SegmentationMetrics(*(float(value) for value in means))

