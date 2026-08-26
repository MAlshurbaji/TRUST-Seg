from __future__ import annotations

import torch
from torch.nn import functional as F


def confidence_weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    elementwise = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (weight * elementwise).sum() / (weight.sum() + eps)


def confidence_weighted_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (weight * probability * target).sum()
    denominator = (weight * probability).sum() + (weight * target).sum()
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def trustseg_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = confidence_weighted_bce(logits, target, weight)
    dice = confidence_weighted_dice(logits, target, weight)
    total = bce + dice
    return total, {"bce": float(bce.detach()), "dice": float(dice.detach())}

