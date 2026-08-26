import numpy as np
import torch

from trustseg.data import collate_volumes, pad_to_factor, unpad
from trustseg.ensemble import aggregate_teacher_masks, normalized_reliability
from trustseg.losses import trustseg_loss
from trustseg.models import UNet3D
from trustseg.refinement import smooth_delta_attention


def test_reliability_weights_sum_to_one():
    weights = normalized_reliability({"a": 0.8, "b": 0.6, "c": 0.4})
    assert np.isclose(sum(weights.values()), 1.0)
    assert weights["a"] > weights["b"] > weights["c"]


def test_ensemble_matches_weighted_bernoulli_variance():
    masks = {
        "a": np.asarray([[[1.0, 0.0]]]),
        "b": np.asarray([[[0.0, 0.0]]]),
    }
    weights = {"a": 0.75, "b": 0.25}
    bbox = np.ones((1, 1, 2), dtype=np.float32)
    consensus, uncertainty, confidence = aggregate_teacher_masks(
        masks,
        weights,
        bbox,
        alpha_conf=1.0,
    )

    assert np.allclose(consensus, [[[0.75, 0.0]]])
    assert np.allclose(uncertainty, 4.0 * consensus * (1.0 - consensus))
    assert np.allclose(confidence, np.exp(-uncertainty))


def test_sda_preserves_bbox_support():
    student = np.asarray([[[0.9, 0.9]]], dtype=np.float32)
    teacher = np.asarray([[[0.2, 0.2]]], dtype=np.float32)
    uncertainty = np.asarray([[[0.8, 0.8]]], dtype=np.float32)
    bbox = np.asarray([[[1.0, 0.0]]], dtype=np.float32)

    refined, refined_uncertainty, _, _ = smooth_delta_attention(
        student,
        teacher,
        uncertainty,
        bbox,
        uncertainty_weight=0.7,
        student_threshold=0.3,
        confidence_threshold=0.0,
        attention_steepness=8.0,
        alpha_conf=1.0,
    )
    assert refined[0, 0, 0] > teacher[0, 0, 0]
    assert refined[0, 0, 1] == 0.0
    assert refined_uncertainty[0, 0, 1] == 0.0

def test_sda_attention_alpha_is_voxel_wise_and_dynamic():
    student = np.asarray([[[0.9, 0.55]]], dtype=np.float32)
    teacher = np.asarray([[[0.2, 0.8]]], dtype=np.float32)
    uncertainty = np.asarray([[[0.8, 0.2]]], dtype=np.float32)
    bbox = np.ones_like(student)

    _, _, _, attention_alpha = smooth_delta_attention(
        student,
        teacher,
        uncertainty,
        bbox,
        uncertainty_weight=0.7,
        student_threshold=0.0,
        confidence_threshold=0.0,
        attention_steepness=8.0,
        alpha_conf=1.0,
    )

    assert attention_alpha[0, 0, 0] > 0.5
    assert attention_alpha[0, 0, 1] < 0.5
    assert not np.isclose(attention_alpha[0, 0, 0], attention_alpha[0, 0, 1])
    
def test_unet_preserves_spatial_shape():
    model = UNet3D(in_channels=2, base_channels=4)
    inputs = torch.randn(1, 2, 16, 32, 32)
    assert model(inputs).shape == (1, 1, 16, 32, 32)
    assert isinstance(model.dropout, torch.nn.Dropout)
    assert not isinstance(model.dropout, torch.nn.Dropout3d)


def test_trustseg_loss_matches_confidence_weighted_equations():
    logits = torch.tensor([[[[[0.0, 1.0]]]]])
    target = torch.tensor([[[[[0.0, 1.0]]]]])
    weight = torch.tensor([[[[[1.0, 0.25]]]]])

    total, parts = trustseg_loss(logits, target, weight)
    probability = torch.sigmoid(logits)
    elementwise_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    expected_bce = (weight * elementwise_bce).sum() / weight.sum()
    expected_dice = 1.0 - (
        2.0 * (weight * probability * target).sum() + 1e-8
    ) / ((weight * probability).sum() + (weight * target).sum() + 1e-8)

    assert torch.allclose(total, expected_bce + expected_dice)
    assert np.isclose(parts["bce"], float(expected_bce))
    assert np.isclose(parts["dice"], float(expected_dice))


def test_volume_batching_masks_padding_from_the_loss():
    def sample(case_id: str, depth: int) -> dict[str, object]:
        return {
            "case_id": case_id,
            "inputs": torch.ones(2, depth, 16, 16),
            "target": torch.ones(1, depth, 16, 16),
            "weight": torch.ones(1, depth, 16, 16),
            "shape": (depth, 16, 16),
            "reference_path": f"{case_id}_image.nii.gz",
        }

    batch = collate_volumes([sample("001", 15), sample("002", 17)])
    assert batch["inputs"].shape == (2, 2, 17, 16, 16)
    assert torch.count_nonzero(batch["weight"][0, :, 15]) == 0

    padded, padding = pad_to_factor(batch["inputs"], factor=16)
    assert padded.shape == (2, 2, 32, 16, 16)
    assert torch.equal(unpad(padded, padding), batch["inputs"])
