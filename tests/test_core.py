import numpy as np
import torch

from trustseg.ensemble import aggregate_teacher_masks, normalized_reliability
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
    consensus, uncertainty, confidence = aggregate_teacher_masks(masks, weights, bbox, 1.0)

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

