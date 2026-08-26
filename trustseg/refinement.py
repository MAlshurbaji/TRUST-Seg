from __future__ import annotations

import numpy as np


def smooth_delta_attention(
    student_probability: np.ndarray,
    teacher_consensus: np.ndarray,
    teacher_uncertainty: np.ndarray,
    bbox: np.ndarray,
    uncertainty_weight: float,
    student_threshold: float,
    confidence_threshold: float,
    attention_steepness: float,
    alpha_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    student_probability = np.clip(student_probability, 0.0, 1.0).astype(np.float32)
    teacher_consensus = np.clip(teacher_consensus, 0.0, 1.0).astype(np.float32)
    teacher_uncertainty = np.clip(teacher_uncertainty, 0.0, 1.0).astype(np.float32)
    bbox_gate = (bbox > 0).astype(np.float32)

    student_confidence = 2.0 * np.abs(student_probability - 0.5)
    teacher_certainty = 1.0 - teacher_uncertainty
    delta = student_confidence - teacher_certainty
    attention_alpha = 1.0 / (1.0 + np.exp(-float(attention_steepness) * delta))

    candidate = (
        attention_alpha * student_probability
        + (1.0 - attention_alpha) * teacher_consensus
    )
    safeguard = (student_confidence >= float(student_threshold)).astype(np.float32)
    refined = safeguard * candidate + (1.0 - safeguard) * teacher_consensus
    refined *= bbox_gate

    disagreement = np.abs(student_probability - teacher_consensus)
    refined_uncertainty = (
        float(uncertainty_weight) * teacher_uncertainty
        + (1.0 - float(uncertainty_weight)) * disagreement
    )
    refined_uncertainty = np.clip(refined_uncertainty, 0.0, 1.0) * bbox_gate

    confidence = np.exp(-float(alpha_conf) * refined_uncertainty)
    if confidence_threshold > 0:
        confidence *= confidence >= float(confidence_threshold)

    return (
        refined.astype(np.float32),
        refined_uncertainty.astype(np.float32),
        confidence.astype(np.float32),
        attention_alpha.astype(np.float32),
    )
