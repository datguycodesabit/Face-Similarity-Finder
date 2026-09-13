from __future__ import annotations

import numpy as np

from face_match.geometry import display_similarity, normalize_landmarks, rms_distance

from .helpers import structural_points


def test_normalization_is_translation_rotation_and_scale_invariant() -> None:
    points = structural_points()
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = points.copy()
    transformed[:, :2] = transformed[:, :2] @ rotation.T
    transformed = transformed * 3.7 + np.array([12.0, -7.0, 2.5])
    np.testing.assert_allclose(
        normalize_landmarks(points), normalize_landmarks(transformed), atol=1e-10
    )


def test_rms_distance_is_deterministic_and_similarity_is_monotonic() -> None:
    first = normalize_landmarks(structural_points(5))
    second = normalize_landmarks(structural_points(6))
    expected = float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))
    assert rms_distance(first, second) == expected
    assert rms_distance(first, second) == rms_distance(first, second)
    assert display_similarity(0.1) > display_similarity(0.2) > display_similarity(0.5)
