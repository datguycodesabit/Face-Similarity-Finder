from __future__ import annotations

import numpy as np

from face_match.geometry import DENSE_LANDMARK_INDICES
from face_match.shape_v3 import (
    COMPONENT_NAMES,
    COMPONENT_WEIGHTS,
    DESCRIPTOR_NAMES,
    HybridCalibration,
    RobustCalibration,
    isotropic_landmarks,
    normalize_v3_landmarks,
    rigidly_align_regions,
    shape_descriptors,
    surface_distances,
)

from .helpers import dense_points


def _full_mesh(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    full = rng.normal(0.0, 0.03, (468, 3))
    dense = dense_points(seed)
    for position, mesh_index in enumerate(DENSE_LANDMARK_INDICES):
        full[mesh_index] = dense[position]
    return full


def test_isotropic_conversion_undoes_image_axis_normalization() -> None:
    pixels = _full_mesh(30) * 120.0 + np.array([300.0, 400.0, 0.0])
    first = pixels / np.array([600.0, 800.0, 600.0])
    second = pixels / np.array([1200.0, 900.0, 1200.0])
    assert np.allclose(isotropic_landmarks(first, 600, 800), pixels)
    assert np.allclose(isotropic_landmarks(second, 1200, 900), pixels)
    assert np.allclose(
        normalize_v3_landmarks(first, 600, 800),
        normalize_v3_landmarks(second, 1200, 900),
    )


def test_v3_descriptors_are_complete_finite_and_nonduplicated() -> None:
    measurements = shape_descriptors(dense_points(41))
    assert tuple(measurements) == DESCRIPTOR_NAMES
    assert all(np.isfinite(value) for value in measurements.values())
    assert "jaw_taper" not in measurements
    assert measurements["upper_jaw_ratio"] != measurements["gonial_width_ratio"]
    assert measurements["chin_roundness"] >= 1.0


def test_robust_calibration_round_trips_and_uses_fixed_component_weights() -> None:
    samples = [shape_descriptors(dense_points(seed)) for seed in range(50, 70)]
    calibration = RobustCalibration.fit(samples)
    restored = RobustCalibration.from_json(calibration.to_json())
    total, components = restored.distance(samples[0], samples[1])
    assert tuple(components) == COMPONENT_NAMES
    assert np.isclose(total, sum(COMPONENT_WEIGHTS[name] * components[name] for name in components))
    assert np.isclose(restored.distance(samples[0], samples[0])[0], 0.0)


def test_jaw_change_is_isolated_from_eye_and_nose_components() -> None:
    samples = [shape_descriptors(dense_points(seed)) for seed in range(70, 100)]
    calibration = RobustCalibration.fit(samples)
    baseline = dict(samples[0])
    changed = dict(baseline)
    changed["gonial_width_ratio"] += 0.05
    changed["lower_jaw_ratio"] += 0.04
    changed["chin_width_ratio"] += 0.03
    _, components = calibration.distance(baseline, changed)
    assert components["jaw_chin"] > 0.0
    assert components["eye_brow_geometry"] == 0.0
    assert components["nose_midface_geometry"] == 0.0


def _regions(value: float = 0.0) -> dict[str, np.ndarray]:
    base = dense_points(123)[:12] + value
    return {name: base + index * 0.1 for index, name in enumerate(COMPONENT_NAMES)}


def test_flame_alignment_is_rigid_and_does_not_hide_scale_changes() -> None:
    fixed = _regions()
    angle = np.deg2rad(31.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    moved = {name: points @ rotation + np.array([4.0, -2.0, 1.0]) for name, points in fixed.items()}
    aligned = rigidly_align_regions(moved, fixed)
    assert all(np.allclose(aligned[name], fixed[name]) for name in COMPONENT_NAMES)
    scaled = {name: points * 1.2 for name, points in fixed.items()}
    assert surface_distances(scaled, fixed)["jaw_chin"] > 0.0


def test_hybrid_calibration_uses_the_fixed_five_part_score() -> None:
    descriptors = [shape_descriptors(dense_points(seed)) for seed in range(130, 140)]
    regions = [_regions(index * 0.01) for index in range(10)]
    calibration = HybridCalibration.fit(descriptors, regions)
    changed_regions = {name: points.copy() for name, points in regions[0].items()}
    changed_regions["jaw_chin"][0, 0] += 0.05
    total, components = calibration.distance(
        descriptors[0], changed_regions, descriptors[0], regions[0]
    )
    assert components["jaw_chin"] > 0.0
    assert components["eye_brow_geometry"] == 0.0
    assert components["nose_midface_geometry"] == 0.0
    assert np.isclose(total, sum(COMPONENT_WEIGHTS[name] * components[name] for name in components))
    restored = HybridCalibration.from_json(calibration.to_json())
    assert restored.component_scales == calibration.component_scales
