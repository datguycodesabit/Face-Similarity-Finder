from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)
LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
EYEBROWS = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)
NOSE = (168, 6, 197, 195, 5, 4, 45, 220, 115, 48, 64, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275)
LIPS = (
    0,
    13,
    14,
    17,
    37,
    39,
    40,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    146,
    178,
    181,
    185,
    191,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,
)

# The old endpoint keeps its 99-point input contract. V2 uses every base-mesh
# point except expression-sensitive lips; iris points 468-477 are never selected.
LANDMARK_INDICES = FACE_OVAL + LEFT_EYE + RIGHT_EYE + EYEBROWS + NOSE
DENSE_LANDMARK_INDICES = tuple(index for index in range(468) if index not in frozenset(LIPS))
DENSE_INDEX = {mesh_index: position for position, mesh_index in enumerate(DENSE_LANDMARK_INDICES)}
ALIGNMENT_FEATURES = frozenset(LEFT_EYE + RIGHT_EYE + EYEBROWS + NOSE)
DENSE_WEIGHTS = np.asarray(
    [
        4.0 if index in FACE_OVAL else 0.25 if index in ALIGNMENT_FEATURES else 1.0
        for index in DENSE_LANDMARK_INDICES
    ],
    dtype=np.float64,
)
DESCRIPTOR_NAMES = (
    "length_width_ratio",
    "forehead_cheek_ratio",
    "temple_cheek_ratio",
    "jaw_cheek_ratio",
    "jaw_taper",
    "chin_cheek_ratio",
    "jaw_angularity",
    "upper_third",
    "middle_third",
    "lower_third",
)
LANDMARK_VERSION = "structural-v2-dense-weighted-three-view"

FloatArray = NDArray[np.float64]


def _select(full_landmarks: FloatArray, indices: Sequence[int]) -> FloatArray:
    points = np.asarray(full_landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("landmarks must have shape (N, 3)")
    if points.shape[0] <= max(indices):
        raise ValueError("landmark mesh does not contain the required points")
    return cast(FloatArray, points[np.asarray(indices)])


def select_landmarks(full_landmarks: FloatArray) -> FloatArray:
    return _select(full_landmarks, LANDMARK_INDICES)


def select_dense_landmarks(full_landmarks: FloatArray) -> FloatArray:
    return _select(full_landmarks, DENSE_LANDMARK_INDICES)


def _scale(points: FloatArray, weights: FloatArray) -> FloatArray:
    if not np.isfinite(points).all():
        raise ValueError("landmarks contain non-finite values")
    scale = float(np.sqrt(np.average(np.sum(points**2, axis=1), weights=weights)))
    if scale < 1e-12:
        raise ValueError("landmark scale is degenerate")
    return points / scale


def normalize_landmarks(points: FloatArray) -> FloatArray:
    """Legacy 99-point translation, roll, and scale normalization."""
    data = np.asarray(points, dtype=np.float64)
    if data.shape != (len(LANDMARK_INDICES), 3):
        raise ValueError(f"expected {(len(LANDMARK_INDICES), 3)}, got {data.shape}")
    left_start = len(FACE_OVAL)
    right_start = left_start + len(LEFT_EYE)
    left_center = data[left_start:right_start, :2].mean(axis=0)
    right_center = data[right_start : right_start + len(RIGHT_EYE), :2].mean(axis=0)
    eye_axis = right_center - left_center
    if np.linalg.norm(eye_axis) < 1e-12:
        raise ValueError("eye centres are degenerate")
    angle = np.arctan2(eye_axis[1], eye_axis[0])
    cosine, sine = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    centred = data - data.mean(axis=0, keepdims=True)
    centred[:, :2] = centred[:, :2] @ rotation.T
    return _scale(centred, np.ones(len(data), dtype=np.float64))


def normalize_dense_landmarks(
    points: FloatArray, pose_rotation: FloatArray | None = None
) -> FloatArray:
    """Remove translation, full 3D head pose, and uniform scale."""
    data = np.asarray(points, dtype=np.float64)
    expected = (len(DENSE_LANDMARK_INDICES), 3)
    if data.shape != expected:
        raise ValueError(f"expected {expected}, got {data.shape}")
    centred = data - np.average(data, axis=0, weights=DENSE_WEIGHTS)
    if pose_rotation is not None:
        rotation = np.asarray(pose_rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("pose rotation must have shape (3, 3)")
        centred = centred @ rotation
    return _scale(centred, DENSE_WEIGHTS)


def rms_distance(first: FloatArray, second: FloatArray) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("vectors must have identical shapes")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def weighted_dense_distance(first: FloatArray, second: FloatArray) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    expected = (len(DENSE_LANDMARK_INDICES), 3)
    if a.shape != expected or b.shape != expected:
        raise ValueError(f"dense vectors must both have shape {expected}")
    aligned = align_dense_mesh(b, a)
    return float(np.sqrt(np.average(np.sum((a - aligned) ** 2, axis=1), weights=DENSE_WEIGHTS)))


def align_dense_mesh(moving: FloatArray, reference: FloatArray) -> FloatArray:
    """Rigidly align an already centered/scaled dense mesh with weighted Kabsch."""
    moving_array = np.asarray(moving, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    expected = (len(DENSE_LANDMARK_INDICES), 3)
    if moving_array.shape != expected or reference_array.shape != expected:
        raise ValueError(f"dense vectors must both have shape {expected}")
    moving_centered = moving_array - np.average(moving_array, axis=0, weights=DENSE_WEIGHTS)
    reference_centered = reference_array - np.average(
        reference_array, axis=0, weights=DENSE_WEIGHTS
    )
    covariance = (moving_centered * DENSE_WEIGHTS[:, None]).T @ reference_centered
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return cast(FloatArray, moving_centered @ rotation)


def _distance(points: FloatArray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[DENSE_INDEX[first]] - points[DENSE_INDEX[second]]))


def _angle(points: FloatArray, first: int, vertex: int, third: int) -> float:
    a = points[DENSE_INDEX[first]] - points[DENSE_INDEX[vertex]]
    b = points[DENSE_INDEX[third]] - points[DENSE_INDEX[vertex]]
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)) / np.pi)


def shape_measurements(points: FloatArray) -> dict[str, float]:
    data = np.asarray(points, dtype=np.float64)
    if data.shape != (len(DENSE_LANDMARK_INDICES), 3):
        raise ValueError("shape measurements require a normalized dense mesh")
    length, cheek = _distance(data, 10, 152), _distance(data, 234, 454)
    if min(length, cheek) < 1e-12:
        raise ValueError("face dimensions are degenerate")
    jaw = _distance(data, 172, 397)
    return {
        "length_width_ratio": length / cheek,
        "forehead_cheek_ratio": _distance(data, 103, 332) / cheek,
        "temple_cheek_ratio": _distance(data, 127, 356) / cheek,
        "jaw_cheek_ratio": jaw / cheek,
        "jaw_taper": 1.0 - jaw / cheek,
        "chin_cheek_ratio": _distance(data, 148, 377) / cheek,
        "jaw_angularity": 1.0 - (_angle(data, 234, 172, 152) + _angle(data, 454, 397, 152)) / 2.0,
        "upper_third": _distance(data, 10, 168) / length,
        "middle_third": _distance(data, 168, 2) / length,
        "lower_third": _distance(data, 2, 152) / length,
    }


def descriptor_vector(measurements: Mapping[str, float]) -> FloatArray:
    return np.asarray([measurements[name] for name in DESCRIPTOR_NAMES], dtype=np.float64)


def descriptor_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    difference = descriptor_vector(first) - descriptor_vector(second)
    return float(np.sqrt(np.mean(difference**2)))


def combined_shape_distance(
    first_points: FloatArray,
    first_measurements: Mapping[str, float],
    second_points: FloatArray,
    second_measurements: Mapping[str, float],
) -> tuple[float, float, float]:
    proportions = descriptor_distance(first_measurements, second_measurements)
    silhouette = weighted_dense_distance(first_points, second_points)
    return 0.7 * proportions + 0.3 * silhouette, silhouette, proportions


def aggregate_views(vectors: Sequence[FloatArray]) -> tuple[FloatArray, float]:
    if len(vectors) != 3:
        raise ValueError("exactly three face views are required")
    front = np.asarray(vectors[0], dtype=np.float64)
    stack = np.stack([front, *(align_dense_mesh(vector, front) for vector in vectors[1:])])
    median = np.median(stack, axis=0)
    disagreement = float(np.mean(np.std(stack, axis=0)))
    return median, round(100.0 * max(0.0, 1.0 - disagreement / 0.12), 1)


def shape_memberships(measurements: Mapping[str, float]) -> list[tuple[str, float]]:
    """Deterministic soft salon-style labels; scores are not probabilities."""
    aspect, forehead = measurements["length_width_ratio"], measurements["forehead_cheek_ratio"]
    jaw, angularity = measurements["jaw_cheek_ratio"], measurements["jaw_angularity"]

    def near(value: float, target: float, width: float) -> float:
        return max(0.0, 1.0 - abs(value - target) / width)

    scores = {
        "round": (near(aspect, 1.08, 0.22) + near(angularity, 0.25, 0.22)) / 2,
        "square": (near(aspect, 1.10, 0.25) + near(angularity, 0.48, 0.25) + near(jaw, 0.96, 0.18))
        / 3,
        "oblong": max(0.0, min(1.0, (aspect - 1.30) / 0.25)),
        "heart": (
            max(0.0, min(1.0, (forehead - 0.98) / 0.12)) + max(0.0, min(1.0, (0.94 - jaw) / 0.14))
        )
        / 2,
        "triangle": (
            max(0.0, min(1.0, (jaw - 0.96) / 0.14)) + max(0.0, min(1.0, (1.0 - forehead) / 0.12))
        )
        / 2,
        "diamond": (
            max(0.0, min(1.0, (0.98 - forehead) / 0.10)) + max(0.0, min(1.0, (0.98 - jaw) / 0.10))
        )
        / 2,
        "oval": (near(aspect, 1.32, 0.30) + near(forehead, 0.94, 0.16) + near(jaw, 0.88, 0.16)) / 3,
    }
    return sorted(
        ((name, round(score, 4)) for name, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def display_similarity(distance: float) -> float:
    return round(100.0 / (1.0 + 4.0 * max(0.0, distance)), 1)
