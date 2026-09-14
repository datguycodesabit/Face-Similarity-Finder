from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .geometry import DENSE_INDEX, DENSE_LANDMARK_INDICES, DENSE_WEIGHTS

FloatArray = NDArray[np.float64]

V3_ALGORITHM_VERSION = "shape-v3-isotropic-mica-calibrated"

COMPONENT_NAMES = (
    "jaw_chin",
    "outline_cheeks",
    "global_proportions",
    "eye_brow_geometry",
    "nose_midface_geometry",
)

COMPONENT_WEIGHTS: dict[str, float] = {
    "jaw_chin": 0.45,
    "outline_cheeks": 0.20,
    "global_proportions": 0.15,
    "eye_brow_geometry": 0.10,
    "nose_midface_geometry": 0.10,
}

DESCRIPTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "jaw_chin": (
        "upper_jaw_ratio",
        "middle_jaw_ratio",
        "gonial_width_ratio",
        "lower_jaw_ratio",
        "chin_width_ratio",
        "jaw_flare",
        "gonial_angle",
        "chin_projection",
        "chin_roundness",
    ),
    "outline_cheeks": (
        "forehead_width_ratio",
        "upper_face_width_ratio",
        "cheek_depth_ratio",
    ),
    "global_proportions": (
        "length_width_ratio",
        "upper_third",
        "middle_third",
        "lower_third",
    ),
    "eye_brow_geometry": (
        "eye_spacing_ratio",
        "mean_eye_width_ratio",
        "inner_brow_spacing_ratio",
    ),
    "nose_midface_geometry": (
        "nose_width_ratio",
        "nose_length_ratio",
        "midface_projection",
    ),
}

DESCRIPTOR_NAMES = tuple(
    descriptor for component in COMPONENT_NAMES for descriptor in DESCRIPTOR_GROUPS[component]
)


def isotropic_landmarks(
    normalized_landmarks: FloatArray, image_width: int, image_height: int
) -> FloatArray:
    """Convert MediaPipe image-normalized x/y/z into one isotropic camera scale."""
    points = np.asarray(normalized_landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("landmarks must have shape (N, 3)")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    scale = np.asarray([image_width, image_height, image_width], dtype=np.float64)
    return points * scale


def normalize_v3_landmarks(
    normalized_landmarks: FloatArray,
    image_width: int,
    image_height: int,
    pose_rotation: FloatArray | None = None,
) -> FloatArray:
    """Select the expression-reduced mesh and remove translation, pose, and scale."""
    full = isotropic_landmarks(normalized_landmarks, image_width, image_height)
    if full.shape[0] <= max(DENSE_LANDMARK_INDICES):
        raise ValueError("landmark mesh does not contain the required points")
    selected = full[np.asarray(DENSE_LANDMARK_INDICES)]
    centered = selected - np.average(selected, axis=0, weights=DENSE_WEIGHTS)
    if pose_rotation is not None:
        rotation = np.asarray(pose_rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("pose rotation must have shape (3, 3)")
        centered = centered @ rotation
    scale = float(np.sqrt(np.average(np.sum(centered**2, axis=1), weights=DENSE_WEIGHTS)))
    if scale < 1e-12:
        raise ValueError("landmark scale is degenerate")
    return cast(FloatArray, centered / scale)


def _point(points: FloatArray, mesh_index: int) -> FloatArray:
    return cast(FloatArray, points[DENSE_INDEX[mesh_index]])


def _distance(points: FloatArray, first: int, second: int) -> float:
    return float(np.linalg.norm(_point(points, first) - _point(points, second)))


def _angle(points: FloatArray, first: int, vertex: int, third: int) -> float:
    a = _point(points, first) - _point(points, vertex)
    b = _point(points, third) - _point(points, vertex)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        raise ValueError("jaw angle is degenerate")
    return float(np.arccos(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)) / np.pi)


def _path_length(points: FloatArray, indices: Sequence[int]) -> float:
    return sum(
        _distance(points, first, second)
        for first, second in zip(indices, indices[1:], strict=False)
    )


def shape_descriptors(points: FloatArray) -> dict[str, float]:
    """Measure nonduplicated haircut-relevant geometry on a normalized dense mesh."""
    mesh = np.asarray(points, dtype=np.float64)
    expected = (len(DENSE_LANDMARK_INDICES), 3)
    if mesh.shape != expected or not np.isfinite(mesh).all():
        raise ValueError(f"shape descriptors require a finite mesh with shape {expected}")

    face_length = _distance(mesh, 10, 152)
    cheek_width = _distance(mesh, 234, 454)
    if min(face_length, cheek_width) < 1e-12:
        raise ValueError("face dimensions are degenerate")

    upper_jaw = _distance(mesh, 132, 361)
    middle_jaw = _distance(mesh, 58, 288)
    gonial_width = _distance(mesh, 172, 397)
    lower_jaw = _distance(mesh, 136, 365)
    chin_width = _distance(mesh, 176, 400)
    chin_arc = _path_length(mesh, (176, 148, 152, 377, 400))
    cheek_midpoint = (_point(mesh, 234) + _point(mesh, 454)) / 2.0
    temple_midpoint = (_point(mesh, 127) + _point(mesh, 356)) / 2.0
    eye_width = (_distance(mesh, 33, 133) + _distance(mesh, 362, 263)) / 2.0

    descriptors = {
        "upper_jaw_ratio": upper_jaw / cheek_width,
        "middle_jaw_ratio": middle_jaw / cheek_width,
        "gonial_width_ratio": gonial_width / cheek_width,
        "lower_jaw_ratio": lower_jaw / cheek_width,
        "chin_width_ratio": chin_width / cheek_width,
        "jaw_flare": gonial_width / upper_jaw,
        "gonial_angle": (_angle(mesh, 58, 172, 136) + _angle(mesh, 288, 397, 365)) / 2.0,
        "chin_projection": float((_point(mesh, 152)[2] - cheek_midpoint[2]) / cheek_width),
        "chin_roundness": chin_arc / max(chin_width, 1e-12),
        "forehead_width_ratio": _distance(mesh, 103, 332) / cheek_width,
        "upper_face_width_ratio": _distance(mesh, 54, 284) / cheek_width,
        "cheek_depth_ratio": abs(float((cheek_midpoint[2] - temple_midpoint[2]) / cheek_width)),
        "length_width_ratio": face_length / cheek_width,
        "upper_third": _distance(mesh, 10, 168) / face_length,
        "middle_third": _distance(mesh, 168, 2) / face_length,
        "lower_third": _distance(mesh, 2, 152) / face_length,
        "eye_spacing_ratio": _distance(mesh, 133, 362) / cheek_width,
        "mean_eye_width_ratio": eye_width / cheek_width,
        "inner_brow_spacing_ratio": _distance(mesh, 107, 336) / cheek_width,
        "nose_width_ratio": _distance(mesh, 98, 327) / cheek_width,
        "nose_length_ratio": _distance(mesh, 168, 1) / face_length,
        "midface_projection": float((_point(mesh, 1)[2] - cheek_midpoint[2]) / cheek_width),
    }
    if tuple(descriptors) != DESCRIPTOR_NAMES:
        raise AssertionError("descriptor order does not match the V3 contract")
    return descriptors


def descriptor_array(measurements: Mapping[str, float]) -> FloatArray:
    try:
        values = np.asarray([measurements[name] for name in DESCRIPTOR_NAMES], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"missing V3 descriptor: {error.args[0]}") from error
    if not np.isfinite(values).all():
        raise ValueError("V3 descriptors must be finite")
    return values


@dataclass(frozen=True)
class RobustCalibration:
    median: FloatArray
    scale: FloatArray
    whiteners: dict[str, FloatArray]

    @classmethod
    def fit(
        cls, samples: Sequence[Mapping[str, float]], shrinkage: float = 0.10
    ) -> RobustCalibration:
        if len(samples) < 3:
            raise ValueError("at least three reference samples are required for calibration")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be between zero and one")
        matrix = np.stack([descriptor_array(sample) for sample in samples])
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-6)
        standardized = (matrix - median) / scale
        whiteners: dict[str, FloatArray] = {}
        for component in COMPONENT_NAMES:
            indices = [DESCRIPTOR_NAMES.index(name) for name in DESCRIPTOR_GROUPS[component]]
            block = standardized[:, indices]
            covariance = np.atleast_2d(np.cov(block, rowvar=False, ddof=1))
            diagonal = np.diag(np.diag(covariance))
            regularized = (1.0 - shrinkage) * covariance + shrinkage * diagonal
            regularized += np.eye(len(indices), dtype=np.float64) * 1e-6
            eigenvalues, eigenvectors = np.linalg.eigh(regularized)
            inverse_root = (
                eigenvectors
                @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-6)))
                @ eigenvectors.T
            )
            whiteners[component] = cast(FloatArray, inverse_root)
        return cls(median, scale, whiteners)

    def standardize(self, measurements: Mapping[str, float]) -> FloatArray:
        return (descriptor_array(measurements) - self.median) / self.scale

    def component_distances(
        self, first: Mapping[str, float], second: Mapping[str, float]
    ) -> dict[str, float]:
        difference = self.standardize(first) - self.standardize(second)
        distances: dict[str, float] = {}
        for component in COMPONENT_NAMES:
            indices = [DESCRIPTOR_NAMES.index(name) for name in DESCRIPTOR_GROUPS[component]]
            whitened = self.whiteners[component] @ difference[indices]
            distances[component] = float(np.sqrt(np.mean(whitened**2)))
        return distances

    def distance(
        self, first: Mapping[str, float], second: Mapping[str, float]
    ) -> tuple[float, dict[str, float]]:
        components = self.component_distances(first, second)
        total = sum(COMPONENT_WEIGHTS[name] * components[name] for name in COMPONENT_NAMES)
        return total, components

    def to_json(self) -> dict[str, Any]:
        return {
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "whiteners": {name: matrix.tolist() for name, matrix in self.whiteners.items()},
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> RobustCalibration:
        if tuple(payload.get("descriptor_names", ())) != DESCRIPTOR_NAMES:
            raise ValueError("calibration descriptor contract is incompatible")
        median = np.asarray(payload["median"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        if median.shape != (len(DESCRIPTOR_NAMES),) or scale.shape != median.shape:
            raise ValueError("calibration vectors have invalid dimensions")
        whiteners = {
            name: np.asarray(payload["whiteners"][name], dtype=np.float64)
            for name in COMPONENT_NAMES
        }
        for name, matrix in whiteners.items():
            expected = len(DESCRIPTOR_GROUPS[name])
            if matrix.shape != (expected, expected):
                raise ValueError(f"calibration whitener for {name} has invalid dimensions")
        return cls(median, scale, whiteners)


def rigidly_align_regions(
    moving: Mapping[str, FloatArray], fixed: Mapping[str, FloatArray]
) -> dict[str, FloatArray]:
    """Rigidly align corresponding FLAME vertices without changing their scale or shape."""
    if set(moving) != set(COMPONENT_NAMES) or set(fixed) != set(COMPONENT_NAMES):
        raise ValueError("FLAME regions must contain the complete V3 component contract")
    moving_blocks = [np.asarray(moving[name], dtype=np.float64) for name in COMPONENT_NAMES]
    fixed_blocks = [np.asarray(fixed[name], dtype=np.float64) for name in COMPONENT_NAMES]
    if any(
        first.ndim != 2
        or first.shape[1] != 3
        or first.shape != second.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        for first, second in zip(moving_blocks, fixed_blocks, strict=True)
    ):
        raise ValueError("corresponding FLAME regions must be finite N x 3 arrays")
    moving_all = np.concatenate(moving_blocks)
    fixed_all = np.concatenate(fixed_blocks)
    moving_center = np.mean(moving_all, axis=0)
    fixed_center = np.mean(fixed_all, axis=0)
    covariance = (moving_all - moving_center).T @ (fixed_all - fixed_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return {
        name: cast(FloatArray, (block - moving_center) @ rotation + fixed_center)
        for name, block in zip(COMPONENT_NAMES, moving_blocks, strict=True)
    }


def surface_distances(
    first: Mapping[str, FloatArray], second: Mapping[str, FloatArray]
) -> dict[str, float]:
    """Return rigid-only corresponding-vertex RMS distances for FLAME surface groups."""
    aligned = rigidly_align_regions(first, second)
    return {
        name: float(np.sqrt(np.mean(np.sum((aligned[name] - second[name]) ** 2, axis=1))))
        for name in ("jaw_chin", "outline_cheeks")
    }


@dataclass(frozen=True)
class HybridCalibration:
    """Population calibration that puts all five V3 components on comparable scales."""

    descriptors: RobustCalibration
    component_scales: dict[str, float]

    @classmethod
    def fit(
        cls,
        descriptor_samples: Sequence[Mapping[str, float]],
        region_samples: Sequence[Mapping[str, FloatArray]],
    ) -> HybridCalibration:
        if len(descriptor_samples) != len(region_samples):
            raise ValueError("descriptor and FLAME region sample counts must match")
        if len(descriptor_samples) < 3:
            raise ValueError("at least three identity prototypes are required for calibration")
        descriptors = RobustCalibration.fit(descriptor_samples)
        samples: dict[str, list[float]] = {name: [] for name in COMPONENT_NAMES}
        for first in range(len(descriptor_samples)):
            for second in range(first + 1, len(descriptor_samples)):
                surfaces = surface_distances(region_samples[first], region_samples[second])
                scalar = descriptors.component_distances(
                    descriptor_samples[first], descriptor_samples[second]
                )
                samples["jaw_chin"].append(surfaces["jaw_chin"])
                samples["outline_cheeks"].append(surfaces["outline_cheeks"])
                for name in COMPONENT_NAMES[2:]:
                    samples[name].append(scalar[name])
        scales = {name: max(float(np.median(values)), 1e-9) for name, values in samples.items()}
        return cls(descriptors, scales)

    def component_distances(
        self,
        first_descriptors: Mapping[str, float],
        first_regions: Mapping[str, FloatArray],
        second_descriptors: Mapping[str, float],
        second_regions: Mapping[str, FloatArray],
    ) -> dict[str, float]:
        surfaces = surface_distances(first_regions, second_regions)
        scalar = self.descriptors.component_distances(first_descriptors, second_descriptors)
        raw = {
            "jaw_chin": surfaces["jaw_chin"],
            "outline_cheeks": surfaces["outline_cheeks"],
            "global_proportions": scalar["global_proportions"],
            "eye_brow_geometry": scalar["eye_brow_geometry"],
            "nose_midface_geometry": scalar["nose_midface_geometry"],
        }
        return {name: raw[name] / self.component_scales[name] for name in COMPONENT_NAMES}

    def distance(
        self,
        first_descriptors: Mapping[str, float],
        first_regions: Mapping[str, FloatArray],
        second_descriptors: Mapping[str, float],
        second_regions: Mapping[str, FloatArray],
    ) -> tuple[float, dict[str, float]]:
        components = self.component_distances(
            first_descriptors, first_regions, second_descriptors, second_regions
        )
        total = sum(COMPONENT_WEIGHTS[name] * components[name] for name in COMPONENT_NAMES)
        return total, components

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "v3-hybrid",
            "descriptors": self.descriptors.to_json(),
            "component_scales": self.component_scales,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> HybridCalibration:
        if payload.get("kind") != "v3-hybrid":
            raise ValueError("calibration is not a V3 hybrid calibration")
        descriptors = RobustCalibration.from_json(cast(Mapping[str, Any], payload["descriptors"]))
        raw_scales = cast(Mapping[str, Any], payload["component_scales"])
        scales = {name: float(raw_scales[name]) for name in COMPONENT_NAMES}
        if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
            raise ValueError("hybrid calibration component scales must be positive and finite")
        return cls(descriptors, scales)


def maximum_descriptor_correlation(samples: Sequence[Mapping[str, float]]) -> float:
    matrix = np.stack([descriptor_array(sample) for sample in samples])
    correlation = np.asarray(np.corrcoef(matrix, rowvar=False), dtype=np.float64)
    np.fill_diagonal(correlation, 0.0)
    return float(np.nanmax(np.abs(correlation)))


def calibrated_shape_labels(
    measurements: Mapping[str, float], population: Sequence[Mapping[str, float]]
) -> list[dict[str, Any]]:
    """Derive styling labels from gallery percentiles rather than fixed ratio cutoffs."""
    if len(population) < 5:
        raise ValueError("at least five population prototypes are required for shape labels")

    def percentile(name: str) -> float:
        values = np.asarray([sample[name] for sample in population], dtype=np.float64)
        return float(
            (np.sum(values < measurements[name]) + 0.5 * np.sum(values == measurements[name]))
            / len(values)
        )

    names = (
        "length_width_ratio",
        "forehead_width_ratio",
        "upper_face_width_ratio",
        "gonial_width_ratio",
        "chin_width_ratio",
        "gonial_angle",
        "chin_roundness",
    )
    ranks = {name: percentile(name) for name in names}
    scores = {
        "oblong": ranks["length_width_ratio"],
        "round": (
            1.0 - ranks["length_width_ratio"] + ranks["gonial_angle"] + ranks["chin_roundness"]
        )
        / 3.0,
        "square": (
            ranks["gonial_width_ratio"] + ranks["chin_width_ratio"] + (1.0 - ranks["gonial_angle"])
        )
        / 3.0,
        "heart": (
            ranks["forehead_width_ratio"]
            + (1.0 - ranks["gonial_width_ratio"])
            + (1.0 - ranks["chin_width_ratio"])
        )
        / 3.0,
        "diamond": (
            1.0
            - ranks["forehead_width_ratio"]
            + 1.0
            - ranks["upper_face_width_ratio"]
            + 1.0
            - ranks["gonial_width_ratio"]
        )
        / 3.0,
        "oval": (
            1.0
            - abs(ranks["length_width_ratio"] - 0.65)
            + 1.0
            - abs(ranks["gonial_width_ratio"] - 0.45)
            + 1.0
            - abs(ranks["chin_width_ratio"] - 0.45)
        )
        / 3.0,
    }
    support = {
        "oblong": ("length_width_ratio",),
        "round": ("length_width_ratio", "gonial_angle", "chin_roundness"),
        "square": ("gonial_width_ratio", "chin_width_ratio", "gonial_angle"),
        "heart": ("forehead_width_ratio", "gonial_width_ratio", "chin_width_ratio"),
        "diamond": ("forehead_width_ratio", "upper_face_width_ratio", "gonial_width_ratio"),
        "oval": ("length_width_ratio", "gonial_width_ratio", "chin_width_ratio"),
    }
    ordered = sorted(scores, key=lambda label: (-scores[label], label))[:2]
    return [
        {
            "label": label,
            "score": round(scores[label] * 100.0, 1),
            "supporting_measurements": [
                {
                    "name": name,
                    "value": round(float(measurements[name]), 4),
                    "population_percentile": round(ranks[name] * 100.0, 1),
                }
                for name in support[label]
            ],
        }
        for label in ordered
    ]
