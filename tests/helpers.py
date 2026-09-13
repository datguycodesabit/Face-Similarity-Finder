from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from face_match.detector import Detection
from face_match.errors import FaceCountError
from face_match.geometry import (
    DENSE_LANDMARK_INDICES,
    FACE_OVAL,
    LANDMARK_INDICES,
    LEFT_EYE,
    RIGHT_EYE,
    normalize_dense_landmarks,
    normalize_landmarks,
    shape_measurements,
)


def structural_points(seed: int = 2) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    points = rng.normal(0, 0.25, (len(LANDMARK_INDICES), 3))
    left = len(FACE_OVAL)
    right = left + len(LEFT_EYE)
    points[left:right, :2] += np.array([-0.5, 0.0])
    points[right : right + len(RIGHT_EYE), :2] += np.array([0.5, 0.0])
    return points


@dataclass
class FakeDetector:
    result: Detection
    face_count: int = 1
    model_version: str = "fake-v1"

    def detect_one(self, image: Image.Image) -> Detection:
        del image
        if self.face_count != 1:
            raise FaceCountError(self.face_count)
        return self.result


@dataclass
class SequenceDetector:
    results: list[Detection]
    model_version: str = "fake-v2"

    def __post_init__(self) -> None:
        self.calls = 0

    def detect_one(self, image: Image.Image) -> Detection:
        del image
        result = self.results[self.calls % len(self.results)]
        self.calls += 1
        return result


def dense_points(seed: int = 2) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    points = rng.normal(0, 0.08, (len(DENSE_LANDMARK_INDICES), 3))
    index = {mesh: position for position, mesh in enumerate(DENSE_LANDMARK_INDICES)}
    anchors = {
        10: (0.0, -1.0, 0.0),
        152: (0.0, 1.0, 0.0),
        234: (-0.75, 0.0, 0.0),
        454: (0.75, 0.0, 0.0),
        103: (-0.58, -0.55, 0.0),
        332: (0.58, -0.55, 0.0),
        127: (-0.66, -0.25, 0.0),
        356: (0.66, -0.25, 0.0),
        172: (-0.55, 0.58, 0.0),
        397: (0.55, 0.58, 0.0),
        148: (-0.22, 0.86, 0.0),
        377: (0.22, 0.86, 0.0),
        168: (0.0, -0.25, -0.05),
        2: (0.0, 0.25, -0.12),
    }
    for mesh, value in anchors.items():
        points[index[mesh]] = value
    return points


def dense_detection(seed: int = 2, yaw: float = 0.0, pitch: float = 0.0) -> Detection:
    normalized = normalize_dense_landmarks(dense_points(seed))
    overlay = [[0.5, 0.5] for _ in DENSE_LANDMARK_INDICES]
    return Detection(
        normalized,
        overlay,
        shape_measurements(normalized),
        yaw=yaw,
        pitch=pitch,
        quality=1.0,
    )


def fake_detector(seed: int = 2) -> FakeDetector:
    points = structural_points(seed)
    overlay = [[float((point[0] + 1) / 2), float((point[1] + 1) / 2)] for point in points]
    return FakeDetector(Detection(normalized=normalize_landmarks(points), overlay=overlay))
