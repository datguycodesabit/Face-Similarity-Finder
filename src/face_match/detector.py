from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .errors import FaceCountError
from .geometry import (
    DENSE_LANDMARK_INDICES,
    normalize_dense_landmarks,
    normalize_landmarks,
    select_dense_landmarks,
    select_landmarks,
    shape_measurements,
)
from .shape_v3 import normalize_v3_landmarks, shape_descriptors


@dataclass(frozen=True)
class Detection:
    normalized: NDArray[np.float64]
    overlay: list[list[float]]
    measurements: dict[str, float] = field(default_factory=dict)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    quality: float = 1.0
    eligible: bool = True
    exclusion_reason: str | None = None
    legacy_normalized: NDArray[np.float64] | None = None
    v3_normalized: NDArray[np.float64] | None = None
    v3_measurements: dict[str, float] = field(default_factory=dict)


class Detector(Protocol):
    model_version: str

    def detect_one(self, image: Image.Image) -> Detection: ...


class MediaPipeDetector:
    model_version = "mediapipe-face-landmarker-v2-v3-standard-crop"

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model missing at {model_path}. Run: uv run face-match setup-model"
            )
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.model_version = f"mediapipe-face-landmarker-v2-v3-crop-sha256-{digest}"
        import mediapipe as mp  # type: ignore[import-untyped]

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _detect(self, image: Image.Image) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        result = self._landmarker.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        )
        count = len(result.face_landmarks)
        if count != 1:
            raise FaceCountError(count)
        full = np.asarray([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float64)
        matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
        return full, matrix

    @staticmethod
    def _standardized_crop(image: Image.Image, full: NDArray[np.float64]) -> Image.Image:
        pixel_x = full[:, 0] * image.width
        pixel_y = full[:, 1] * image.height
        width = float(np.ptp(pixel_x))
        height = float(np.ptp(pixel_y))
        side = max(width, height) * 1.55
        center_x = float((pixel_x.min() + pixel_x.max()) / 2.0)
        center_y = float((pixel_y.min() + pixel_y.max()) / 2.0)
        box = (
            center_x - side / 2.0,
            center_y - side / 2.0,
            center_x + side / 2.0,
            center_y + side / 2.0,
        )
        return image.transform(
            (512, 512),
            Image.Transform.EXTENT,
            box,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )

    def detect_one(self, image: Image.Image) -> Detection:
        full, matrix = self._detect(image)
        rotation = matrix[:3, :3]
        yaw = float(np.degrees(np.arctan2(rotation[0, 2], rotation[2, 2])))
        pitch = float(
            np.degrees(
                np.arctan2(
                    -rotation[1, 2],
                    np.sqrt(rotation[1, 0] ** 2 + rotation[1, 1] ** 2),
                )
            )
        )
        roll = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
        selected = select_dense_landmarks(full)
        normalized = normalize_dense_landmarks(selected, rotation)
        standardized = self._standardized_crop(image, full)
        v3_full, v3_matrix = self._detect(standardized)
        v3_normalized = normalize_v3_landmarks(
            v3_full, standardized.width, standardized.height, v3_matrix[:3, :3]
        )
        legacy_normalized = normalize_landmarks(select_landmarks(full))
        in_frame = np.mean(
            (full[:, 0] >= 0.0) & (full[:, 0] <= 1.0) & (full[:, 1] >= 0.0) & (full[:, 1] <= 1.0)
        )
        width = float(full[:, 0].max() - full[:, 0].min())
        height = float(full[:, 1].max() - full[:, 1].min())
        quality = min(float(in_frame), min(1.0, min(width, height) / 0.28))
        reasons = []
        if abs(yaw) > 15.0:
            reasons.append("reference pose is not frontal")
        if abs(pitch) > 12.0:
            reasons.append("reference pitch is too steep")
        if quality < 0.85:
            reasons.append("face crop or landmark coverage is too weak")
        overlay = [
            [round(float(full[index, 0]), 6), round(float(full[index, 1]), 6)]
            for index in DENSE_LANDMARK_INDICES
        ]
        return Detection(
            normalized=normalized,
            overlay=overlay,
            measurements=shape_measurements(normalized),
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
            quality=round(quality, 4),
            eligible=not reasons,
            exclusion_reason="; ".join(reasons) or None,
            legacy_normalized=legacy_normalized,
            v3_normalized=v3_normalized,
            v3_measurements=shape_descriptors(v3_normalized),
        )

    def close(self) -> None:
        self._landmarker.close()
