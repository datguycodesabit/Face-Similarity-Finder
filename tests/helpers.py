from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

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
from face_match.mica import MicaViewResult
from face_match.shape_v3 import COMPONENT_NAMES, HybridCalibration, shape_descriptors
from face_match.v3_database import V3Database


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
        v3_normalized=normalized,
        v3_measurements=shape_descriptors(normalized),
    )


def fake_detector(seed: int = 2) -> FakeDetector:
    points = structural_points(seed)
    overlay = [[float((point[0] + 1) / 2), float((point[1] + 1) / 2)] for point in points]
    return FakeDetector(Detection(normalized=normalize_landmarks(points), overlay=overlay))


def mica_result(seed: int) -> MicaViewResult:
    rng = np.random.default_rng(seed)
    return MicaViewResult(
        shape_code=rng.normal(size=8),
        regions={
            name: rng.normal(size=(12, 3)) * 0.01 + index * 0.04
            for index, name in enumerate(COMPONENT_NAMES)
        },
        reprojection_error=0.02,
    )


@dataclass
class FakeMicaEngine:
    engine_version: str = "mica-test-v1"

    def __post_init__(self) -> None:
        self.calls = 0

    def analyze(
        self,
        images: Sequence[bytes],
        *,
        validate_identity: bool = False,
        anchor_index: int = 0,
    ) -> list[MicaViewResult]:
        assert len(images) == 3
        assert anchor_index in range(3)
        del validate_identity
        seed = 900 + self.calls * 10
        self.calls += 1
        return [mica_result(seed + index) for index in range(3)]

    def close(self) -> None:
        return None


def populate_v3_database(database: V3Database, dataset: Path, identities: int = 6) -> None:
    import hashlib

    from face_match.commons_pack import parse_manifest

    raw_identities = []
    payloads: dict[tuple[int, int], bytes] = {}
    for identity in range(identities):
        images = []
        for view in range(3):
            stream = BytesIO()
            Image.new(
                "RGB",
                (96, 96),
                (70 + identity * 15, 90 + view * 20, 120 + identity * 5),
            ).save(stream, "PNG")
            payload = stream.getvalue()
            payloads[(identity, view)] = payload
            images.append(
                {
                    "page_id": identity * 3 + view + 1,
                    "revision_id": identity * 3 + view + 100,
                    "original_url": (f"https://upload.wikimedia.org/fixture/{identity}-{view}.png"),
                    "author": f"Fixture author {identity}",
                    "license_id": "CC-BY-4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "selected_display": view == 0,
                }
            )
        raw_identities.append(
            {"name": f"Reference {identity}", "wikidata_id": f"Q{identity + 1}", "images": images}
        )
    manifest = parse_manifest(
        {"gallery_version": "commons-test-v1", "identities": raw_identities},
        expected_identities=identities,
    )
    database.register_manifest(manifest, dataset)
    with database.connect() as connection:
        identity_rows = list(connection.execute("SELECT id FROM identities ORDER BY id"))
        image_rows = list(connection.execute("SELECT * FROM images ORDER BY id"))
    for row in image_rows:
        image_id = int(row["id"])
        identity_index = (image_id - 1) // 3
        view_index = (image_id - 1) % 3
        local_path = Path(str(row["local_path"]))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(payloads[(identity_index, view_index)])
        database.save_image_shape(
            image_id,
            str(row["sha256"]),
            "mica-test-v1",
            mica_result(1000 + identity_index * 10 + view_index),
            shape_descriptors(dense_points(1000 + identity_index * 10 + view_index)),
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            quality=1.0,
        )
    for row in identity_rows:
        database.build_prototype(int(row["id"]))
    prototypes = database.prototype_rows()
    descriptors = [__import__("json").loads(str(row["descriptor_json"])) for row in prototypes]
    regions = [database.prototype_regions(int(row["identity_id"])) for row in prototypes]
    database.save_hybrid_calibration(
        "commons-test-calibration-v1", HybridCalibration.fit(descriptors, regions)
    )
    database.set_metadata("algorithm_version", "shape-v3-isotropic-mica-calibrated")
    database.set_metadata("mica_engine_version", "mica-test-v1")
    database.set_metadata("indexing_status", "complete")
