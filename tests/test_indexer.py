from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from face_match.config import Settings
from face_match.database import Database
from face_match.detector import Detection
from face_match.geometry import LANDMARK_VERSION, normalize_landmarks
from face_match.indexer import index_dataset

from .helpers import FakeDetector, structural_points


def _settings(tmp_path: Path, dataset: Path) -> Settings:
    return Settings(
        tmp_path, tmp_path / "model.task", tmp_path / "index.sqlite3", dataset, 1024 * 1024, 1000, 8
    )


def _write_image(path: Path, color: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 120), (color, color, color)).save(path, "JPEG")


def test_indexing_is_resumable_idempotent_and_records_invalid(tmp_path: Path) -> None:
    dataset = tmp_path / "lfw"
    _write_image(dataset / "Ada_Lovelace" / "Ada_Lovelace_0001.jpg")
    _write_image(dataset / "Grace_Hopper" / "Grace_Hopper_0001.jpg", 80)
    broken = dataset / "Broken" / "Broken_0001.jpg"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not an image")
    settings = _settings(tmp_path, dataset)
    database = Database(settings.database_path)
    points = structural_points()
    detector = FakeDetector(Detection(normalize_landmarks(points), [[0.5, 0.5]] * len(points)))

    first = index_dataset(dataset, database, detector, settings)
    second = index_dataset(dataset, database, detector, settings)

    assert first == {"discovered": 3, "indexed": 2, "invalid": 1, "skipped": 0}
    assert second == {"discovered": 3, "indexed": 0, "invalid": 0, "skipped": 3}
    assert database.status() == {
        "total": 3,
        "indexed": 2,
        "invalid": 1,
        "pending": 0,
        "eligible": 2,
        "excluded": 0,
        "identities": 2,
    }
    assert database.metadata()["landmark_version"] == LANDMARK_VERSION


def test_changed_model_version_forces_reindex(tmp_path: Path) -> None:
    dataset = tmp_path / "lfw"
    image = dataset / "Person_One" / "Person_One_0001.jpg"
    _write_image(image)
    settings = _settings(tmp_path, dataset)
    database = Database(settings.database_path)
    points = structural_points()
    first_detector = FakeDetector(
        Detection(normalize_landmarks(points), [[0.4, 0.4]] * len(points))
    )
    index_dataset(dataset, database, first_detector, settings)
    second_detector = FakeDetector(
        Detection(normalize_landmarks(points), [[0.4, 0.4]] * len(points)), model_version="fake-v2"
    )
    result = index_dataset(dataset, database, second_detector, settings)
    assert result["indexed"] == 1


def test_changed_landmark_version_forces_reindex(tmp_path: Path) -> None:
    dataset = tmp_path / "lfw"
    image = dataset / "Person_One" / "Person_One_0001.jpg"
    _write_image(image)
    settings = _settings(tmp_path, dataset)
    database = Database(settings.database_path)
    detector = FakeDetector(
        Detection(
            normalize_landmarks(structural_points()),
            [[0.4, 0.4]] * len(structural_points()),
        )
    )
    index_dataset(dataset, database, detector, settings)
    with database.connect() as connection:
        connection.execute("UPDATE images SET landmark_version='obsolete-v0'")
    result = index_dataset(dataset, database, detector, settings)
    assert result["indexed"] == 1


def test_interrupted_index_resumes_without_repeating_completed_work(tmp_path: Path) -> None:
    dataset = tmp_path / "lfw"
    _write_image(dataset / "First_Person" / "First_Person_0001.jpg")
    _write_image(dataset / "Second_Person" / "Second_Person_0001.jpg")
    settings = _settings(tmp_path, dataset)
    database = Database(settings.database_path)
    complete_detector = FakeDetector(
        Detection(
            normalize_landmarks(structural_points()),
            [[0.4, 0.4]] * len(structural_points()),
        )
    )

    class InterruptingDetector:
        model_version = "fake-v1"

        def __init__(self) -> None:
            self.calls = 0

        def detect_one(self, image: Image.Image) -> Detection:
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return complete_detector.detect_one(image)

    with pytest.raises(KeyboardInterrupt):
        index_dataset(dataset, database, InterruptingDetector(), settings)
    assert database.status()["indexed"] == 1
    assert database.metadata()["indexing_status"] == "running"

    resumed = index_dataset(dataset, database, complete_detector, settings)
    assert resumed == {"discovered": 2, "indexed": 1, "invalid": 0, "skipped": 1}
    assert database.metadata()["indexing_status"] == "complete"
