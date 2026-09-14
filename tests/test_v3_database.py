from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from face_match.commons_pack import CommonsManifest, parse_manifest
from face_match.mica import MicaViewResult
from face_match.shape_v3 import COMPONENT_NAMES, RobustCalibration, shape_descriptors
from face_match.v3_database import V3Database

from .helpers import dense_points


def _manifest() -> CommonsManifest:
    images = []
    for index in range(3):
        images.append(
            {
                "page_id": index + 1,
                "revision_id": index + 11,
                "original_url": f"https://upload.wikimedia.org/example/{index}.jpg",
                "author": "Photographer",
                "license_id": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "sha256": hashlib.sha256(f"image-{index}".encode()).hexdigest(),
                "selected_display": index == 0,
            }
        )
    return parse_manifest(
        {
            "gallery_version": "test-v1",
            "identities": [{"name": "Test Person", "wikidata_id": "Q42", "images": images}],
        },
        expected_identities=1,
    )


def _mica_result(value: float) -> MicaViewResult:
    return MicaViewResult(
        np.asarray([value, value / 2], dtype=np.float64),
        {
            component: np.full((4, 3), value + index, dtype=np.float64)
            for index, component in enumerate(COMPONENT_NAMES)
        },
        0.05,
    )


def test_v3_database_builds_one_prototype_and_preserves_attribution(tmp_path: Path) -> None:
    database = V3Database(tmp_path / "v3.sqlite3")
    manifest = _manifest()
    database.register_manifest(manifest, tmp_path / "commons")
    measurements = [shape_descriptors(dense_points(seed)) for seed in range(3)]
    with database.connect() as connection:
        rows = list(connection.execute("SELECT id, sha256 FROM images ORDER BY id"))
        identity_id = int(connection.execute("SELECT id FROM identities").fetchone()["id"])
    for index, row in enumerate(rows):
        database.save_image_shape(
            int(row["id"]),
            str(row["sha256"]),
            "mica-test-v1",
            _mica_result(1.0 + index * 0.01),
            measurements[index],
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            quality=1.0,
        )
    database.build_prototype(identity_id)
    result = database.prototype_rows()
    assert len(result) == 1
    assert result[0]["name"] == "Test Person"
    assert result[0]["author"] == "Photographer"
    assert result[0]["license_id"] == "CC-BY-4.0"
    assert database.status() == {
        "identities": 1,
        "images": 3,
        "indexed": 3,
        "invalid": 0,
        "prototypes": 1,
    }


def test_v3_database_resumes_by_hash_engine_and_algorithm(tmp_path: Path) -> None:
    database = V3Database(tmp_path / "v3.sqlite3")
    manifest = _manifest()
    database.register_manifest(manifest, tmp_path / "commons")
    with database.connect() as connection:
        row = connection.execute("SELECT id, sha256 FROM images ORDER BY id LIMIT 1").fetchone()
    image_id, digest = int(row["id"]), str(row["sha256"])
    assert database.needs_indexing(image_id, digest, "mica-test-v1")
    database.save_image_shape(
        image_id,
        digest,
        "mica-test-v1",
        _mica_result(1.0),
        shape_descriptors(dense_points(5)),
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        quality=1.0,
    )
    assert not database.needs_indexing(image_id, digest, "mica-test-v1")
    assert database.needs_indexing(image_id, "0" * 64, "mica-test-v1")
    assert database.needs_indexing(image_id, digest, "mica-test-v2")


def test_v3_database_prunes_images_removed_by_a_new_manifest(tmp_path: Path) -> None:
    database = V3Database(tmp_path / "v3.sqlite3")
    original = _manifest()
    database.register_manifest(original, tmp_path / "commons")
    raw_images = []
    for index, image in enumerate(original.identities[0].images):
        raw_images.append(
            {
                "page_id": image.page_id if index < 2 else 99,
                "revision_id": image.revision_id if index < 2 else 199,
                "original_url": (
                    image.original_url
                    if index < 2
                    else "https://upload.wikimedia.org/example/replacement.jpg"
                ),
                "author": image.author,
                "license_id": image.license_id,
                "license_url": image.license_url,
                "sha256": (
                    image.sha256 if index < 2 else hashlib.sha256(b"replacement-image").hexdigest()
                ),
                "selected_display": index == 0,
            }
        )
    replacement = parse_manifest(
        {
            "gallery_version": "test-v2",
            "identities": [{"name": "Test Person", "wikidata_id": "Q42", "images": raw_images}],
        },
        expected_identities=1,
    )
    database.register_manifest(replacement, tmp_path / "commons")
    with database.connect() as connection:
        rows = list(connection.execute("SELECT commons_page_id FROM images ORDER BY id"))
    assert [int(row["commons_page_id"]) for row in rows] == [1, 2, 99]


def test_v3_database_round_trips_active_calibration(tmp_path: Path) -> None:
    database = V3Database(tmp_path / "v3.sqlite3")
    database.initialize()
    samples = [shape_descriptors(dense_points(seed)) for seed in range(20, 40)]
    expected = RobustCalibration.fit(samples)
    database.save_calibration("calibration-test-v1", expected)
    actual = database.active_calibration()
    assert actual is not None
    assert np.allclose(actual.median, expected.median)
    assert np.allclose(actual.scale, expected.scale)
