from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from face_match.database import Database
from face_match.geometry import (
    DENSE_INDEX,
    LANDMARK_INDICES,
    aggregate_views,
    combined_shape_distance,
    normalize_dense_landmarks,
    normalize_landmarks,
    shape_measurements,
    shape_memberships,
)
from face_match.recommendations import recommend_haircuts

from .helpers import dense_points


def test_dense_normalization_removes_full_3d_pose_translation_and_scale() -> None:
    points = dense_points(11)
    x, y, z = np.radians([18.0, -24.0, 9.0])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    rotation = rz @ ry @ rx
    transformed = 2.7 * (points @ rotation.T) + np.array([4.0, -3.0, 1.5])
    assert np.allclose(
        normalize_dense_landmarks(points),
        normalize_dense_landmarks(transformed, rotation),
        atol=1e-10,
    )


def test_v2_is_more_pose_stable_than_legacy_roll_only_normalization() -> None:
    points = dense_points(19)
    yaw = np.radians(25.0)
    rotation = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    transformed = points @ rotation.T + np.array([1.0, -2.0, 0.5])
    legacy_rows = [DENSE_INDEX[index] for index in LANDMARK_INDICES]
    legacy_error = np.sqrt(
        np.mean(
            np.sum(
                (
                    normalize_landmarks(points[legacy_rows])
                    - normalize_landmarks(transformed[legacy_rows])
                )
                ** 2,
                axis=1,
            )
        )
    )
    v2_error = np.sqrt(
        np.mean(
            np.sum(
                (
                    normalize_dense_landmarks(points)
                    - normalize_dense_landmarks(transformed, rotation)
                )
                ** 2,
                axis=1,
            )
        )
    )
    assert v2_error < 1e-10
    assert v2_error < legacy_error


def test_weighted_distance_and_three_view_median_are_deterministic() -> None:
    first = normalize_dense_landmarks(dense_points(3))
    second = normalize_dense_landmarks(dense_points(4))
    measurements = shape_measurements(first)
    total, silhouette, proportions = combined_shape_distance(
        first, measurements, second, shape_measurements(second)
    )
    assert total == 0.7 * proportions + 0.3 * silhouette
    aggregate, agreement = aggregate_views([first, first + 0.01, first - 0.01])
    _, identical_agreement = aggregate_views([first, first, first])
    _, divergent_agreement = aggregate_views(
        [first, second, normalize_dense_landmarks(dense_points(5))]
    )
    assert np.allclose(aggregate, first)
    assert 0.0 <= agreement <= 100.0
    assert identical_agreement == 100.0
    assert divergent_agreement < identical_agreement


def test_shape_descriptors_match_independent_geometric_formulas() -> None:
    points = normalize_dense_landmarks(dense_points(23))
    measurements = shape_measurements(points)

    def distance(first: int, second: int) -> float:
        return float(np.linalg.norm(points[DENSE_INDEX[first]] - points[DENSE_INDEX[second]]))

    def angle(first: int, vertex: int, third: int) -> float:
        a = points[DENSE_INDEX[first]] - points[DENSE_INDEX[vertex]]
        b = points[DENSE_INDEX[third]] - points[DENSE_INDEX[vertex]]
        return float(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) / np.pi)

    length, cheek = distance(10, 152), distance(234, 454)
    jaw = distance(172, 397)
    expected = {
        "length_width_ratio": length / cheek,
        "forehead_cheek_ratio": distance(103, 332) / cheek,
        "temple_cheek_ratio": distance(127, 356) / cheek,
        "jaw_cheek_ratio": jaw / cheek,
        "jaw_taper": 1.0 - jaw / cheek,
        "chin_cheek_ratio": distance(148, 377) / cheek,
        "jaw_angularity": 1.0 - (angle(234, 172, 152) + angle(454, 397, 152)) / 2.0,
        "upper_third": distance(10, 168) / length,
        "middle_third": distance(168, 2) / length,
        "lower_third": distance(2, 152) / length,
    }
    assert measurements.keys() == expected.keys()
    assert all(np.isclose(measurements[name], value) for name, value in expected.items())


@pytest.mark.parametrize(
    ("expected", "updates"),
    [
        (
            "round",
            {
                "length_width_ratio": 1.10,
                "forehead_cheek_ratio": 0.98,
                "jaw_cheek_ratio": 0.98,
                "jaw_angularity": 0.20,
            },
        ),
        (
            "square",
            {
                "length_width_ratio": 1.15,
                "forehead_cheek_ratio": 1.0,
                "jaw_cheek_ratio": 0.92,
                "jaw_angularity": 0.55,
            },
        ),
        (
            "oblong",
            {
                "length_width_ratio": 1.50,
                "forehead_cheek_ratio": 1.0,
                "jaw_cheek_ratio": 0.98,
                "jaw_angularity": 0.35,
            },
        ),
        (
            "heart",
            {
                "length_width_ratio": 1.30,
                "forehead_cheek_ratio": 1.10,
                "jaw_cheek_ratio": 0.80,
                "jaw_angularity": 0.35,
            },
        ),
        (
            "triangle",
            {
                "length_width_ratio": 1.30,
                "forehead_cheek_ratio": 0.85,
                "jaw_cheek_ratio": 1.10,
                "jaw_angularity": 0.35,
            },
        ),
        (
            "diamond",
            {
                "length_width_ratio": 1.40,
                "forehead_cheek_ratio": 0.90,
                "jaw_cheek_ratio": 0.90,
                "jaw_angularity": 0.35,
            },
        ),
        (
            "oval",
            {
                "length_width_ratio": 1.32,
                "forehead_cheek_ratio": 0.94,
                "jaw_cheek_ratio": 0.88,
                "jaw_angularity": 0.34,
            },
        ),
    ],
)
def test_shape_membership_archetype_boundaries(expected: str, updates: dict[str, float]) -> None:
    measurements = shape_measurements(normalize_dense_landmarks(dense_points(8)))
    measurements.update(updates)
    assert shape_memberships(measurements)[0][0] == expected


def test_shape_labels_and_haircut_preferences_are_deterministic() -> None:
    measurements = shape_measurements(normalize_dense_landmarks(dense_points(8)))
    measurements.update(
        length_width_ratio=1.52,
        forehead_cheek_ratio=0.94,
        jaw_cheek_ratio=0.86,
        jaw_angularity=0.30,
    )
    memberships = shape_memberships(measurements)
    assert memberships[0][0] == "oblong"
    recommendations = recommend_haircuts(
        memberships,
        {"length": "medium", "texture": "wavy", "effort": "moderate", "goal": "add-width"},
    )
    assert len(recommendations) == 3
    assert recommendations == recommend_haircuts(
        memberships,
        {"length": "medium", "texture": "wavy", "effort": "moderate", "goal": "add-width"},
    )
    assert {item["name"] for item in recommendations} <= {
        "Layered bob or lob",
        "Side-parted layers",
        "Soft shag",
        "Curtain fringe with layers",
        "Rounded layered shape",
    }


def test_schema_v1_database_migrates_without_losing_images(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE subjects (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
        CREATE TABLE images (
            id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL, source_path TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL, vector BLOB, vector_shape TEXT, overlay_json TEXT, error TEXT,
            model_version TEXT NOT NULL, landmark_version TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        INSERT INTO subjects(id, name) VALUES(1, 'Person');
        INSERT INTO images VALUES(1, 1, 'person.jpg', 'pending', NULL, NULL, NULL, NULL,
            'fake-v1', 'old', CURRENT_TIMESTAMP);
        """
    )
    connection.commit()
    connection.close()

    database = Database(path)
    database.initialize()
    with database.connect() as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(images)")}
        count = migrated.execute("SELECT COUNT(*) count FROM images").fetchone()["count"]
    assert {"descriptor_json", "yaw", "pitch", "quality", "match_eligible"} <= columns
    assert count == 1
