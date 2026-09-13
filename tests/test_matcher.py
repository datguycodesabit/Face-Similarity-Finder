from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from face_match.database import Database
from face_match.geometry import (
    DENSE_WEIGHTS,
    descriptor_vector,
    normalize_landmarks,
    shape_measurements,
)
from face_match.matcher import brute_force_oracle, rank_matches, rank_shape_matches

from .helpers import dense_detection, structural_points


def test_unique_identity_aggregation_matches_numpy_oracle(tmp_path: Path) -> None:
    database = Database(tmp_path / "index.sqlite3")
    database.initialize()
    query = normalize_landmarks(structural_points(10))
    records = []
    for identity_index in range(6):
        name = f"Person {identity_index}"
        for photo_index in range(2):
            vector = query + (identity_index + photo_index / 10) * 0.01
            image_id = database.register_image(
                name, f"p{identity_index}_{photo_index}.jpg", "fake-v1"
            )
            database.save_vector(image_id, vector, [[0.5, 0.5]], "fake-v1")
            records.append((name, image_id, vector))
    actual = rank_matches(database, query)
    oracle = brute_force_oracle(query, records)
    assert [(item.identity, item.image_id, item.distance) for item in actual] == oracle
    assert len({item.identity for item in actual}) == 5
    assert [item.distance for item in actual] == sorted(item.distance for item in actual)


def test_v2_weighted_ranking_matches_independent_oracle_and_excludes_ineligible(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "v2.sqlite3")
    database.initialize()
    query = dense_detection(70).normalized
    query_measurements = shape_measurements(query)
    records = []
    for index in range(6):
        detection = dense_detection(80 + index)
        image_id = database.register_image(f"Person {index}", f"p{index}.jpg", "fake-v2")
        database.save_vector(
            image_id,
            detection.normalized,
            detection.overlay,
            "fake-v2",
            detection.measurements,
        )
        records.append((f"Person {index}", image_id, detection.normalized, detection.measurements))
    excluded = dense_detection(70)
    excluded_id = database.register_image("Excluded exact match", "excluded.jpg", "fake-v2")
    database.save_vector(
        excluded_id,
        excluded.normalized,
        excluded.overlay,
        "fake-v2",
        excluded.measurements,
        eligible=False,
        exclusion_reason="pose",
    )

    def oracle_distance(vector: NDArray[np.float64], measurements: dict[str, float]) -> float:
        moving = vector - np.average(vector, axis=0, weights=DENSE_WEIGHTS)
        reference = query - np.average(query, axis=0, weights=DENSE_WEIGHTS)
        left, _, right = np.linalg.svd((moving * DENSE_WEIGHTS[:, None]).T @ reference)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        silhouette = float(
            np.sqrt(
                np.average(
                    np.sum((reference - moving @ rotation) ** 2, axis=1), weights=DENSE_WEIGHTS
                )
            )
        )
        descriptor = float(
            np.sqrt(
                np.mean(
                    (descriptor_vector(query_measurements) - descriptor_vector(measurements)) ** 2
                )
            )
        )
        return 0.7 * descriptor + 0.3 * silhouette

    expected = sorted(
        (
            (name, image_id, oracle_distance(vector, measurements))
            for name, image_id, vector, measurements in records
        ),
        key=lambda item: (item[2], item[0]),
    )[:5]
    actual = rank_shape_matches(database, query, query_measurements)
    assert [(item.identity, item.image_id, item.distance) for item in actual] == expected
    assert all(item.identity != "Excluded exact match" for item in actual)
