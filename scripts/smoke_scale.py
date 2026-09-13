from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from face_match.database import Database
from face_match.geometry import LANDMARK_INDICES, LANDMARK_VERSION
from face_match.matcher import brute_force_oracle, rank_matches

IMAGE_COUNT = 13_233
IDENTITY_COUNT = 5_749


def main() -> None:
    """Exercise SQLite loading and exact ranking at LFW's published scale."""
    shape = (len(LANDMARK_INDICES), 3)
    query = np.zeros(shape, dtype=np.float64)
    records: list[tuple[str, int, np.ndarray[tuple[int, ...], np.dtype[np.float64]]]] = []

    with tempfile.TemporaryDirectory(prefix="face-match-scale-") as temporary:
        database = Database(Path(temporary) / "scale.sqlite3")
        database.initialize()
        with database.connect() as connection:
            connection.executemany(
                "INSERT INTO subjects(id, name) VALUES(?, ?)",
                (
                    (index, f"Synthetic Person {index:04d}")
                    for index in range(1, IDENTITY_COUNT + 1)
                ),
            )
            image_rows = []
            for image_index in range(1, IMAGE_COUNT + 1):
                subject_id = ((image_index - 1) % IDENTITY_COUNT) + 1
                value = subject_id / IDENTITY_COUNT + image_index / IMAGE_COUNT * 1e-6
                vector = np.full(shape, value, dtype=np.float64)
                name = f"Synthetic Person {subject_id:04d}"
                records.append((name, image_index, vector))
                image_rows.append(
                    (
                        image_index,
                        subject_id,
                        f"synthetic/{image_index:05d}.jpg",
                        vector.tobytes(),
                        json.dumps(shape),
                        "[]",
                        LANDMARK_VERSION,
                    )
                )
            connection.executemany(
                """INSERT INTO images(
                    id, subject_id, source_path, status, vector, vector_shape, overlay_json,
                    model_version, landmark_version
                ) VALUES(?, ?, ?, 'indexed', ?, ?, ?, 'synthetic-scale-v1', ?)""",
                image_rows,
            )

        started = time.perf_counter()
        actual = rank_matches(database, query)
        elapsed = time.perf_counter() - started
        oracle = brute_force_oracle(query, records)
        actual_tuples = [(item.identity, item.image_id, item.distance) for item in actual]
        if actual_tuples != oracle:
            raise SystemExit("Scale ranking differs from the independent NumPy oracle.")
        if len({item.identity for item in actual}) != 5:
            raise SystemExit("Scale ranking did not return five unique identities.")
        print(
            f"PASS: {IMAGE_COUNT:,} images, {IDENTITY_COUNT:,} identities, "
            f"five exact oracle matches in {elapsed:.3f}s"
        )


if __name__ == "__main__":
    main()
