from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import LANDMARK_VERSION

SCHEMA_VERSION = "2"


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS subjects (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
                    source_path TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'indexed', 'invalid')),
                    vector BLOB,
                    vector_shape TEXT,
                    overlay_json TEXT,
                    descriptor_json TEXT,
                    yaw REAL NOT NULL DEFAULT 0,
                    pitch REAL NOT NULL DEFAULT 0,
                    roll REAL NOT NULL DEFAULT 0,
                    quality REAL NOT NULL DEFAULT 1,
                    match_eligible INTEGER NOT NULL DEFAULT 1,
                    exclusion_reason TEXT,
                    error TEXT,
                    model_version TEXT NOT NULL,
                    landmark_version TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
                CREATE INDEX IF NOT EXISTS idx_images_subject_status
                    ON images(subject_id, status);
                """
            )
            columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(images)")}
            migrations = {
                "descriptor_json": "TEXT",
                "yaw": "REAL NOT NULL DEFAULT 0",
                "pitch": "REAL NOT NULL DEFAULT 0",
                "roll": "REAL NOT NULL DEFAULT 0",
                "quality": "REAL NOT NULL DEFAULT 1",
                "match_eligible": "INTEGER NOT NULL DEFAULT 1",
                "exclusion_reason": "TEXT",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE images ADD COLUMN {name} {declaration}")
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            connection.execute("PRAGMA optimize")

    def set_metadata(self, key: str, value: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)", (key, value)
            )

    def metadata(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        with self.connect() as connection:
            return {
                row["key"]: row["value"] for row in connection.execute("SELECT * FROM metadata")
            }

    def register_image(self, subject: str, source_path: str, model_version: str) -> int:
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO subjects(name) VALUES(?)", (subject,))
            subject_id = connection.execute(
                "SELECT id FROM subjects WHERE name = ?", (subject,)
            ).fetchone()["id"]
            connection.execute(
                """INSERT OR IGNORE INTO images(
                    subject_id, source_path, status, model_version, landmark_version
                ) VALUES(?, ?, 'pending', ?, ?)""",
                (subject_id, source_path, model_version, LANDMARK_VERSION),
            )
            row = connection.execute(
                "SELECT id FROM images WHERE source_path = ?", (source_path,)
            ).fetchone()
            return int(row["id"])

    def needs_indexing(self, image_id: int, model_version: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status, model_version, landmark_version FROM images WHERE id = ?",
                (image_id,),
            ).fetchone()
        return bool(
            row is None
            or row["status"] == "pending"
            or row["model_version"] != model_version
            or row["landmark_version"] != LANDMARK_VERSION
        )

    def save_vector(
        self,
        image_id: int,
        vector: NDArray[np.float64],
        overlay: list[list[float]],
        model_version: str,
        measurements: dict[str, float] | None = None,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        quality: float = 1.0,
        eligible: bool = True,
        exclusion_reason: str | None = None,
    ) -> None:
        array = np.asarray(vector, dtype=np.float64)
        with self.connect() as connection:
            connection.execute(
                """UPDATE images SET status='indexed', vector=?, vector_shape=?, overlay_json=?,
                descriptor_json=?, yaw=?, pitch=?, roll=?, quality=?, match_eligible=?,
                exclusion_reason=?, error=NULL, model_version=?, landmark_version=?,
                updated_at=CURRENT_TIMESTAMP
                WHERE id=?""",
                (
                    array.tobytes(),
                    json.dumps(array.shape),
                    json.dumps(overlay, separators=(",", ":")),
                    json.dumps(measurements or {}, separators=(",", ":")),
                    yaw,
                    pitch,
                    roll,
                    quality,
                    int(eligible),
                    exclusion_reason,
                    model_version,
                    LANDMARK_VERSION,
                    image_id,
                ),
            )

    def save_invalid(self, image_id: int, error: str, model_version: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE images SET status='invalid', vector=NULL, vector_shape=NULL,
                overlay_json=NULL, descriptor_json=NULL, match_eligible=0,
                exclusion_reason=NULL, error=?, model_version=?, landmark_version=?,
                updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                (error[:500], model_version, LANDMARK_VERSION, image_id),
            )

    def status(self) -> dict[str, Any]:
        counts = {
            "total": 0,
            "indexed": 0,
            "invalid": 0,
            "pending": 0,
            "eligible": 0,
            "excluded": 0,
            "identities": 0,
        }
        if not self.path.exists():
            return counts
        with self.connect() as connection:
            for row in connection.execute(
                "SELECT status, COUNT(*) count FROM images GROUP BY status"
            ):
                counts[str(row["status"])] = int(row["count"])
                counts["total"] += int(row["count"])
            for row in connection.execute(
                """SELECT match_eligible, COUNT(*) count FROM images
                WHERE status='indexed' GROUP BY match_eligible"""
            ):
                key = "eligible" if int(row["match_eligible"]) else "excluded"
                counts[key] = int(row["count"])
            row = connection.execute(
                """SELECT COUNT(DISTINCT subject_id) count FROM images
                WHERE status='indexed' AND match_eligible=1"""
            ).fetchone()
            counts["identities"] = int(row["count"])
        return counts

    def indexed_rows(self, *, eligible_only: bool = True) -> list[sqlite3.Row]:
        if not self.path.exists():
            return []
        with self.connect() as connection:
            eligibility = " AND images.match_eligible=1" if eligible_only else ""
            return list(
                connection.execute(
                    """SELECT images.id, subjects.name, images.source_path, images.vector,
                images.vector_shape, images.overlay_json, images.descriptor_json,
                images.yaw, images.pitch, images.roll, images.quality,
                images.match_eligible, images.exclusion_reason FROM images
                JOIN subjects ON subjects.id = images.subject_id
                WHERE images.status='indexed' AND images.landmark_version=?"""
                    + eligibility
                    + " ORDER BY images.id",
                    (LANDMARK_VERSION,),
                )
            )

    @staticmethod
    def decode_vector(row: sqlite3.Row) -> NDArray[np.float64]:
        shape = tuple(json.loads(row["vector_shape"]))
        return np.frombuffer(row["vector"], dtype=np.float64).reshape(shape)
