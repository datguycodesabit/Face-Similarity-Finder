from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .commons_pack import CommonsManifest
from .mica import MicaViewResult, aggregate_mica_views
from .shape_v3 import (
    COMPONENT_NAMES,
    DESCRIPTOR_NAMES,
    V3_ALGORITHM_VERSION,
    HybridCalibration,
    RobustCalibration,
)

FloatArray = NDArray[np.float64]
V3_SCHEMA_VERSION = "3"


def _encode_array(array: FloatArray) -> tuple[bytes, str]:
    value = np.asarray(array, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("V3 vectors must be finite")
    return value.tobytes(), json.dumps(value.shape)


def _decode_array(blob: bytes, shape: str) -> FloatArray:
    dimensions = tuple(int(value) for value in json.loads(shape))
    return np.frombuffer(blob, dtype=np.float64).reshape(dimensions)


class V3Database:
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
                CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY,
                    wikidata_id TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
                    commons_page_id INTEGER NOT NULL UNIQUE,
                    commons_revision_id INTEGER NOT NULL,
                    original_url TEXT NOT NULL,
                    author TEXT NOT NULL,
                    license_id TEXT NOT NULL,
                    license_url TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    local_path TEXT NOT NULL,
                    selected_display INTEGER NOT NULL CHECK(selected_display IN (0, 1)),
                    status TEXT NOT NULL CHECK(status IN ('pending', 'indexed', 'invalid')),
                    error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS image_shapes (
                    image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
                    image_sha256 TEXT NOT NULL,
                    engine_version TEXT NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    shape_code BLOB NOT NULL,
                    shape_code_shape TEXT NOT NULL,
                    descriptor_json TEXT NOT NULL,
                    yaw REAL NOT NULL,
                    pitch REAL NOT NULL,
                    roll REAL NOT NULL,
                    quality REAL NOT NULL,
                    reprojection_error REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS image_regions (
                    image_id INTEGER NOT NULL REFERENCES image_shapes(image_id) ON DELETE CASCADE,
                    component TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    vector_shape TEXT NOT NULL,
                    PRIMARY KEY(image_id, component)
                );
                CREATE TABLE IF NOT EXISTS prototypes (
                    identity_id INTEGER PRIMARY KEY REFERENCES identities(id) ON DELETE CASCADE,
                    shape_code BLOB NOT NULL,
                    shape_code_shape TEXT NOT NULL,
                    descriptor_json TEXT NOT NULL,
                    source_count INTEGER NOT NULL,
                    rejected_image_id INTEGER REFERENCES images(id),
                    engine_version TEXT NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS prototype_regions (
                    identity_id INTEGER NOT NULL
                        REFERENCES prototypes(identity_id) ON DELETE CASCADE,
                    component TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    vector_shape TEXT NOT NULL,
                    PRIMARY KEY(identity_id, component)
                );
                CREATE TABLE IF NOT EXISTS calibrations (
                    version TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    active INTEGER NOT NULL CHECK(active IN (0, 1)),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_v3_images_identity_status
                    ON images(identity_id, status);
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (V3_SCHEMA_VERSION,),
            )
            connection.execute("PRAGMA optimize")

    def set_metadata(self, key: str, value: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)", (key, value)
            )

    def metadata(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        with self.connect() as connection:
            return {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }

    def register_manifest(self, manifest: CommonsManifest, dataset_root: Path) -> None:
        self.initialize()
        with self.connect() as connection:
            retained_qids = [identity.wikidata_id for identity in manifest.identities]
            if retained_qids:
                placeholders = ",".join("?" for _ in retained_qids)
                connection.execute(
                    f"DELETE FROM identities WHERE wikidata_id NOT IN ({placeholders})",
                    retained_qids,
                )
            else:
                connection.execute("DELETE FROM identities")
            for identity in manifest.identities:
                connection.execute(
                    """INSERT INTO identities(wikidata_id, name) VALUES(?, ?)
                    ON CONFLICT(wikidata_id) DO UPDATE SET name=excluded.name""",
                    (identity.wikidata_id, identity.name),
                )
                identity_id = int(
                    connection.execute(
                        "SELECT id FROM identities WHERE wikidata_id=?", (identity.wikidata_id,)
                    ).fetchone()["id"]
                )
                retained_pages = [image.page_id for image in identity.images]
                page_placeholders = ",".join("?" for _ in retained_pages)
                connection.execute(
                    f"DELETE FROM images WHERE identity_id=? "
                    f"AND commons_page_id NOT IN ({page_placeholders})",
                    (identity_id, *retained_pages),
                )
                for image in identity.images:
                    local_path = str(
                        (dataset_root / identity.wikidata_id / image.local_name).resolve()
                    )
                    existing = connection.execute(
                        "SELECT commons_revision_id, sha256 FROM images WHERE commons_page_id=?",
                        (image.page_id,),
                    ).fetchone()
                    changed = bool(
                        existing
                        and (
                            int(existing["commons_revision_id"]) != image.revision_id
                            or str(existing["sha256"]) != image.sha256
                        )
                    )
                    connection.execute(
                        """INSERT INTO images(
                            identity_id, commons_page_id, commons_revision_id, original_url,
                            author, license_id, license_url, sha256, local_path,
                            selected_display, status
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                        ON CONFLICT(commons_page_id) DO UPDATE SET
                            identity_id=excluded.identity_id,
                            commons_revision_id=excluded.commons_revision_id,
                            original_url=excluded.original_url,
                            author=excluded.author,
                            license_id=excluded.license_id,
                            license_url=excluded.license_url,
                            sha256=excluded.sha256,
                            local_path=excluded.local_path,
                            selected_display=excluded.selected_display,
                            status=CASE WHEN ? THEN 'pending' ELSE images.status END,
                            error=CASE WHEN ? THEN NULL ELSE images.error END,
                            updated_at=CURRENT_TIMESTAMP""",
                        (
                            identity_id,
                            image.page_id,
                            image.revision_id,
                            image.original_url,
                            image.author,
                            image.license_id,
                            image.license_url,
                            image.sha256,
                            local_path,
                            int(image.selected_display),
                            changed,
                            changed,
                        ),
                    )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('gallery_version', ?)",
                (manifest.gallery_version,),
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('dataset_root', ?)",
                (str(dataset_root.resolve()),),
            )

    def needs_indexing(self, image_id: int, image_sha256: str, engine_version: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT images.status, image_shapes.image_sha256, image_shapes.engine_version,
                image_shapes.algorithm_version FROM images LEFT JOIN image_shapes
                ON image_shapes.image_id=images.id WHERE images.id=?""",
                (image_id,),
            ).fetchone()
        return bool(
            row is None
            or row["status"] != "indexed"
            or row["image_sha256"] != image_sha256
            or row["engine_version"] != engine_version
            or row["algorithm_version"] != V3_ALGORITHM_VERSION
        )

    def save_image_shape(
        self,
        image_id: int,
        image_sha256: str,
        engine_version: str,
        result: MicaViewResult,
        descriptors: Mapping[str, float],
        *,
        yaw: float,
        pitch: float,
        roll: float,
        quality: float,
    ) -> None:
        if tuple(descriptors) != DESCRIPTOR_NAMES:
            raise ValueError("V3 descriptor contract is incomplete or out of order")
        code, code_shape = _encode_array(result.shape_code)
        with self.connect() as connection:
            identity = connection.execute(
                "SELECT identity_id FROM images WHERE id=?", (image_id,)
            ).fetchone()
            if identity is None:
                raise ValueError(f"unknown V3 image id {image_id}")
            connection.execute(
                "DELETE FROM prototypes WHERE identity_id=?", (int(identity["identity_id"]),)
            )
            connection.execute(
                """INSERT OR REPLACE INTO image_shapes(
                    image_id, image_sha256, engine_version, algorithm_version,
                    shape_code, shape_code_shape, descriptor_json, yaw, pitch, roll,
                    quality, reprojection_error
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    image_id,
                    image_sha256,
                    engine_version,
                    V3_ALGORITHM_VERSION,
                    code,
                    code_shape,
                    json.dumps(descriptors, separators=(",", ":")),
                    yaw,
                    pitch,
                    roll,
                    quality,
                    result.reprojection_error,
                ),
            )
            connection.execute("DELETE FROM image_regions WHERE image_id=?", (image_id,))
            for component in COMPONENT_NAMES:
                vector, shape = _encode_array(result.regions[component])
                connection.execute(
                    "INSERT INTO image_regions(image_id, component, vector, vector_shape) "
                    "VALUES(?, ?, ?, ?)",
                    (image_id, component, vector, shape),
                )
            connection.execute(
                "UPDATE images SET status='indexed', error=NULL, updated_at=CURRENT_TIMESTAMP "
                "WHERE id=?",
                (image_id,),
            )

    def save_invalid(self, image_id: int, error: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM image_shapes WHERE image_id=?", (image_id,))
            connection.execute(
                """UPDATE images SET status='invalid', error=?,
                updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                (error[:500], image_id),
            )

    def _image_result(self, connection: sqlite3.Connection, image_id: int) -> MicaViewResult:
        row = connection.execute(
            "SELECT * FROM image_shapes WHERE image_id=?", (image_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"V3 image {image_id} is not indexed")
        region_rows = connection.execute(
            "SELECT component, vector, vector_shape FROM image_regions WHERE image_id=?",
            (image_id,),
        )
        regions = {
            str(region["component"]): _decode_array(region["vector"], region["vector_shape"])
            for region in region_rows
        }
        if set(regions) != set(COMPONENT_NAMES):
            raise ValueError(f"V3 image {image_id} has incomplete FLAME regions")
        return MicaViewResult(
            _decode_array(row["shape_code"], row["shape_code_shape"]),
            regions,
            float(row["reprojection_error"]),
        )

    def build_prototype(self, identity_id: int) -> None:
        with self.connect() as connection:
            rows = list(
                connection.execute(
                    """SELECT images.id, image_shapes.descriptor_json,
                    image_shapes.engine_version FROM images JOIN image_shapes
                    ON image_shapes.image_id=images.id
                    WHERE images.identity_id=? AND images.status='indexed'
                    AND image_shapes.algorithm_version=? ORDER BY images.id""",
                    (identity_id, V3_ALGORITHM_VERSION),
                )
            )
            if len(rows) != 3:
                raise ValueError(
                    f"identity {identity_id} requires exactly three indexed Commons images"
                )
            engine_versions = {str(row["engine_version"]) for row in rows}
            if len(engine_versions) != 1:
                raise ValueError(f"identity {identity_id} mixes MICA engine versions")
            results = [self._image_result(connection, int(row["id"])) for row in rows]
            aggregate = aggregate_mica_views(results)
            descriptor_rows = [
                json.loads(rows[index]["descriptor_json"]) for index in aggregate.accepted_views
            ]
            descriptors = {
                name: float(np.median([values[name] for values in descriptor_rows]))
                for name in DESCRIPTOR_NAMES
            }
            code, code_shape = _encode_array(aggregate.shape_code)
            rejected_image_id = (
                int(rows[aggregate.rejected_view]["id"])
                if aggregate.rejected_view is not None
                else None
            )
            connection.execute(
                """INSERT OR REPLACE INTO prototypes(
                    identity_id, shape_code, shape_code_shape, descriptor_json,
                    source_count, rejected_image_id, engine_version, algorithm_version,
                    updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (
                    identity_id,
                    code,
                    code_shape,
                    json.dumps(descriptors, separators=(",", ":")),
                    len(aggregate.accepted_views),
                    rejected_image_id,
                    next(iter(engine_versions)),
                    V3_ALGORITHM_VERSION,
                ),
            )
            connection.execute("DELETE FROM prototype_regions WHERE identity_id=?", (identity_id,))
            for component in COMPONENT_NAMES:
                vector, shape = _encode_array(aggregate.regions[component])
                connection.execute(
                    """INSERT INTO prototype_regions(identity_id, component, vector, vector_shape)
                    VALUES(?, ?, ?, ?)""",
                    (identity_id, component, vector, shape),
                )

    def save_calibration(self, version: str, calibration: RobustCalibration) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE calibrations SET active=0")
            connection.execute(
                """INSERT OR REPLACE INTO calibrations(version, payload_json, active)
                VALUES(?, ?, 1)""",
                (version, json.dumps(calibration.to_json(), separators=(",", ":"))),
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('calibration_version', ?)",
                (version,),
            )

    def active_calibration(self) -> RobustCalibration | None:
        if not self.path.is_file():
            return None
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM calibrations WHERE active=1"
            ).fetchone()
        return RobustCalibration.from_json(json.loads(row["payload_json"])) if row else None

    def save_hybrid_calibration(self, version: str, calibration: HybridCalibration) -> None:
        with self.connect() as connection:
            connection.execute("UPDATE calibrations SET active=0")
            connection.execute(
                """INSERT OR REPLACE INTO calibrations(version, payload_json, active)
                VALUES(?, ?, 1)""",
                (version, json.dumps(calibration.to_json(), separators=(",", ":"))),
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('calibration_version', ?)",
                (version,),
            )

    def active_hybrid_calibration(self) -> HybridCalibration | None:
        if not self.path.is_file():
            return None
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM calibrations WHERE active=1"
            ).fetchone()
        return HybridCalibration.from_json(json.loads(row["payload_json"])) if row else None

    def status(self) -> dict[str, int]:
        counts = {"identities": 0, "images": 0, "indexed": 0, "invalid": 0, "prototypes": 0}
        if not self.path.is_file():
            return counts
        with self.connect() as connection:
            counts["identities"] = int(
                connection.execute("SELECT COUNT(*) count FROM identities").fetchone()["count"]
            )
            for row in connection.execute(
                "SELECT status, COUNT(*) count FROM images GROUP BY status"
            ):
                counts[str(row["status"])] = int(row["count"])
                counts["images"] += int(row["count"])
            counts["prototypes"] = int(
                connection.execute("SELECT COUNT(*) count FROM prototypes").fetchone()["count"]
            )
        return counts

    def prototype_rows(self) -> list[sqlite3.Row]:
        if not self.path.is_file():
            return []
        with self.connect() as connection:
            return list(
                connection.execute(
                    """SELECT prototypes.*, identities.name, identities.wikidata_id,
                    images.id display_image_id, images.commons_page_id,
                    images.original_url, images.author,
                    images.license_id, images.license_url, images.local_path
                    FROM prototypes JOIN identities ON identities.id=prototypes.identity_id
                    JOIN images ON images.identity_id=identities.id AND images.selected_display=1
                    WHERE prototypes.algorithm_version=? ORDER BY identities.id""",
                    (V3_ALGORITHM_VERSION,),
                )
            )

    def identity_image_rows(self) -> list[tuple[int, list[sqlite3.Row]]]:
        if not self.path.is_file():
            return []
        with self.connect() as connection:
            identities = list(connection.execute("SELECT id FROM identities ORDER BY id"))
            return [
                (
                    int(identity["id"]),
                    list(
                        connection.execute(
                            "SELECT * FROM images WHERE identity_id=? ORDER BY id",
                            (int(identity["id"]),),
                        )
                    ),
                )
                for identity in identities
            ]

    def prototype_regions(self, identity_id: int) -> dict[str, FloatArray]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT component, vector, vector_shape FROM prototype_regions WHERE identity_id=?",
                (identity_id,),
            )
            return {
                str(row["component"]): _decode_array(row["vector"], row["vector_shape"])
                for row in rows
            }

    def prototype_records(self) -> list[tuple[sqlite3.Row, dict[str, FloatArray]]]:
        """Load all rankable prototypes and regions with one database connection."""
        if not self.path.is_file():
            return []
        with self.connect() as connection:
            rows = list(
                connection.execute(
                    """SELECT prototypes.*, identities.name, identities.wikidata_id,
                    images.id display_image_id, images.commons_page_id,
                    images.original_url, images.author,
                    images.license_id, images.license_url, images.local_path
                    FROM prototypes JOIN identities ON identities.id=prototypes.identity_id
                    JOIN images ON images.identity_id=identities.id AND images.selected_display=1
                    WHERE prototypes.algorithm_version=? ORDER BY identities.id""",
                    (V3_ALGORITHM_VERSION,),
                )
            )
            region_rows = connection.execute(
                """SELECT identity_id, component, vector, vector_shape
                FROM prototype_regions ORDER BY identity_id, component"""
            )
            regions: dict[int, dict[str, FloatArray]] = {}
            for region in region_rows:
                regions.setdefault(int(region["identity_id"]), {})[str(region["component"])] = (
                    _decode_array(region["vector"], region["vector_shape"])
                )
        return [(row, regions[int(row["identity_id"])]) for row in rows]

    @staticmethod
    def decode_shape_code(row: sqlite3.Row) -> FloatArray:
        return _decode_array(row["shape_code"], row["shape_code_shape"])
