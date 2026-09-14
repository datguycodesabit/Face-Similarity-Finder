from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .commons_pack import CommonsManifest
from .config import Settings
from .detector import Detection, Detector
from .images import decode_image
from .mica import MicaEngine
from .shape_v3 import V3_ALGORITHM_VERSION, HybridCalibration, maximum_descriptor_correlation
from .v3_database import V3Database

Progress = Callable[[int, int, str], None]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_gallery_views(detections: list[Detection]) -> None:
    if not detections:
        raise ValueError("no gallery views were detected")
    if not any(abs(item.yaw) <= 10.0 for item in detections):
        raise ValueError("identity requires at least one near-frontal view (≤10° yaw)")
    for item in detections:
        if abs(item.yaw) > 35.0:
            raise ValueError("gallery view yaw exceeds 35°")
        if abs(item.pitch) > 12.0:
            raise ValueError("gallery view pitch exceeds 12°")
        if item.quality < 0.85:
            raise ValueError("face crop or jaw landmark coverage is too weak")
        if not item.v3_measurements:
            raise ValueError("detector did not return V3 isotropic descriptors")


def index_v3_gallery(
    manifest: CommonsManifest,
    database: V3Database,
    detector: Detector,
    mica: MicaEngine,
    settings: Settings,
    progress: Progress | None = None,
) -> dict[str, int | str]:
    """Build or resume identity-level V3 prototypes without modifying the V2 database."""
    dataset_root = settings.v3_dataset_path or settings.project_root / "data" / "commons_v3"
    database.register_manifest(manifest, dataset_root)
    database.set_metadata("indexing_status", "running")
    database.set_metadata("algorithm_version", V3_ALGORITHM_VERSION)
    database.set_metadata("mica_engine_version", mica.engine_version)
    strict_settings = replace(
        settings,
        min_dimension=768,
        max_dimension=max(settings.max_dimension, 12000),
        max_upload_bytes=max(settings.max_upload_bytes, 40 * 1024 * 1024),
    )
    completed = indexed = reused = invalid = 0
    identity_rows = database.identity_image_rows()
    try:
        for identity_id, rows in identity_rows:
            if len(rows) != 3:
                raise ValueError(f"identity {identity_id} does not have exactly three images")
            needs_indexing = [
                database.needs_indexing(int(row["id"]), str(row["sha256"]), mica.engine_version)
                for row in rows
            ]
            if not any(needs_indexing):
                try:
                    database.build_prototype(identity_id)
                    reused += 1
                except ValueError:
                    pass
                completed += 1
                if progress:
                    progress(completed, len(identity_rows), str(rows[0]["local_path"]))
                continue
            try:
                payloads = [Path(str(row["local_path"])).read_bytes() for row in rows]
                for row, payload in zip(rows, payloads, strict=True):
                    actual = _sha256(payload)
                    if actual != str(row["sha256"]):
                        raise ValueError(
                            f"checksum mismatch for Commons page {row['commons_page_id']}"
                        )
                images = [decode_image(payload, strict_settings) for payload in payloads]
                detections = [detector.detect_one(image) for image in images]
                _validate_gallery_views(list(detections))
                anchor_index = next(
                    index for index, row in enumerate(rows) if bool(row["selected_display"])
                )
                results = mica.analyze(
                    payloads,
                    validate_identity=True,
                    anchor_index=anchor_index,
                )
                for row, result, detection, needs_save in zip(
                    rows, results, detections, needs_indexing, strict=True
                ):
                    if not needs_save:
                        continue
                    database.save_image_shape(
                        int(row["id"]),
                        str(row["sha256"]),
                        mica.engine_version,
                        result,
                        detection.v3_measurements,
                        yaw=detection.yaw,
                        pitch=detection.pitch,
                        roll=detection.roll,
                        quality=detection.quality,
                    )
                database.build_prototype(identity_id)
                indexed += 1
            except Exception as error:
                invalid += 1
                for row in rows:
                    database.save_invalid(int(row["id"]), str(error))
            completed += 1
            if progress:
                progress(completed, len(identity_rows), str(rows[0]["local_path"]))

        prototype_rows = database.prototype_rows()
        if len(prototype_rows) < 5:
            raise ValueError("at least five valid identity prototypes are required for calibration")
        descriptors = [json.loads(str(row["descriptor_json"])) for row in prototype_rows]
        correlation = maximum_descriptor_correlation(descriptors)
        if correlation >= 0.95:
            raise ValueError(
                f"V3 descriptor independence gate failed: maximum |r| is {correlation:.4f}"
            )
        regions = [database.prototype_regions(int(row["identity_id"])) for row in prototype_rows]
        calibration = HybridCalibration.fit(descriptors, regions)
        calibration_version = f"{manifest.gallery_version}-{mica.engine_version}"
        database.save_hybrid_calibration(calibration_version, calibration)
        database.set_metadata("indexing_status", "complete")
        database.set_metadata("descriptor_max_correlation", f"{correlation:.8f}")
    except Exception:
        database.set_metadata("indexing_status", "failed")
        raise
    return {
        "identities": len(identity_rows),
        "indexed": indexed,
        "reused": reused,
        "invalid": invalid,
        "status": "complete",
    }
