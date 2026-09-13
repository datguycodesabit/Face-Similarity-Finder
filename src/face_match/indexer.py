from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .config import Settings
from .database import Database
from .detector import Detector
from .errors import FaceMatchError
from .geometry import LANDMARK_VERSION
from .images import decode_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def discover_images(dataset: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def subject_name(path: Path, dataset: Path) -> str:
    relative = path.relative_to(dataset)
    folder = relative.parts[0] if len(relative.parts) > 1 else path.stem.rsplit("_", 1)[0]
    return folder.replace("_", " ").strip()


def index_dataset(
    dataset: Path,
    database: Database,
    detector: Detector,
    settings: Settings,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")
    database.initialize()
    database.set_metadata("indexing_status", "running")
    files = discover_images(dataset)
    registered = [
        (
            path,
            str(path.relative_to(dataset)),
            database.register_image(
                subject_name(path, dataset),
                str(path.relative_to(dataset)),
                detector.model_version,
            ),
        )
        for path in files
    ]
    indexed = invalid = skipped = 0
    for position, (path, relative, image_id) in enumerate(registered, start=1):
        if not database.needs_indexing(image_id, detector.model_version):
            skipped += 1
        else:
            try:
                data = path.read_bytes()
                detection = detector.detect_one(decode_image(data, settings))
                database.save_vector(
                    image_id,
                    detection.normalized,
                    detection.overlay,
                    detector.model_version,
                    detection.measurements,
                    detection.yaw,
                    detection.pitch,
                    detection.roll,
                    detection.quality,
                    detection.eligible,
                    detection.exclusion_reason,
                )
                indexed += 1
            except (FaceMatchError, OSError, RuntimeError, ValueError) as error:
                database.save_invalid(image_id, str(error), detector.model_version)
                invalid += 1
        if progress:
            progress(position, len(files), relative)
    database.set_metadata("dataset_root", str(dataset.resolve()))
    database.set_metadata("model_version", detector.model_version)
    database.set_metadata("landmark_version", LANDMARK_VERSION)
    database.set_metadata("indexing_status", "complete")
    return {"discovered": len(files), "indexed": indexed, "invalid": invalid, "skipped": skipped}
