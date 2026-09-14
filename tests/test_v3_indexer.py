from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from face_match.commons_pack import CommonsManifest, parse_manifest
from face_match.config import Settings
from face_match.v3_database import V3Database
from face_match.v3_indexer import _validate_gallery_views, index_v3_gallery

from .helpers import FakeMicaEngine, SequenceDetector, dense_detection


def _gallery(root: Path, identities: int = 5) -> CommonsManifest:
    raw_identities = []
    payloads: dict[tuple[int, int], bytes] = {}
    for identity in range(identities):
        images = []
        for view in range(3):
            stream = BytesIO()
            Image.new("RGB", (768, 768), (80 + identity * 10, 90 + view * 10, 120)).save(
                stream, "PNG"
            )
            payload = stream.getvalue()
            payloads[(identity, view)] = payload
            images.append(
                {
                    "page_id": identity * 3 + view + 1,
                    "revision_id": identity * 3 + view + 100,
                    "original_url": (f"https://upload.wikimedia.org/fixture/{identity}-{view}.png"),
                    "author": "Fixture photographer",
                    "license_id": "CC-BY-SA-4.0",
                    "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "selected_display": view == 0,
                }
            )
        raw_identities.append(
            {"name": f"Person {identity}", "wikidata_id": f"Q{identity + 1}", "images": images}
        )
    manifest = parse_manifest(
        {"gallery_version": "index-test-v1", "identities": raw_identities},
        expected_identities=identities,
    )
    for identity_index, identity_entry in enumerate(manifest.identities):
        identity_root = root / identity_entry.wikidata_id
        identity_root.mkdir(parents=True, exist_ok=True)
        for view, image in enumerate(identity_entry.images):
            (identity_root / image.local_name).write_bytes(payloads[(identity_index, view)])
    return manifest


def test_v3_index_builds_and_resumes_identity_prototypes(
    tmp_path: Path, monkeypatch: object
) -> None:
    gallery_root = tmp_path / "commons"
    manifest = _gallery(gallery_root)
    detections = [
        dense_detection(200 + identity * 3 + view, (0.0, -20.0, 20.0)[view])
        for identity in range(5)
        for view in range(3)
    ]
    detector = SequenceDetector(detections)
    mica = FakeMicaEngine()
    database = V3Database(tmp_path / "v3.sqlite3")
    settings = Settings(
        tmp_path,
        tmp_path / "model.task",
        tmp_path / "v2.sqlite3",
        tmp_path / "lfw",
        v3_database_path=tmp_path / "v3.sqlite3",
        v3_dataset_path=gallery_root,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "face_match.v3_indexer.maximum_descriptor_correlation", lambda _: 0.5
    )
    first = index_v3_gallery(manifest, database, detector, mica, settings)
    assert first["indexed"] == 5
    assert database.status()["prototypes"] == 5
    assert database.active_hybrid_calibration() is not None
    assert mica.calls == 5
    second = index_v3_gallery(manifest, database, detector, mica, settings)
    assert second["reused"] == 5
    assert mica.calls == 5


def test_v3_gallery_requires_one_frontal_view_and_adequate_jaw_coverage() -> None:
    all_side = [dense_detection(seed, yaw=20.0) for seed in range(3)]
    try:
        _validate_gallery_views(all_side)
    except ValueError as error:
        assert "near-frontal" in str(error)
    else:
        raise AssertionError("gallery validation accepted three side views")

    weak = list(all_side)
    weak[0] = dense_detection(1, yaw=0.0)
    object.__setattr__(weak[1], "quality", 0.4)
    try:
        _validate_gallery_views(weak)
    except ValueError as error:
        assert "jaw landmark coverage" in str(error)
    else:
        raise AssertionError("gallery validation accepted weak jaw coverage")
