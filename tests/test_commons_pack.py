from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from face_match.commons_pack import (
    ManifestError,
    download_manifest_images,
    parse_manifest,
    verify_remote_manifest,
)


def _image(page_id: int, *, display: bool = False) -> dict[str, object]:
    content = f"image-{page_id}".encode()
    return {
        "page_id": page_id,
        "revision_id": page_id + 100,
        "original_url": f"https://upload.wikimedia.org/example/{page_id}.jpg",
        "author": "Example Photographer",
        "license_id": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "sha256": hashlib.sha256(content).hexdigest(),
        "selected_display": display,
    }


def _manifest() -> dict[str, object]:
    return {
        "gallery_version": "commons-500-v1",
        "identities": [
            {
                "name": "Example Person",
                "wikidata_id": "Q123",
                "images": [_image(1, display=True), _image(2), _image(3)],
            }
        ],
    }


def test_commons_manifest_requires_locked_attributed_allowlisted_images() -> None:
    manifest = parse_manifest(_manifest(), expected_identities=1)
    assert manifest.gallery_version == "commons-500-v1"
    assert manifest.image_count == 3
    assert manifest.identities[0].images[0].license_id == "CC-BY-4.0"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("license_id", "ARR", "not allowlisted"),
        ("author", "", "nonempty string"),
        ("original_url", "http://example.test/a.jpg", "upload.wikimedia.org"),
        ("sha256", "bad", "64 lowercase"),
    ],
)
def test_commons_manifest_rejects_unsafe_or_unattributed_images(
    field: str, value: object, message: str
) -> None:
    payload = _manifest()
    identity = payload["identities"][0]  # type: ignore[index]
    identity["images"][0][field] = value
    with pytest.raises(ManifestError, match=message):
        parse_manifest(payload, expected_identities=1)


def test_commons_manifest_requires_three_images_and_one_display() -> None:
    payload = _manifest()
    identity = payload["identities"][0]  # type: ignore[index]
    identity["images"] = [_image(1, display=True), _image(2)]
    with pytest.raises(ManifestError, match="exactly three"):
        parse_manifest(payload, expected_identities=1)

    payload = _manifest()
    identity = payload["identities"][0]  # type: ignore[index]
    for image in identity["images"]:
        image["selected_display"] = False
    with pytest.raises(ManifestError, match="exactly one selected"):
        parse_manifest(payload, expected_identities=1)


def test_commons_manifest_verifies_remote_revision_license_and_attribution() -> None:
    manifest = parse_manifest(_manifest(), expected_identities=1)

    def fetch(page_ids: tuple[int, ...]) -> dict[int, dict[str, object]]:
        return {
            page_id: {
                "pageid": page_id,
                "revisions": [{"revid": page_id + 100}],
                "imageinfo": [
                    {
                        "url": f"https://upload.wikimedia.org/example/{page_id}.jpg",
                        "extmetadata": {
                            "Artist": {"value": "<b>Example Photographer</b>"},
                            "LicenseShortName": {"value": "CC BY 4.0"},
                            "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
                        },
                    }
                ],
            }
            for page_id in page_ids
        }

    verify_remote_manifest(manifest, fetch)
    changed = fetch((1, 2, 3))
    changed[2]["revisions"] = [{"revid": 999}]
    with pytest.raises(ManifestError, match="revision changed"):
        verify_remote_manifest(manifest, lambda _: changed)


def test_commons_download_is_checksum_locked_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _manifest()
    for image in payload["identities"][0]["images"]:  # type: ignore[index]
        page_id = int(image["page_id"])
        image["sha256"] = hashlib.sha256(f"image-{page_id}".encode()).hexdigest()
    manifest = parse_manifest(payload, expected_identities=1)

    class Response(BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    calls = 0

    def open_image(url: object, timeout: int) -> Response:
        nonlocal calls
        del timeout
        calls += 1
        page_id = Path(str(getattr(url, "full_url", url))).stem
        return Response(f"image-{page_id}".encode())

    monkeypatch.setattr("face_match.commons_pack.urllib.request.urlopen", open_image)
    first = download_manifest_images(manifest, tmp_path / "commons", verify_remote=False)
    second = download_manifest_images(manifest, tmp_path / "commons", verify_remote=False)
    assert first == {"downloaded": 3, "reused": 0, "total": 3}
    assert second == {"downloaded": 0, "reused": 3, "total": 3}
    assert calls == 3
