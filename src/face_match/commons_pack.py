from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode, urlparse

ALLOWED_LICENSES = frozenset(
    {
        "CC0-1.0",
        "CC-BY-2.0",
        "CC-BY-2.5",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC-BY-SA-2.0",
        "CC-BY-SA-2.5",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
        "PUBLIC-DOMAIN",
    }
)
WIKIDATA_PATTERN = re.compile(r"Q[1-9][0-9]*")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
LICENSE_ALIASES = {
    "CC0": "CC0-1.0",
    "CC0 1.0": "CC0-1.0",
    "PUBLIC DOMAIN": "PUBLIC-DOMAIN",
}


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class CommonsImage:
    page_id: int
    revision_id: int
    original_url: str
    author: str
    license_id: str
    license_url: str
    sha256: str
    selected_display: bool

    @property
    def extension(self) -> str:
        suffix = Path(urlparse(self.original_url).path).suffix.lower()
        return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".img"

    @property
    def local_name(self) -> str:
        return f"{self.page_id}-{self.revision_id}-{self.sha256[:12]}{self.extension}"


@dataclass(frozen=True)
class CommonsIdentity:
    name: str
    wikidata_id: str
    images: tuple[CommonsImage, ...]


@dataclass(frozen=True)
class CommonsManifest:
    gallery_version: str
    identities: tuple[CommonsIdentity, ...]

    @property
    def image_count(self) -> int:
        return sum(len(identity.images) for identity in self.identities)


def _required_string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"manifest field {name} must be a nonempty string")
    return value.strip()


def _parse_image(payload: Mapping[str, Any]) -> CommonsImage:
    try:
        page_id = int(payload["page_id"])
        revision_id = int(payload["revision_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError("Commons page_id and revision_id must be positive integers") from error
    if min(page_id, revision_id) <= 0:
        raise ManifestError("Commons page_id and revision_id must be positive integers")
    original_url = _required_string(payload, "original_url")
    parsed = urlparse(original_url)
    if parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org":
        raise ManifestError("original_url must be an HTTPS upload.wikimedia.org URL")
    license_id = _required_string(payload, "license_id").upper()
    if license_id not in ALLOWED_LICENSES:
        raise ManifestError(f"license {license_id} is not allowlisted")
    license_url = _required_string(payload, "license_url")
    if urlparse(license_url).scheme != "https":
        raise ManifestError("license_url must use HTTPS")
    sha256 = _required_string(payload, "sha256").lower()
    if not SHA256_PATTERN.fullmatch(sha256):
        raise ManifestError("sha256 must contain 64 lowercase hexadecimal characters")
    return CommonsImage(
        page_id=page_id,
        revision_id=revision_id,
        original_url=original_url,
        author=_required_string(payload, "author"),
        license_id=license_id,
        license_url=license_url,
        sha256=sha256,
        selected_display=payload.get("selected_display") is True,
    )


def parse_manifest(
    payload: Mapping[str, Any], *, expected_identities: int | None = 500
) -> CommonsManifest:
    gallery_version = _required_string(payload, "gallery_version")
    raw_identities = payload.get("identities")
    if not isinstance(raw_identities, list):
        raise ManifestError("manifest identities must be a list")
    if expected_identities is not None and len(raw_identities) != expected_identities:
        raise ManifestError(
            f"manifest must contain exactly {expected_identities} identities; "
            f"received {len(raw_identities)}"
        )
    identities: list[CommonsIdentity] = []
    seen_names: set[str] = set()
    seen_wikidata: set[str] = set()
    seen_images: set[tuple[int, int]] = set()
    seen_hashes: set[str] = set()
    for raw_identity in raw_identities:
        if not isinstance(raw_identity, Mapping):
            raise ManifestError("each identity must be an object")
        name = _required_string(raw_identity, "name")
        wikidata_id = _required_string(raw_identity, "wikidata_id").upper()
        if not WIKIDATA_PATTERN.fullmatch(wikidata_id):
            raise ManifestError(f"invalid Wikidata identifier: {wikidata_id}")
        if name.casefold() in seen_names or wikidata_id in seen_wikidata:
            raise ManifestError(f"duplicate Commons identity: {name}")
        raw_images = raw_identity.get("images")
        if not isinstance(raw_images, list) or len(raw_images) != 3:
            raise ManifestError(f"{name} must have exactly three reference images")
        images = tuple(_parse_image(image) for image in raw_images)
        if sum(image.selected_display for image in images) != 1:
            raise ManifestError(f"{name} must have exactly one selected display image")
        for image in images:
            page_revision = (image.page_id, image.revision_id)
            if page_revision in seen_images or image.sha256 in seen_hashes:
                raise ManifestError(f"duplicate reference image for {name}")
            seen_images.add(page_revision)
            seen_hashes.add(image.sha256)
        identities.append(CommonsIdentity(name, wikidata_id, images))
        seen_names.add(name.casefold())
        seen_wikidata.add(wikidata_id)
    return CommonsManifest(gallery_version, tuple(identities))


def load_manifest(path: Path, *, expected_identities: int | None = 500) -> CommonsManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"could not read Commons manifest: {error}") from error
    if not isinstance(payload, Mapping):
        raise ManifestError("Commons manifest root must be an object")
    return parse_manifest(payload, expected_identities=expected_identities)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_metadata(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def _license_id(value: str) -> str:
    cleaned = " ".join(value.upper().replace("_", " ").split())
    if cleaned in LICENSE_ALIASES:
        return LICENSE_ALIASES[cleaned]
    return cleaned.replace(" ", "-")


def _metadata_value(metadata: Mapping[str, Any], name: str) -> str:
    field = metadata.get(name)
    return str(field.get("value", "")) if isinstance(field, Mapping) else ""


def _remote_batch(page_ids: tuple[int, ...]) -> Mapping[int, Mapping[str, Any]]:
    query = urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "pageids": "|".join(str(page_id) for page_id in page_ids),
            "prop": "imageinfo|revisions",
            "iiprop": "url|extmetadata",
            "rvprop": "ids",
        }
    )
    request = urllib.request.Request(
        f"{COMMONS_API}?{query}",
        headers={"User-Agent": "FaceShapeStudio/0.1 (local personal-use manifest verifier)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if isinstance(payload, Mapping) and isinstance(payload.get("error"), Mapping):
        error = payload["error"]
        raise ManifestError(
            f"Commons API error {error.get('code', 'unknown')}: "
            f"{error.get('info', 'manifest verification failed')}"
        )
    pages = payload.get("query", {}).get("pages", []) if isinstance(payload, Mapping) else []
    return {
        int(page["pageid"]): page
        for page in pages
        if isinstance(page, Mapping) and "pageid" in page
    }


def verify_remote_manifest(
    manifest: CommonsManifest,
    fetch_batch: Callable[[tuple[int, ...]], Mapping[int, Mapping[str, Any]]] = _remote_batch,
) -> None:
    """Verify that every manifest entry still names the locked Commons revision/metadata."""
    expected = {
        image.page_id: image for identity in manifest.identities for image in identity.images
    }
    page_ids = tuple(expected)
    # extmetadata is an expensive Commons property; keep batches deliberately small.
    for offset in range(0, len(page_ids), 10):
        batch_ids = page_ids[offset : offset + 10]
        remote = fetch_batch(batch_ids)
        for page_id in batch_ids:
            image = expected[page_id]
            page = remote.get(page_id)
            if page is None or page.get("missing") is True:
                raise ManifestError(f"Commons page {page_id} is missing")
            revisions = page.get("revisions")
            information = page.get("imageinfo")
            if (
                not isinstance(revisions, list)
                or not revisions
                or not isinstance(information, list)
                or not information
            ):
                raise ManifestError(f"Commons page {page_id} lacks revision or image metadata")
            revision = revisions[0]
            info = information[0]
            if (
                not isinstance(revision, Mapping)
                or int(revision.get("revid", 0)) != image.revision_id
            ):
                raise ManifestError(f"Commons page {page_id} revision changed")
            if not isinstance(info, Mapping) or str(info.get("url", "")) != image.original_url:
                raise ManifestError(f"Commons page {page_id} original file URL changed")
            metadata = info.get("extmetadata")
            if not isinstance(metadata, Mapping):
                raise ManifestError(f"Commons page {page_id} lacks attribution metadata")

            remote_license = _license_id(_metadata_value(metadata, "LicenseShortName"))
            remote_license_url = _metadata_value(metadata, "LicenseUrl")
            remote_author = _plain_metadata(_metadata_value(metadata, "Artist"))
            if remote_license != image.license_id:
                raise ManifestError(f"Commons page {page_id} license changed")
            if remote_license_url.rstrip("/") != image.license_url.rstrip("/"):
                raise ManifestError(f"Commons page {page_id} license URL changed")
            if remote_author != image.author:
                raise ManifestError(f"Commons page {page_id} author attribution changed")


def download_manifest_images(
    manifest: CommonsManifest,
    destination: Path,
    progress: Callable[[int, int, CommonsIdentity, CommonsImage], None] | None = None,
    *,
    verify_remote: bool = True,
) -> dict[str, int]:
    """Download locked Commons revisions. Existing checksum-valid files are reused."""
    if verify_remote:
        verify_remote_manifest(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = reused = 0
    completed = 0
    for identity in manifest.identities:
        identity_root = destination / identity.wikidata_id
        identity_root.mkdir(parents=True, exist_ok=True)
        for image in identity.images:
            target = identity_root / image.local_name
            if target.is_file() and _sha256(target) == image.sha256:
                reused += 1
            else:
                temporary = target.with_suffix(target.suffix + ".partial")
                try:
                    request = urllib.request.Request(
                        image.original_url,
                        headers={
                            "User-Agent": (
                                "FaceShapeStudio/0.1 "
                                "(local personal-use Commons reference downloader)"
                            )
                        },
                    )
                    with (
                        urllib.request.urlopen(request, timeout=60) as response,
                        temporary.open("wb") as output,
                    ):
                        shutil.copyfileobj(response, output)
                    actual = _sha256(temporary)
                    if actual != image.sha256:
                        raise ManifestError(
                            f"checksum mismatch for Commons page {image.page_id}: "
                            f"expected {image.sha256}, received {actual}"
                        )
                    temporary.replace(target)
                    downloaded += 1
                except (OSError, URLError):
                    temporary.unlink(missing_ok=True)
                    raise
            completed += 1
            if progress:
                progress(completed, manifest.image_count, identity, image)
    return {"downloaded": downloaded, "reused": reused, "total": manifest.image_count}
