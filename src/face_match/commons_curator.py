from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import unquote, urlencode

from .commons_candidates import USER_AGENT
from .commons_pack import (
    ALLOWED_LICENSES,
    COMMONS_API,
    CommonsImage,
    _license_id,
    _plain_metadata,
    parse_manifest,
)
from .config import Settings
from .detector import Detection, Detector
from .errors import FaceMatchError
from .images import decode_image
from .mica import MicaEngine, aggregate_mica_views


@dataclass(frozen=True)
class CommonsCandidate:
    name: str
    wikidata_id: str
    commons_category: str
    primary_file: str


@dataclass(frozen=True)
class CommonsSource:
    page_id: int
    revision_id: int
    original_url: str
    author: str
    license_id: str
    license_url: str
    width: int
    height: int
    mime: str
    primary: bool
    analysis_url: str | None = None


class CommonsServiceError(RuntimeError):
    """A transient or protocol-level Commons failure, not a rejected identity."""


_MIN_DOWNLOAD_INTERVAL_SECONDS = 1.0
_last_download_started = 0.0
_MIN_API_INTERVAL_SECONDS = 0.5
_last_api_started = 0.0


def load_candidate_catalog(path: Path) -> tuple[CommonsCandidate, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("Commons candidate catalog has an incompatible schema")
    raw = payload.get("candidates")
    if not isinstance(raw, list):
        raise ValueError("Commons candidate catalog is missing candidates")
    candidates: list[CommonsCandidate] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Commons candidate entry must be an object")
        values = [
            str(item.get(name, "")).strip()
            for name in ("name", "wikidata_id", "commons_category", "primary_file")
        ]
        if not all(values) or values[1] in seen:
            raise ValueError("Commons candidate fields must be nonempty and QIDs unique")
        values[3] = unquote(values[3]).replace("_", " ")
        candidates.append(CommonsCandidate(*values))
        seen.add(values[1])
    if len(candidates) < 500:
        raise ValueError("Commons curation requires at least 500 surplus candidates")
    return tuple(candidates)


def _fetch_json(parameters: Mapping[str, str]) -> Mapping[str, Any]:
    global _last_api_started
    payload: Any = None
    for attempt in range(5):
        wait = _MIN_API_INTERVAL_SECONDS - (time.monotonic() - _last_api_started)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(
            f"{COMMONS_API}?{urlencode(parameters)}", headers={"User-Agent": USER_AGENT}
        )
        try:
            _last_api_started = time.monotonic()
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            break
        except HTTPError as error:
            if error.code != 429 and not 500 <= error.code < 600:
                raise
            if attempt == 4:
                raise CommonsServiceError(
                    f"Commons API remained unavailable (HTTP {error.code})"
                ) from error
            retry_after = error.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else float(2**attempt)
            except ValueError:
                delay = float(2**attempt)
            time.sleep(min(max(delay, 1.0), 30.0))
    if not isinstance(payload, Mapping):
        raise ValueError("Commons returned a non-object response")
    api_error = payload.get("error")
    if isinstance(api_error, Mapping):
        code = str(api_error.get("code", "unknown"))
        info = str(api_error.get("info", "Commons API request failed"))
        raise CommonsServiceError(f"Commons API error {code}: {info}")
    return payload


def category_file_titles(
    category: str,
    *,
    limit: int = 30,
    fetch: Callable[[Mapping[str, str]], Mapping[str, Any]] = _fetch_json,
) -> tuple[str, ...]:
    if not 3 <= limit <= 100:
        raise ValueError("Commons category file limit must be between 3 and 100")
    title = category if category.startswith("Category:") else f"Category:{category}"
    payload = fetch(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "categorymembers",
            "cmtitle": title,
            "cmtype": "file",
            "cmlimit": str(limit),
            "cmprop": "ids|title",
        }
    )
    members = payload.get("query", {}).get("categorymembers", [])
    if not isinstance(members, list):
        return ()
    return tuple(
        str(member["title"])
        for member in members
        if isinstance(member, Mapping) and str(member.get("title", "")).startswith("File:")
    )


def _metadata_value(metadata: Mapping[str, Any], name: str) -> str:
    field = metadata.get(name)
    return str(field.get("value", "")) if isinstance(field, Mapping) else ""


def source_metadata(
    titles: tuple[str, ...],
    primary_file: str,
    *,
    fetch: Callable[[Mapping[str, str]], Mapping[str, Any]] = _fetch_json,
) -> tuple[CommonsSource, ...]:
    primary_title = primary_file if primary_file.startswith("File:") else f"File:{primary_file}"
    ordered = tuple(dict.fromkeys((primary_title, *titles)))
    sources: list[CommonsSource] = []
    seen_pages: set[int] = set()
    for offset in range(0, len(ordered), 25):
        payload = fetch(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "titles": "|".join(ordered[offset : offset + 25]),
                "prop": "imageinfo|revisions",
                "iiprop": "url|extmetadata|size|mime",
                "iiurlwidth": "1440",
                "iiextmetadatafilter": "Artist|LicenseShortName|LicenseUrl",
                "rvprop": "ids",
            }
        )
        pages = payload.get("query", {}).get("pages", [])
        if not isinstance(pages, list):
            continue
        for page in pages:
            if not isinstance(page, Mapping) or page.get("missing") is True:
                continue
            revisions = page.get("revisions")
            imageinfo = page.get("imageinfo")
            if (
                not isinstance(revisions, list)
                or not revisions
                or not isinstance(imageinfo, list)
                or not imageinfo
            ):
                continue
            revision, info = revisions[0], imageinfo[0]
            if not isinstance(revision, Mapping) or not isinstance(info, Mapping):
                continue
            metadata = info.get("extmetadata")
            if not isinstance(metadata, Mapping):
                continue
            license_id = _license_id(_metadata_value(metadata, "LicenseShortName"))
            author = _plain_metadata(_metadata_value(metadata, "Artist"))
            license_url = _metadata_value(metadata, "LicenseUrl")
            mime = str(info.get("mime", ""))
            width, height = int(info.get("width", 0)), int(info.get("height", 0))
            original_url = str(info.get("url", ""))
            analysis_url = str(info.get("thumburl", ""))
            page_id = int(page["pageid"])
            if (
                page_id in seen_pages
                or license_id not in ALLOWED_LICENSES
                or not author
                or not license_url.startswith("https://")
                or not original_url.startswith("https://upload.wikimedia.org/")
                or mime not in {"image/jpeg", "image/png", "image/webp"}
                or min(width, height) < 768
            ):
                continue
            sources.append(
                CommonsSource(
                    page_id=page_id,
                    revision_id=int(revision["revid"]),
                    original_url=original_url,
                    author=author,
                    license_id=license_id,
                    license_url=license_url,
                    width=width,
                    height=height,
                    mime=mime,
                    primary=str(page.get("title", "")) == primary_title,
                    analysis_url=(
                        analysis_url
                        if analysis_url.startswith("https://thumb.wikimedia.org/")
                        or analysis_url.startswith("https://upload.wikimedia.org/")
                        else None
                    ),
                )
            )
            seen_pages.add(page_id)
    sources.sort(key=lambda source: (not source.primary, source.page_id))
    return tuple(sources)


def candidate_sources(
    candidate: CommonsCandidate,
    *,
    limit: int = 30,
    fetch: Callable[[Mapping[str, str]], Mapping[str, Any]] = _fetch_json,
) -> tuple[CommonsSource, ...]:
    titles = category_file_titles(candidate.commons_category, limit=limit, fetch=fetch)
    return source_metadata(titles, candidate.primary_file, fetch=fetch)


def _download_url(url: str, maximum_bytes: int) -> bytes:
    global _last_download_started
    payload: bytes | None = None
    for attempt in range(5):
        wait = _MIN_DOWNLOAD_INTERVAL_SECONDS - (time.monotonic() - _last_download_started)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            _last_download_started = time.monotonic()
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = bytes(response.read(maximum_bytes + 1))
            break
        except HTTPError as error:
            if error.code != 429 and not 500 <= error.code < 600:
                raise
            if attempt == 4:
                raise CommonsServiceError(
                    f"Commons download remained unavailable (HTTP {error.code})"
                ) from error
            retry_after = error.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else float(2**attempt)
            except ValueError:
                delay = float(2**attempt)
            time.sleep(min(max(delay, 1.0), 30.0))
    if payload is None:  # pragma: no cover - loop either succeeds or raises
        raise CommonsServiceError("Commons download failed without a response")
    if len(payload) > maximum_bytes:
        raise ValueError("Commons image exceeds the download size limit")
    return payload


def _download_source(source: CommonsSource, maximum_bytes: int) -> bytes:
    """Download the API-provided analysis thumbnail, falling back only if unavailable."""
    return _download_url(source.analysis_url or source.original_url, maximum_bytes)


def _download_original_source(source: CommonsSource, maximum_bytes: int) -> bytes:
    return _download_url(source.original_url, maximum_bytes)


def _acceptable_detection(detection: Detection) -> bool:
    return bool(
        abs(detection.yaw) <= 35.0
        and abs(detection.pitch) <= 12.0
        and detection.quality >= 0.85
        and detection.v3_measurements
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def curate_gallery(
    candidates: Sequence[CommonsCandidate],
    destination: Path,
    manifest_path: Path,
    detector: Detector,
    mica: MicaEngine,
    settings: Settings,
    *,
    target: int = 500,
    sources_for: Callable[[CommonsCandidate], Sequence[CommonsSource]] = candidate_sources,
    download: Callable[[CommonsSource, int], bytes] = _download_source,
    download_original: Callable[[CommonsSource, int], bytes] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, int | str]:
    """Accept exactly `target` three-photo identities after real local vision checks."""
    if target < 5 or len(candidates) < target:
        raise ValueError("curation needs at least target candidates and a target of five or more")
    candidate_material = json.dumps(
        [candidate.__dict__ for candidate in candidates], sort_keys=True, separators=(",", ":")
    ).encode()
    candidate_digest = hashlib.sha256(candidate_material).hexdigest()
    state_path = destination / ".curation-state.json"
    accepted: list[dict[str, Any]] = []
    rejected = 0
    start_index = 0
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, Mapping) or (
            state.get("schema_version") != 1
            or state.get("candidate_digest") != candidate_digest
            or state.get("engine_version") != mica.engine_version
            or state.get("target") != target
        ):
            raise ValueError(
                f"incompatible curation checkpoint at {state_path}; move it aside to restart"
            )
        raw_accepted = state.get("accepted")
        if not isinstance(raw_accepted, list):
            raise ValueError(f"curation checkpoint at {state_path} is corrupt")
        accepted = [dict(item) for item in raw_accepted if isinstance(item, Mapping)]
        rejected = int(state.get("rejected", 0))
        start_index = int(state.get("next_candidate_index", 0))
    strict_settings = Settings(
        settings.project_root,
        settings.model_path,
        settings.database_path,
        settings.dataset_path,
        max_upload_bytes=max(settings.max_upload_bytes, 40 * 1024 * 1024),
        max_dimension=max(settings.max_dimension, 12000),
        min_dimension=768,
        v3_database_path=settings.v3_database_path,
        v3_dataset_path=settings.v3_dataset_path,
        commons_manifest_path=settings.commons_manifest_path,
    )
    original_downloader = (
        _download_original_source if download is _download_source else download
    ) if download_original is None else download_original
    accepted_names = {str(item["name"]).casefold() for item in accepted}
    accepted_pages = {
        int(image["page_id"])
        for identity in accepted
        for image in identity.get("images", [])
        if isinstance(image, Mapping)
    }
    accepted_hashes = {
        str(image["sha256"])
        for identity in accepted
        for image in identity.get("images", [])
        if isinstance(image, Mapping)
    }
    for candidate_index, candidate in enumerate(candidates[start_index:], start=start_index):
        if len(accepted) == target:
            break
        usable: list[tuple[CommonsSource, bytes, Detection]] = []
        chosen: tuple[tuple[CommonsSource, bytes, Detection], ...] | None = None
        mica_attempts = 0
        sources = sources_for(candidate)
        try:
            for source in sources[:18]:
                try:
                    payload = download(source, strict_settings.max_upload_bytes)
                    decoded = decode_image(payload, strict_settings)
                    detection = detector.detect_one(decoded)
                    if _acceptable_detection(detection):
                        usable.append((source, payload, detection))
                except CommonsServiceError:
                    raise
                except (FaceMatchError, OSError, RuntimeError, ValueError):
                    continue
                if len(usable) < 3:
                    continue
                # Only try new triplets containing the most recently accepted image.
                # This both avoids repeat work and lets successful identities stop early.
                for pair in combinations(usable[:-1], 2):
                    if mica_attempts >= 12:
                        break
                    triplet = (*pair, usable[-1])
                    frontal = [
                        index for index, item in enumerate(triplet) if abs(item[2].yaw) <= 10.0
                    ]
                    if not frontal:
                        continue
                    anchor = min(frontal, key=lambda index: abs(triplet[index][2].yaw))
                    mica_attempts += 1
                    try:
                        results = mica.analyze(
                            [item[1] for item in triplet],
                            validate_identity=True,
                            anchor_index=anchor,
                        )
                        aggregate_mica_views(results)
                    except (RuntimeError, ValueError):
                        continue
                    chosen = triplet
                    break
                if chosen is not None or mica_attempts >= 12:
                    break
            if chosen is None:
                raise ValueError("no same-person triplet passed the pose and MICA gates")
            if candidate.name.casefold() in accepted_names:
                raise ValueError("candidate display name duplicates an accepted identity")
            display_index = min(range(3), key=lambda index: abs(chosen[index][2].yaw))
            raw_images: list[dict[str, Any]] = []
            identity_root = destination / candidate.wikidata_id
            identity_root.mkdir(parents=True, exist_ok=True)
            for index, (source, _, _) in enumerate(chosen):
                payload = original_downloader(source, strict_settings.max_upload_bytes)
                digest = hashlib.sha256(payload).hexdigest()
                manifest_image = CommonsImage(
                    source.page_id,
                    source.revision_id,
                    source.original_url,
                    source.author,
                    source.license_id,
                    source.license_url,
                    digest,
                    index == display_index,
                )
                if (
                    manifest_image.page_id in accepted_pages
                    or manifest_image.sha256 in accepted_hashes
                ):
                    raise ValueError(
                        "candidate reuses an image already accepted for another identity"
                    )
                target_path = identity_root / manifest_image.local_name
                if (
                    not target_path.is_file()
                    or hashlib.sha256(target_path.read_bytes()).hexdigest() != digest
                ):
                    temporary = target_path.with_suffix(target_path.suffix + ".partial")
                    temporary.write_bytes(payload)
                    temporary.replace(target_path)
                raw_images.append(
                    {
                        "page_id": manifest_image.page_id,
                        "revision_id": manifest_image.revision_id,
                        "original_url": manifest_image.original_url,
                        "author": manifest_image.author,
                        "license_id": manifest_image.license_id,
                        "license_url": manifest_image.license_url,
                        "sha256": manifest_image.sha256,
                        "selected_display": manifest_image.selected_display,
                    }
                )
            accepted.append(
                {
                    "name": candidate.name,
                    "wikidata_id": candidate.wikidata_id,
                    "images": raw_images,
                }
            )
            accepted_names.add(candidate.name.casefold())
            accepted_pages.update(int(image["page_id"]) for image in raw_images)
            accepted_hashes.update(str(image["sha256"]) for image in raw_images)
        except CommonsServiceError:
            raise
        except (FaceMatchError, OSError, RuntimeError, ValueError):
            rejected += 1
        _write_json_atomic(
            state_path,
            {
                "schema_version": 1,
                "candidate_digest": candidate_digest,
                "engine_version": mica.engine_version,
                "target": target,
                "next_candidate_index": candidate_index + 1,
                "rejected": rejected,
                "accepted": accepted,
            },
        )
        if progress:
            progress(len(accepted), rejected, candidate.name)
    if len(accepted) != target:
        raise ValueError(
            f"candidate surplus produced only {len(accepted)} accepted identities; "
            f"{rejected} were rejected"
        )
    version_material = json.dumps(accepted, sort_keys=True, separators=(",", ":")).encode()
    gallery_version = f"commons-v3-{hashlib.sha256(version_material).hexdigest()[:16]}"
    manifest_payload = {"gallery_version": gallery_version, "identities": accepted}
    parse_manifest(manifest_payload, expected_identities=target)
    _write_json_atomic(manifest_path, manifest_payload)
    state_path.unlink(missing_ok=True)
    return {
        "accepted": len(accepted),
        "rejected": rejected,
        "gallery_version": gallery_version,
    }
