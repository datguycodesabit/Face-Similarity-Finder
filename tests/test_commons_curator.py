from __future__ import annotations

import json
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from face_match.commons_curator import (
    CommonsCandidate,
    CommonsSource,
    candidate_sources,
    curate_gallery,
    load_candidate_catalog,
)
from face_match.config import Settings

from .helpers import FakeDetector, FakeMicaEngine, dense_detection


def test_candidate_catalog_and_sources_filter_license_dimensions_and_attribution(
    tmp_path: Path,
) -> None:
    catalog = {
        "schema_version": 1,
        "candidates": [
            {
                "name": f"Person {index}",
                "wikidata_id": f"Q{index + 1}",
                "commons_category": f"Person {index}",
                "primary_file": f"Portrait {index}.jpg",
            }
            for index in range(500)
        ],
    }
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert len(load_candidate_catalog(path)) == 500

    def fetch(parameters: Mapping[str, str]) -> Mapping[str, object]:
        if parameters.get("list") == "categorymembers":
            return {"query": {"categorymembers": [{"title": "File:Side.jpg"}]}}
        assert "rvlimit" not in parameters
        assert parameters["iiurlwidth"] == "1440"
        pages = []
        for index, title in enumerate(parameters["titles"].split("|")):
            pages.append(
                {
                    "pageid": index + 10,
                    "title": title,
                    "revisions": [{"revid": index + 20}],
                    "imageinfo": [
                        {
                            "url": f"https://upload.wikimedia.org/test/{index}.jpg",
                            "thumburl": (
                                f"https://thumb.wikimedia.org/test/thumb/{index}.jpg"
                            ),
                            "mime": "image/jpeg",
                            "width": 1000,
                            "height": 1200,
                            "extmetadata": {
                                "Artist": {"value": "<b>Photographer</b>"},
                                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                "LicenseUrl": {
                                    "value": "https://creativecommons.org/licenses/by-sa/4.0/"
                                },
                            },
                        }
                    ],
                }
            )
        return {"query": {"pages": pages}}

    sources = candidate_sources(
        CommonsCandidate("Person", "Q1", "Person", "Portrait.jpg"), fetch=fetch
    )
    assert len(sources) == 2
    assert sources[0].primary
    assert sources[0].author == "Photographer"
    assert sources[0].license_id == "CC-BY-SA-4.0"
    assert sources[0].analysis_url == "https://thumb.wikimedia.org/test/thumb/0.jpg"


def test_real_curator_contract_resumes_after_interruption_and_writes_exact_manifest(
    tmp_path: Path,
) -> None:
    candidates = tuple(
        CommonsCandidate(f"Person {index}", f"Q{index + 1}", f"Person {index}", "front.jpg")
        for index in range(6)
    )
    payloads: dict[int, bytes] = {}
    for page_id in range(1, 19):
        stream = BytesIO()
        Image.new("RGB", (768, 768), (page_id * 7 % 255, 90, 130)).save(stream, "JPEG")
        payloads[page_id] = stream.getvalue()

    def sources_for(candidate: CommonsCandidate) -> tuple[CommonsSource, ...]:
        identity = int(candidate.wikidata_id[1:]) - 1
        return tuple(
            CommonsSource(
                page_id=identity * 3 + view + 1,
                revision_id=identity * 3 + view + 101,
                original_url=f"https://upload.wikimedia.org/test/{identity}-{view}.jpg",
                author="Fixture photographer",
                license_id="CC-BY-4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                width=768,
                height=768,
                mime="image/jpeg",
                primary=view == 0,
            )
            for view in range(3)
        )

    settings = Settings(
        tmp_path,
        tmp_path / "model.task",
        tmp_path / "v2.sqlite3",
        tmp_path / "lfw",
        v3_dataset_path=tmp_path / "commons",
    )
    calls = 0

    def interrupted_sources(candidate: CommonsCandidate) -> tuple[CommonsSource, ...]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return sources_for(candidate)

    def download(source: CommonsSource, maximum: int) -> bytes:
        del maximum
        return payloads[source.page_id]

    with pytest.raises(KeyboardInterrupt):
        curate_gallery(
            candidates,
            tmp_path / "commons",
            tmp_path / "manifest.json",
            FakeDetector(dense_detection()),
            FakeMicaEngine(),
            settings,
            target=5,
            sources_for=interrupted_sources,
            download=download,
        )
    assert (tmp_path / "commons" / ".curation-state.json").is_file()
    result = curate_gallery(
        candidates,
        tmp_path / "commons",
        tmp_path / "manifest.json",
        FakeDetector(dense_detection()),
        FakeMicaEngine(),
        settings,
        target=5,
        sources_for=sources_for,
        download=download,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert result["accepted"] == 5
    assert len(manifest["identities"]) == 5
    assert all(len(identity["images"]) == 3 for identity in manifest["identities"])
    assert not (tmp_path / "commons" / ".curation-state.json").exists()
