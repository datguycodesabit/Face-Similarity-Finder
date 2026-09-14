from __future__ import annotations

from face_match.commons_candidates import discover_candidates


def test_candidate_discovery_deduplicates_qids_and_preserves_curation_inputs() -> None:
    bindings = []
    for index in range(501):
        bindings.append(
            {
                "person": {"value": f"http://www.wikidata.org/entity/Q{index + 1}"},
                "personLabel": {"value": f"Person {index}"},
                "category": {"value": f"Person {index}"},
                "image": {
                    "value": f"http://commons.wikimedia.org/wiki/Special:FilePath/P{index}.jpg"
                },
            }
        )
    bindings.append(bindings[0])
    result = discover_candidates(
        2000,
        lambda _: {"results": {"bindings": bindings}},
    )
    assert result["candidate_count"] == 501
    assert result["candidates"][0] == {
        "name": "Person 0",
        "wikidata_id": "Q1",
        "commons_category": "Person 0",
        "primary_file": "P0.jpg",
    }
