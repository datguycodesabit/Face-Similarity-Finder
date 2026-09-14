from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlencode, urlparse

from .commons_pack import WIKIDATA_PATTERN

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = (
    "FaceShapeStudio/0.1 "
    "(https://github.com/datguycodesabit/Face-Similarity-Finder; "
    "local personal-use Commons gallery curator)"
)


def _binding_value(row: Mapping[str, Any], name: str) -> str:
    field = row.get(name)
    return str(field.get("value", "")).strip() if isinstance(field, Mapping) else ""


def candidate_query(limit: int) -> str:
    if not 500 <= limit <= 5000:
        raise ValueError("candidate discovery limit must be between 500 and 5000")
    return f"""
SELECT ?person ?category ?image WHERE {{
  ?person wdt:P31 wd:Q5;
          wdt:P18 ?image;
          wdt:P373 ?category.
}}
LIMIT {limit}
""".strip()


def _fetch_sparql(query: str) -> Mapping[str, Any]:
    request = urllib.request.Request(
        f"{WIKIDATA_SPARQL}?{urlencode({'query': query, 'format': 'json'})}",
        headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, Mapping):
        raise ValueError("Wikidata returned a non-object response")
    return payload


def discover_candidates(
    limit: int = 2000,
    fetch: Callable[[str], Mapping[str, Any]] = _fetch_sparql,
) -> dict[str, Any]:
    """Discover a deterministic surplus; no candidate is accepted without local vision checks."""
    query = candidate_query(limit)
    payload = fetch(query)
    raw_rows = payload.get("results", {}).get("bindings", [])
    if not isinstance(raw_rows, list):
        raise ValueError("Wikidata response does not contain result bindings")
    raw_candidates: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        qid = _binding_value(row, "person").rsplit("/", 1)[-1].upper()
        name = _binding_value(row, "personLabel")
        category = _binding_value(row, "category")
        image_url = _binding_value(row, "image")
        if (
            qid in seen
            or not WIKIDATA_PATTERN.fullmatch(qid)
            or not category
            or urlparse(image_url).hostname != "commons.wikimedia.org"
        ):
            continue
        raw_candidates.append((qid, name, category, image_url))
        seen.add(qid)
    candidates: list[dict[str, Any]] = []
    for qid, name, category, image_url in raw_candidates:
        resolved_name = name or category
        if not resolved_name:
            continue
        candidates.append(
            {
                "name": resolved_name,
                "wikidata_id": qid,
                "commons_category": category,
                "primary_file": unquote(Path(urlparse(image_url).path).name).replace("_", " "),
            }
        )
    if len(candidates) < 500:
        raise ValueError(f"Wikidata produced only {len(candidates)} unique usable candidates")
    return {
        "schema_version": 1,
        "source": WIKIDATA_SPARQL,
        "query": query,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def write_candidates(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
