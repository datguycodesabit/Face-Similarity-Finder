from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from .errors import IndexNotReadyError
from .geometry import display_similarity
from .mica import MicaAggregate
from .shape_v3 import COMPONENT_NAMES, HybridCalibration
from .v3_database import V3Database


@dataclass(frozen=True)
class V3Attribution:
    author: str
    license_id: str
    license_url: str
    commons_url: str


@dataclass(frozen=True)
class V3Match:
    rank: int
    identity_id: int
    identity: str
    wikidata_id: str
    image_id: int
    distance: float
    similarity: float
    breakdown: dict[str, float]
    attribution: V3Attribution


def rank_v3_matches(
    database: V3Database,
    query: MicaAggregate,
    query_descriptors: dict[str, float],
    calibration: HybridCalibration | None = None,
    limit: int = 5,
) -> list[V3Match]:
    active = calibration or database.active_hybrid_calibration()
    if active is None:
        raise IndexNotReadyError("V3 population calibration is missing. Run `face-match index-v3`.")
    scored: list[tuple[float, str, sqlite3.Row, dict[str, float]]] = []
    for row, regions in database.prototype_records():
        reference_descriptors = json.loads(str(row["descriptor_json"]))
        total, components = active.distance(
            query_descriptors,
            query.regions,
            reference_descriptors,
            regions,
        )
        scored.append((total, str(row["name"]), row, components))
    if len(scored) < limit:
        raise IndexNotReadyError(
            f"At least {limit} V3 identity prototypes are required; {len(scored)} are ready."
        )
    scored.sort(key=lambda item: (item[0], item[1].casefold()))
    matches: list[V3Match] = []
    for rank, (distance, name, row, components) in enumerate(scored[:limit], start=1):
        matches.append(
            V3Match(
                rank=rank,
                identity_id=int(row["identity_id"]),
                identity=name,
                wikidata_id=str(row["wikidata_id"]),
                image_id=int(row["display_image_id"]),
                distance=distance,
                similarity=display_similarity(distance),
                breakdown={component: components[component] for component in COMPONENT_NAMES},
                attribution=V3Attribution(
                    author=str(row["author"]),
                    license_id=str(row["license_id"]),
                    license_url=str(row["license_url"]),
                    commons_url=(
                        f"https://commons.wikimedia.org/?curid={int(row['commons_page_id'])}"
                    ),
                ),
            )
        )
    return matches
