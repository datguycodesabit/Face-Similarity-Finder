from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .database import Database
from .errors import IndexNotReadyError
from .geometry import combined_shape_distance, display_similarity, rms_distance


@dataclass(frozen=True)
class Match:
    rank: int
    identity: str
    image_id: int
    distance: float
    similarity: float
    overlay: list[list[float]]


@dataclass(frozen=True)
class ShapeMatch(Match):
    silhouette_distance: float
    proportion_distance: float


def rank_matches(database: Database, query: NDArray[np.float64], limit: int = 5) -> list[Match]:
    best: dict[str, tuple[float, int, list[list[float]]]] = {}
    for row in database.indexed_rows():
        distance = rms_distance(query, database.decode_vector(row))
        current = best.get(str(row["name"]))
        if current is None or (distance, int(row["id"])) < (current[0], current[1]):
            best[str(row["name"])] = (
                distance,
                int(row["id"]),
                json.loads(row["overlay_json"]),
            )
    if len(best) < limit:
        raise IndexNotReadyError(
            f"At least {limit} indexed identities are required; {len(best)} are ready."
        )
    ordered = sorted(best.items(), key=lambda item: (item[1][0], item[0]))[:limit]
    return [
        Match(
            rank=rank,
            identity=name,
            image_id=values[1],
            distance=values[0],
            similarity=display_similarity(values[0]),
            overlay=values[2],
        )
        for rank, (name, values) in enumerate(ordered, start=1)
    ]


def rank_shape_matches(
    database: Database,
    query: NDArray[np.float64],
    measurements: dict[str, float],
    limit: int = 5,
) -> list[ShapeMatch]:
    best: dict[str, tuple[float, int, float, float, list[list[float]]]] = {}
    for row in database.indexed_rows():
        stored_measurements = json.loads(row["descriptor_json"] or "{}")
        if not stored_measurements:
            continue
        total, silhouette, proportions = combined_shape_distance(
            query,
            measurements,
            database.decode_vector(row),
            stored_measurements,
        )
        current = best.get(str(row["name"]))
        if current is None or (total, int(row["id"])) < (current[0], current[1]):
            best[str(row["name"])] = (
                total,
                int(row["id"]),
                silhouette,
                proportions,
                json.loads(row["overlay_json"]),
            )
    if len(best) < limit:
        raise IndexNotReadyError(
            f"At least {limit} eligible v2 identities are required; {len(best)} are ready."
        )
    ordered = sorted(best.items(), key=lambda item: (item[1][0], item[0]))[:limit]
    return [
        ShapeMatch(
            rank=rank,
            identity=name,
            image_id=values[1],
            distance=values[0],
            similarity=display_similarity(values[0]),
            overlay=values[4],
            silhouette_distance=values[2],
            proportion_distance=values[3],
        )
        for rank, (name, values) in enumerate(ordered, start=1)
    ]


def brute_force_oracle(
    query: NDArray[np.float64], records: list[tuple[str, int, NDArray[np.float64]]], limit: int = 5
) -> list[tuple[str, int, float]]:
    best: dict[str, tuple[int, float]] = {}
    for name, image_id, vector in records:
        distance = float(np.sqrt(np.mean(np.sum((query - vector) ** 2, axis=1))))
        if name not in best or (distance, image_id) < (best[name][1], best[name][0]):
            best[name] = (image_id, distance)
    return [
        (name, image_id, distance)
        for name, (image_id, distance) in sorted(
            best.items(), key=lambda item: (item[1][1], item[0])
        )[:limit]
    ]
