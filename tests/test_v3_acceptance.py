from __future__ import annotations

import json
import time
from typing import cast

import numpy as np

from face_match.mica import MicaAggregate
from face_match.shape_v3 import (
    COMPONENT_NAMES,
    DESCRIPTOR_NAMES,
    HybridCalibration,
    maximum_descriptor_correlation,
)
from face_match.v3_database import V3Database
from face_match.v3_matcher import rank_v3_matches


def _descriptors(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {name: float(rng.normal(1.0, 0.12)) for name in DESCRIPTOR_NAMES}


def _regions(seed: int) -> dict[str, np.ndarray]:
    common = np.random.default_rng(1).normal(size=(18, 3)) * 0.1
    variation = np.random.default_rng(seed).normal(size=(18, 3)) * 0.012
    return {name: common + variation * (index + 1) for index, name in enumerate(COMPONENT_NAMES)}


class AcceptanceDatabase:
    def __init__(self, count: int, calibration: HybridCalibration) -> None:
        self.calibration = calibration
        self.rows = []
        self.regions = {}
        for index in range(count):
            self.regions[index] = _regions(index + 10)
            self.rows.append(
                {
                    "identity_id": index,
                    "name": f"Person {index:03d}",
                    "wikidata_id": f"Q{index + 1}",
                    "display_image_id": index + 1,
                    "commons_page_id": index + 1000,
                    "descriptor_json": json.dumps(_descriptors(index + 10)),
                    "author": "Test",
                    "license_id": "CC0-1.0",
                    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "original_url": f"https://upload.wikimedia.org/test/{index}.jpg",
                    "photo_count": 3,
                }
            )

    def active_hybrid_calibration(self) -> HybridCalibration:
        return self.calibration

    def prototype_records(self) -> list[tuple[dict[str, object], dict[str, np.ndarray]]]:
        return [(row, self.regions[int(str(row["identity_id"]))]) for row in self.rows]


def _fixture(count: int = 30) -> tuple[AcceptanceDatabase, HybridCalibration]:
    descriptors = [_descriptors(index + 10) for index in range(count)]
    regions = [_regions(index + 10) for index in range(count)]
    calibration = HybridCalibration.fit(descriptors, regions)
    return AcceptanceDatabase(count, calibration), calibration


def _query(index: int, noise: int | None = None) -> tuple[MicaAggregate, dict[str, float]]:
    descriptors = _descriptors(index + 10)
    regions = _regions(index + 10)
    if noise is not None:
        rng = np.random.default_rng(noise)
        descriptors = {
            name: value + float(rng.normal(0, 0.002)) for name, value in descriptors.items()
        }
        regions = {
            name: value + rng.normal(0, 0.0002, value.shape) for name, value in regions.items()
        }
    return MicaAggregate(np.ones(4), regions, (0, 1, 2), None), descriptors


def test_descriptor_independence_and_ranking_freshness_gates() -> None:
    database, calibration = _fixture()
    samples = [_descriptors(index + 10) for index in range(30)]
    assert maximum_descriptor_correlation(samples) < 0.95
    top_fives = []
    for index in range(20):
        query, descriptors = _query(index)
        top_fives.append(
            tuple(
                match.identity
                for match in rank_v3_matches(
                    cast(V3Database, database), query, descriptors, calibration
                )
            )
        )
    comparisons = [
        top_fives[first] != top_fives[second]
        for first in range(len(top_fives))
        for second in range(first + 1, len(top_fives))
    ]
    assert np.mean(comparisons) >= 0.90


def test_same_person_top_five_stability_and_photo_count_bias_gates() -> None:
    database, calibration = _fixture()
    sets = []
    for noise in (101, 102, 103, 104):
        query, descriptors = _query(7, noise)
        sets.append(
            {
                match.identity
                for match in rank_v3_matches(
                    cast(V3Database, database), query, descriptors, calibration
                )
            }
        )
    jaccards = [
        len(sets[first] & sets[second]) / len(sets[first] | sets[second])
        for first in range(len(sets))
        for second in range(first + 1, len(sets))
    ]
    assert np.mean(jaccards) >= 0.60

    query, descriptors = _query(7)
    before = rank_v3_matches(cast(V3Database, database), query, descriptors, calibration)
    database.rows[7]["photo_count"] = 300
    after = rank_v3_matches(cast(V3Database, database), query, descriptors, calibration)
    assert [(item.identity, item.distance) for item in before] == [
        (item.identity, item.distance) for item in after
    ]


def test_ranking_500_prototypes_finishes_under_one_second() -> None:
    _, calibration = _fixture()
    database = AcceptanceDatabase(500, calibration)
    query, descriptors = _query(12)
    started = time.perf_counter()
    matches = rank_v3_matches(cast(V3Database, database), query, descriptors, calibration)
    elapsed = time.perf_counter() - started
    assert len(matches) == 5
    assert elapsed < 1.0
