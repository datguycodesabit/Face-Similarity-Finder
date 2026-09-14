from __future__ import annotations

import numpy as np

from face_match.mica import MicaAggregate
from face_match.shape_v3 import COMPONENT_NAMES, HybridCalibration, shape_descriptors
from face_match.v3_matcher import rank_v3_matches

from .helpers import dense_points


def _regions(value: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(700)
    base = rng.normal(size=(8, 3))
    return {
        name: base + np.array([0.0, index * 0.2, value * (index + 1)])
        for index, name in enumerate(COMPONENT_NAMES)
    }


class FakeV3Database:
    def __init__(self, calibration: HybridCalibration) -> None:
        self.calibration = calibration
        self.regions = {index: _regions(index * 0.01) for index in range(6)}
        self.rows = [
            {
                "identity_id": index,
                "name": f"Person {index}",
                "wikidata_id": f"Q{index + 1}",
                "display_image_id": index + 100,
                "commons_page_id": index + 1000,
                "descriptor_json": __import__("json").dumps(
                    shape_descriptors(dense_points(800 + index))
                ),
                "author": f"Author {index}",
                "license_id": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "original_url": f"https://upload.wikimedia.org/person-{index}.jpg",
            }
            for index in range(6)
        ]

    def active_hybrid_calibration(self) -> HybridCalibration:
        return self.calibration

    def prototype_rows(self) -> list[dict[str, object]]:
        return self.rows

    def prototype_regions(self, identity_id: int) -> dict[str, np.ndarray]:
        return self.regions[identity_id]

    def prototype_records(self) -> list[tuple[dict[str, object], dict[str, np.ndarray]]]:
        return [(row, self.regions[int(row["identity_id"])]) for row in self.rows]


def test_v3_ranking_uses_one_prototype_per_identity_and_returns_attribution() -> None:
    descriptors = [shape_descriptors(dense_points(800 + index)) for index in range(6)]
    regions = [_regions(index * 0.01) for index in range(6)]
    calibration = HybridCalibration.fit(descriptors, regions)
    database = FakeV3Database(calibration)
    query = MicaAggregate(
        shape_code=np.ones(2),
        regions=regions[2],
        accepted_views=(0, 1, 2),
        rejected_view=None,
    )
    matches = rank_v3_matches(database, query, descriptors[2])  # type: ignore[arg-type]
    assert len(matches) == 5
    assert matches[0].identity == "Person 2"
    assert np.isclose(matches[0].distance, 0.0)
    assert matches[0].attribution.author == "Author 2"
    assert matches[0].attribution.commons_url == "https://commons.wikimedia.org/?curid=1002"
    assert tuple(matches[0].breakdown) == COMPONENT_NAMES
