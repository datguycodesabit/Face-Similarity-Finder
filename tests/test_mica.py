from __future__ import annotations

from pathlib import Path

import numpy as np

from face_match.mica import MicaConfiguration, MicaViewResult, aggregate_mica_views
from face_match.shape_v3 import COMPONENT_NAMES


def _result(value: float, outlier: bool = False) -> MicaViewResult:
    shape = np.asarray([value, value * 0.5, value * 0.25], dtype=np.float64)
    if outlier:
        shape += 20.0
    regions = {
        name: np.full((3, 3), value + index, dtype=np.float64)
        for index, name in enumerate(COMPONENT_NAMES)
    }
    if outlier:
        regions = {name: region + 20.0 for name, region in regions.items()}
    return MicaViewResult(shape, regions, 0.1)


def test_mica_configuration_reports_every_missing_readiness_requirement(tmp_path: Path) -> None:
    configuration = MicaConfiguration(
        home=tmp_path / "MICA",
        worker=tmp_path / "worker.py",
        model=tmp_path / "mica.tar",
        flame_model=tmp_path / "MICA" / "data" / "FLAME2020" / "generic_model.pkl",
        license_accepted=False,
    )
    problems = configuration.problems()
    assert len(problems) == 6
    assert "noncommercial MICA license" in problems[0]
    assert not configuration.ready


def test_mica_three_view_aggregation_rejects_one_shape_outlier() -> None:
    aggregate = aggregate_mica_views([_result(1.0), _result(1.01), _result(1.0, outlier=True)])
    assert aggregate.accepted_views == (0, 1)
    assert aggregate.rejected_view == 2
    assert np.allclose(aggregate.shape_code, np.asarray([1.005, 0.5025, 0.25125]))


def test_mica_three_view_aggregation_keeps_consistent_views() -> None:
    aggregate = aggregate_mica_views([_result(1.0), _result(1.01), _result(0.99)])
    assert aggregate.accepted_views == (0, 1, 2)
    assert aggregate.rejected_view is None
    assert np.allclose(aggregate.shape_code, np.asarray([1.0, 0.5, 0.25]))
