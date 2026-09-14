from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from face_match.config import Settings
from face_match.detector import Detection, Detector, MediaPipeDetector
from face_match.geometry import aggregate_views, weighted_dense_distance
from face_match.mica import (
    MicaConfiguration,
    MicaEngine,
    MicaViewResult,
    MicaWorkerClient,
    aggregate_mica_views,
)
from face_match.web import create_app

VIEWS = ("front", "left", "right")
TRIPLETS = ("a", "b")


class RecordingDetector:
    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        self.model_version = detector.model_version
        self.results: list[Detection] = []

    def detect_one(self, image: Any) -> Detection:
        result = self.detector.detect_one(image)
        self.results.append(result)
        return result

    def close(self) -> None:
        close = getattr(self.detector, "close", None)
        if callable(close):
            close()


class RecordingMica:
    def __init__(self, engine: MicaEngine) -> None:
        self.engine = engine
        self.engine_version = engine.engine_version
        self.results: list[list[MicaViewResult]] = []

    def analyze(
        self,
        images: Sequence[bytes],
        *,
        validate_identity: bool = False,
        anchor_index: int = 0,
    ) -> list[MicaViewResult]:
        result = self.engine.analyze(
            images,
            validate_identity=validate_identity,
            anchor_index=anchor_index,
        )
        self.results.append(result)
        return result

    def close(self) -> None:
        self.engine.close()


def _triplet(identity: Path, name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for view in VIEWS:
        matches = [path for path in identity.glob(f"{name}-{view}.*") if path.is_file()]
        if len(matches) != 1:
            raise ValueError(f"{identity.name} needs exactly one {name}-{view} image")
        result[view] = matches[0]
    return result


def _post(client: TestClient, paths: dict[str, Path]) -> Any:
    with ExitStack() as stack:
        streams = {name: stack.enter_context(path.open("rb")) for name, path in paths.items()}
        return client.post(
            "/api/analyze",
            files={
                name: (paths[name].name, streams[name], "application/octet-stream")
                for name in VIEWS
            },
        )


def _pairwise(values: Sequence[Any], distance: Any) -> list[float]:
    return [distance(values[a], values[b]) for a, b in ((0, 1), (0, 2), (1, 2))]


def _jaccard(first: set[str], second: set[str]) -> float:
    return len(first & second) / len(first | second)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real held-out V3 pose, freshness, and same-person stability gates."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory with 5+ identity folders, each containing a/b front/left/right images.",
    )
    args = parser.parse_args()
    identities = sorted(path for path in args.root.iterdir() if path.is_dir())
    if len(identities) < 5:
        raise SystemExit("Held-out validation requires at least five identity folders")

    settings = Settings.from_env()
    detector = RecordingDetector(MediaPipeDetector(settings.model_path))
    mica = RecordingMica(MicaWorkerClient(MicaConfiguration.from_env(settings.project_root)))
    app = create_app(settings, detector, mica)
    rankings: dict[tuple[str, str], set[str]] = {}
    v2_views: list[list[np.ndarray]] = []
    mica_views: list[list[MicaViewResult]] = []

    with TestClient(app) as client:
        for identity in identities:
            for triplet_name in TRIPLETS:
                detector_start = len(detector.results)
                response = _post(client, _triplet(identity, triplet_name))
                if response.status_code != 200:
                    raise SystemExit(
                        f"{identity.name}/{triplet_name} failed: "
                        f"{response.status_code} {response.text}"
                    )
                payload = response.json()
                rankings[(identity.name, triplet_name)] = {
                    str(match["identity"]) for match in payload["matches"]
                }
                detections = detector.results[detector_start : detector_start + 3]
                v2_views.append([item.normalized for item in detections])
                mica_views.append(mica.results[-1])

    v2_within = [
        distance for views in v2_views for distance in _pairwise(views, weighted_dense_distance)
    ]
    v2_prototypes = [aggregate_views(views)[0] for views in v2_views]
    v2_between = [
        weighted_dense_distance(v2_prototypes[first], v2_prototypes[second])
        for first in range(len(v2_prototypes))
        for second in range(first + 1, len(v2_prototypes))
        if first // 2 != second // 2
    ]

    all_codes = np.stack([result.shape_code for views in mica_views for result in views])
    code_scale = np.maximum(
        1.4826 * np.median(np.abs(all_codes - np.median(all_codes, axis=0)), axis=0), 1e-6
    )

    def mica_distance(first: MicaViewResult, second: MicaViewResult) -> float:
        return float(np.sqrt(np.mean(((first.shape_code - second.shape_code) / code_scale) ** 2)))

    mica_within = [distance for views in mica_views for distance in _pairwise(views, mica_distance)]
    mica_prototypes = [aggregate_mica_views(views).shape_code for views in mica_views]
    mica_between = [
        float(
            np.sqrt(np.mean(((mica_prototypes[first] - mica_prototypes[second]) / code_scale) ** 2))
        )
        for first in range(len(mica_prototypes))
        for second in range(first + 1, len(mica_prototypes))
        if first // 2 != second // 2
    ]
    v2_pose_ratio = float(np.median(v2_within) / np.median(v2_between))
    v3_pose_ratio = float(np.median(mica_within) / np.median(mica_between))
    if v3_pose_ratio > 0.60 * v2_pose_ratio:
        raise SystemExit(
            f"Pose gate failed: V3 ratio {v3_pose_ratio:.4f} is not 40% below "
            f"V2 ratio {v2_pose_ratio:.4f}"
        )

    primary_lists = [rankings[(identity.name, "a")] for identity in identities]
    freshness = float(
        np.mean(
            [
                primary_lists[first] != primary_lists[second]
                for first in range(len(primary_lists))
                for second in range(first + 1, len(primary_lists))
            ]
        )
    )
    if freshness < 0.90:
        raise SystemExit(f"Ranking freshness gate failed: {freshness:.1%} < 90%")
    stability = float(
        np.mean(
            [
                _jaccard(rankings[(identity.name, "a")], rankings[(identity.name, "b")])
                for identity in identities
            ]
        )
    )
    if stability < 0.60:
        raise SystemExit(f"Same-person Jaccard gate failed: {stability:.3f} < 0.60")
    print(
        f"PASS identities={len(identities)} freshness={freshness:.1%} "
        f"Jaccard={stability:.3f} pose-ratio V3={v3_pose_ratio:.4f} "
        f"V2={v2_pose_ratio:.4f} improvement={1.0 - v3_pose_ratio / v2_pose_ratio:.1%}"
    )


if __name__ == "__main__":
    main()
