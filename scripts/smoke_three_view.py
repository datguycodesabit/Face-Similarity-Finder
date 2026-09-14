from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from face_match.config import Settings
from face_match.detector import Detection, Detector, MediaPipeDetector
from face_match.mica import (
    MicaConfiguration,
    MicaEngine,
    MicaViewResult,
    MicaWorkerClient,
    aggregate_mica_views,
)
from face_match.v3_database import V3Database
from face_match.v3_matcher import rank_v3_matches
from face_match.web import create_app


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


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(root: Path, database: Path) -> dict[str, str | None]:
    tracked_roots = (root / "src", root / "scripts", root / "tests")
    files = [
        path
        for tracked_root in tracked_roots
        if tracked_root.is_dir()
        for path in tracked_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    files.extend(path for path in (root / "README.md", root / "pyproject.toml") if path.is_file())
    snapshot = {str(path.relative_to(root)): _digest(path) for path in files}
    snapshot["<v3-database>"] = _digest(database)
    return snapshot


def _post(client: TestClient, paths: Mapping[str, Path]) -> Any:
    names = ("front", "left", "right")
    with ExitStack() as stack:
        streams = [stack.enter_context(paths[name].open("rb")) for name in names]
        return client.post(
            "/api/analyze",
            files={
                name: (paths[name].name, stream, "image/jpeg")
                for name, stream in zip(names, streams, strict=True)
            },
            data={
                "length": "any",
                "texture": "unsure",
                "effort": "moderate",
                "goal": "none",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real local V3 three-view smoke test.")
    parser.add_argument("front", type=Path)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    settings = Settings.from_env()
    database_path = (
        settings.v3_database_path or settings.project_root / "data" / "face_match_v3.sqlite3"
    )
    database = V3Database(database_path)
    if database.status()["prototypes"] < 5 or database.active_hybrid_calibration() is None:
        raise SystemExit("V3 prototype index/calibration is not ready; run `face-match index-v3`.")

    detector = RecordingDetector(MediaPipeDetector(settings.model_path))
    mica = RecordingMica(MicaWorkerClient(MicaConfiguration.from_env(settings.project_root)))
    app = create_app(settings, detector, mica)
    paths = {"front": args.front, "left": args.left, "right": args.right}
    before = _snapshot(settings.project_root, database_path)
    timings: list[float] = []
    response: Any = None
    with TestClient(app) as client:
        for _ in range(args.runs):
            started = time.perf_counter()
            response = _post(client, paths)
            timings.append(time.perf_counter() - started)
            if response.status_code != 200:
                raise SystemExit(f"V3 API failed: {response.status_code} {response.text}")

        duplicate = {"front": args.front, "left": args.front, "right": args.right}
        failure = _post(client, duplicate)
        if failure.status_code != 422:
            raise SystemExit(f"Expected duplicate-photo rejection; received {failure.status_code}")

    after = _snapshot(settings.project_root, database_path)
    if before != after:
        changed = sorted(
            key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
        )
        raise SystemExit(f"Privacy gate failed; query processing changed: {changed}")

    assert response is not None
    payload = response.json()
    matches = payload.get("matches", [])
    if len(matches) != 5 or len({item["identity"] for item in matches}) != 5:
        raise SystemExit("V3 API did not return five unique identity prototypes")
    if tuple(matches[0].get("breakdown", {})) != (
        "jaw_chin",
        "outline_cheeks",
        "global_proportions",
        "eye_brow_geometry",
        "nose_midface_geometry",
    ):
        raise SystemExit("V3 five-part score contract is incomplete")
    if any(not item.get("attribution", {}).get("commons_url") for item in matches):
        raise SystemExit("A V3 result is missing Commons attribution")

    aggregate = aggregate_mica_views(mica.results[args.runs - 1])
    query_detections = detector.results[(args.runs - 1) * 3 : args.runs * 3]
    accepted_measurements = [
        query_detections[index].v3_measurements for index in aggregate.accepted_views
    ]
    measurements = {
        name: float(np.median([sample[name] for sample in accepted_measurements]))
        for name in accepted_measurements[0]
    }
    calibration = database.active_hybrid_calibration()
    assert calibration is not None
    rank_started = time.perf_counter()
    direct = rank_v3_matches(database, aggregate, measurements, calibration)
    rank_elapsed = time.perf_counter() - rank_started
    direct_order = [item.identity for item in direct]
    api_order = [str(item["identity"]) for item in matches]
    if direct_order != api_order:
        raise SystemExit(
            f"API ranking is stale or differs from direct V3 ranking: {api_order} != {direct_order}"
        )

    median_elapsed = float(np.median(timings))
    if median_elapsed >= 20.0:
        raise SystemExit(f"Median MICA analysis exceeded 20 seconds: {median_elapsed:.3f}s")
    if rank_elapsed >= 1.0:
        raise SystemExit(f"V3 database ranking exceeded one second: {rank_elapsed:.3f}s")

    print(
        json.dumps(
            {
                "status": "PASS",
                "analysis_version": payload["analysis_version"],
                "engine": payload["engine"],
                "gallery_version": payload["gallery_version"],
                "median_analysis_seconds": round(median_elapsed, 3),
                "ranking_seconds": round(rank_elapsed, 4),
                "shape": [payload["shape"]["primary"], payload["shape"]["secondary"]],
                "matches": api_order,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
