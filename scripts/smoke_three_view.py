from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from PIL import Image

from face_match.config import Settings
from face_match.database import Database
from face_match.detector import Detection, MediaPipeDetector
from face_match.geometry import (
    DENSE_WEIGHTS,
    aggregate_views,
    descriptor_vector,
    rms_distance,
    shape_measurements,
    weighted_dense_distance,
)
from face_match.matcher import rank_shape_matches
from face_match.web import create_app


class RecordingDetector:
    def __init__(self, detector: MediaPipeDetector) -> None:
        self.detector = detector
        self.model_version = detector.model_version
        self.results: list[Detection] = []

    def detect_one(self, image: Image.Image) -> Detection:
        result = self.detector.detect_one(image)
        self.results.append(result)
        return result

    def close(self) -> None:
        self.detector.close()


def _oracle_distance(
    query: NDArray[np.float64],
    query_measurements: Mapping[str, float],
    reference: NDArray[np.float64],
    reference_measurements: Mapping[str, float],
) -> float:
    moving = reference - np.average(reference, axis=0, weights=DENSE_WEIGHTS)
    target = query - np.average(query, axis=0, weights=DENSE_WEIGHTS)
    left, _, right = np.linalg.svd((moving * DENSE_WEIGHTS[:, None]).T @ target)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    silhouette = float(
        np.sqrt(
            np.average(np.sum((target - moving @ rotation) ** 2, axis=1), weights=DENSE_WEIGHTS)
        )
    )
    descriptor = float(
        np.sqrt(
            np.mean(
                (descriptor_vector(query_measurements) - descriptor_vector(reference_measurements))
                ** 2
            )
        )
    )
    return 0.7 * descriptor + 0.3 * silhouette


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real three-view v2 acceptance check.")
    parser.add_argument("front", type=Path)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    settings = Settings.from_env()
    detector = RecordingDetector(MediaPipeDetector(settings.model_path))
    database = Database(settings.database_path)
    app = create_app(settings, detector)
    before = {path.resolve() for path in settings.project_root.rglob("*") if path.is_file()}
    timings: list[float] = []
    with TestClient(app) as client:
        for _ in range(3):
            started = time.perf_counter()
            with (
                args.front.open("rb") as front,
                args.left.open("rb") as left,
                args.right.open("rb") as right,
            ):
                response = client.post(
                    "/api/analyze",
                    files={
                        "front": (args.front.name, front, "image/jpeg"),
                        "left": (args.left.name, left, "image/jpeg"),
                        "right": (args.right.name, right, "image/jpeg"),
                    },
                    data={
                        "length": "medium",
                        "texture": "wavy",
                        "effort": "moderate",
                        "goal": "add-width",
                    },
                )
            timings.append(time.perf_counter() - started)
            if response.status_code != 200:
                raise SystemExit(f"Three-view API failed: {response.status_code} {response.text}")
    median_elapsed = float(np.median(timings))
    after = {path.resolve() for path in settings.project_root.rglob("*") if path.is_file()}
    payload = response.json()
    if len(payload["matches"]) != 5 or len({item["identity"] for item in payload["matches"]}) != 5:
        raise SystemExit("Three-view API did not return five unique identities")
    if len(payload["recommendations"]) != 3 or len(payload["diagnostics"]) != 3:
        raise SystemExit("Shape guidance or diagnostics are incomplete")
    triplet = detector.results[:3]
    legacy_vectors = [item.legacy_normalized for item in triplet]
    if any(vector is None for vector in legacy_vectors):
        raise SystemExit("Real detector did not expose the legacy comparison vectors")
    legacy = [vector for vector in legacy_vectors if vector is not None]
    pairs = ((0, 1), (0, 2), (1, 2))
    v1_instability = float(np.mean([rms_distance(legacy[a], legacy[b]) for a, b in pairs]))
    v2_instability = float(
        np.mean(
            [
                weighted_dense_distance(triplet[a].normalized, triplet[b].normalized)
                for a, b in pairs
            ]
        )
    )
    if v2_instability >= v1_instability:
        raise SystemExit(
            f"V2 did not improve held-out pose stability: {v2_instability:.6f} "
            f">= legacy {v1_instability:.6f}"
        )
    query_vector, _ = aggregate_views([item.normalized for item in triplet])
    query_measurements = shape_measurements(query_vector)
    direct = rank_shape_matches(database, query_vector, query_measurements)
    best: dict[str, tuple[float, int]] = {}
    for row in database.indexed_rows():
        distance = _oracle_distance(
            query_vector,
            query_measurements,
            database.decode_vector(row),
            json.loads(row["descriptor_json"]),
        )
        name, image_id = str(row["name"]), int(row["id"])
        if name not in best or (distance, image_id) < best[name]:
            best[name] = (distance, image_id)
    oracle = sorted(
        ((name, image_id, distance) for name, (distance, image_id) in best.items()),
        key=lambda item: (item[2], item[0]),
    )[:5]
    api = [
        (
            item["identity"],
            int(item["image_url"].rsplit("/", 1)[1]),
            float(item["distance"]),
        )
        for item in payload["matches"]
    ]
    expected = [(name, image_id, round(distance, 8)) for name, image_id, distance in oracle]
    direct_result = [(item.identity, item.image_id, round(item.distance, 8)) for item in direct]
    if api != direct_result:
        raise SystemExit(f"API ranking differs from direct matcher: {api} != {direct_result}")
    if [(name, image_id) for name, image_id, _ in api] != [
        (name, image_id) for name, image_id, _ in expected
    ] or not np.allclose(
        [distance for _, _, distance in api],
        [distance for _, _, distance in expected],
        rtol=0.0,
        atol=5e-6,
    ):
        raise SystemExit(f"API ranking differs from independent oracle: {api} != {expected}")
    if median_elapsed >= 5.0:
        raise SystemExit(f"Median analysis exceeded the five-second target: {median_elapsed:.3f}s")
    if before != after:
        raise SystemExit(f"Query analysis changed project files: {before ^ after}")
    names = ", ".join(item["identity"] for item in payload["matches"])
    print(
        f"PASS median {median_elapsed:.3f}s · "
        f"{payload['shape']['primary']}/{payload['shape']['secondary']} "
        f"· {payload['shape']['three_view_agreement']:.1f}% agreement · "
        f"stability v2 {v2_instability:.4f} < v1 {v1_instability:.4f} · {names}"
    )


if __name__ == "__main__":
    main()
