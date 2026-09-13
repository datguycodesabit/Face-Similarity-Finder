from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from face_match.config import Settings
from face_match.database import Database
from face_match.detector import MediaPipeDetector
from face_match.matcher import brute_force_oracle, rank_matches
from face_match.web import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify three held-out LFW-scale queries.")
    parser.add_argument("queries", nargs=3, type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    database = Database(settings.database_path)
    detector = MediaPipeDetector(settings.model_path)
    rows = database.indexed_rows()
    records = [(str(row["name"]), int(row["id"]), database.decode_vector(row)) for row in rows]
    app = create_app(settings, detector)
    with TestClient(app) as client:
        for path in args.queries:
            before = {item.resolve() for item in settings.project_root.rglob("*") if item.is_file()}
            with Image.open(path) as image:
                detection = detector.detect_one(image.convert("RGB"))
            ranked = rank_matches(database, detection.normalized)
            oracle = brute_force_oracle(detection.normalized, records)
            actual = [(item.identity, item.image_id, item.distance) for item in ranked]
            if actual != oracle:
                raise SystemExit(f"Ranking mismatch for {path}")
            if len({item.identity for item in ranked}) != 5:
                raise SystemExit(f"Non-unique results for {path}")

            with path.open("rb") as query:
                response = client.post(
                    "/api/match", files={"image": (path.name, query, "image/jpeg")}
                )
            if response.status_code != 200:
                raise SystemExit(f"API failed for {path}: {response.status_code} {response.text}")
            api_matches = response.json()["matches"]
            if len(api_matches) != 5 or len({item["identity"] for item in api_matches}) != 5:
                raise SystemExit(f"API did not return five unique results for {path}")
            api_pairs = [
                (item["identity"], int(item["image_url"].rsplit("/", 1)[1])) for item in api_matches
            ]
            expected_pairs = [(name, image_id) for name, image_id, _ in oracle]
            if api_pairs != expected_pairs:
                raise SystemExit(f"API ordering mismatch for {path}")
            api_distances = [float(item["distance"]) for item in api_matches]
            expected_distances = [round(distance, 8) for _, _, distance in oracle]
            if not np.allclose(api_distances, expected_distances, rtol=0.0, atol=1e-12):
                raise SystemExit(f"API distances differ from oracle for {path}")

            after = {item.resolve() for item in settings.project_root.rglob("*") if item.is_file()}
            if after != before:
                raise SystemExit(
                    f"Query processing changed project files for {path}: {after ^ before}"
                )
            print(f"PASS {path}: " + ", ".join(item.identity for item in ranked))


if __name__ == "__main__":
    main()
