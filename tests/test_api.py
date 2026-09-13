from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from face_match.config import Settings
from face_match.database import Database
from face_match.web import create_app

from .helpers import FakeDetector, SequenceDetector, dense_detection, fake_detector


def _png(size: tuple[int, int] = (120, 120)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, "#8b7064").save(stream, "PNG")
    return stream.getvalue()


def _gif() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (120, 120), "#8b7064").save(stream, "GIF")
    return stream.getvalue()


def _pattern_png(seed: int) -> bytes:
    stream = BytesIO()
    image = Image.new("RGB", (120, 120), "#d6b195")
    draw = ImageDraw.Draw(image)
    for offset in range(seed + 1):
        x = 8 + offset * 13
        draw.rectangle((x, 0, x + 5, 119), fill=(30 + seed * 20, 55, 80))
    image.save(stream, "PNG")
    return stream.getvalue()


def _file_snapshot(root: Path) -> dict[Path, str]:
    return {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _client(tmp_path: Path, face_count: int = 1, identities: int = 6) -> TestClient:
    dataset = tmp_path / "lfw"
    dataset.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        tmp_path, tmp_path / "model.task", tmp_path / "index.sqlite3", dataset, 150_000, 500, 8
    )
    detector = fake_detector()
    detector.face_count = face_count
    database = Database(settings.database_path)
    database.initialize()
    query = detector.result.normalized
    for index in range(identities):
        name = f"Reference {index}"
        path = dataset / f"reference_{index}.png"
        path.write_bytes(_png())
        image_id = database.register_image(name, path.name, detector.model_version)
        database.save_vector(image_id, query + index * 0.01, [[0.5, 0.5]], detector.model_version)
    database.set_metadata("dataset_root", str(dataset))
    database.set_metadata("indexing_status", "complete")
    return TestClient(create_app(settings, detector))


def _analysis_client(
    tmp_path: Path,
    poses: tuple[float, float, float] = (0, -22, 22),
    pitches: tuple[float, float, float] = (0, 0, 0),
) -> TestClient:
    dataset = tmp_path / "lfw"
    dataset.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        tmp_path, tmp_path / "model.task", tmp_path / "index.sqlite3", dataset, 150_000, 500, 8
    )
    detector = SequenceDetector(
        [
            dense_detection(40, poses[0], pitches[0]),
            dense_detection(41, poses[1], pitches[1]),
            dense_detection(42, poses[2], pitches[2]),
        ]
    )
    database = Database(settings.database_path)
    database.initialize()
    for index in range(6):
        detection = dense_detection(50 + index)
        path = dataset / f"reference_{index}.png"
        path.write_bytes(_pattern_png(index))
        image_id = database.register_image(f"Reference {index}", path.name, detector.model_version)
        database.save_vector(
            image_id,
            detection.normalized,
            detection.overlay,
            detector.model_version,
            detection.measurements,
        )
    database.set_metadata("dataset_root", str(dataset))
    database.set_metadata("indexing_status", "complete")
    database.set_metadata("landmark_version", "structural-v2-dense-weighted-three-view")
    return TestClient(create_app(settings, detector))


def test_valid_match_returns_exactly_five_ordered_distinct_results(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post("/api/match", files={"image": ("face.png", _png(), "image/png")})
        status = client.get("/api/status").json()
    assert response.status_code == 200
    matches = response.json()["matches"]
    assert len(matches) == 5
    assert len({match["identity"] for match in matches}) == 5
    assert [match["distance"] for match in matches] == sorted(
        match["distance"] for match in matches
    )
    expected = [round(float(np.sqrt(3) * index * 0.01), 8) for index in range(5)]
    assert [match["distance"] for match in matches] == expected
    assert [match["identity"] for match in matches] == [f"Reference {index}" for index in range(5)]
    assert status["model_ready"] and status["index_ready"] and status["local_only"]


def test_incompatible_landmark_version_requires_reindex(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        database = cast(FastAPI, client.app).state.database
        database.set_metadata("landmark_version", "obsolete-v0")
        status = client.get("/api/status")
        response = client.post("/api/match", files={"image": ("face.png", _png(), "image/png")})
    assert status.json()["index_ready"] is False
    assert response.status_code == 409
    assert "incompatible" in response.json()["error"]


def test_upload_rejections_and_incomplete_index(tmp_path: Path) -> None:
    with _client(tmp_path / "corrupt") as client:
        before = _file_snapshot(tmp_path)
        assert (
            client.post(
                "/api/match", files={"image": ("bad.jpg", b"bad", "image/jpeg")}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/match", files={"image": ("small.png", _png((4, 4)), "image/png")}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/match", files={"image": ("huge.png", b"x" * 150_001, "image/png")}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/match", files={"image": ("valid.gif", _gif(), "image/gif")}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/match", files={"image": ("wide.png", _png((501, 501)), "image/png")}
            ).status_code
            == 422
        )
        after = _file_snapshot(tmp_path)
        assert after == before
    with _client(tmp_path / "few", identities=4) as client:
        assert (
            client.post(
                "/api/match", files={"image": ("face.png", _png(), "image/png")}
            ).status_code
            == 409
        )


def test_zero_and_multiple_faces_are_actionable(tmp_path: Path) -> None:
    for count, expected in [(0, "No face"), (2, "Found 2 faces")]:
        with _client(tmp_path / str(count), face_count=count) as client:
            response = client.post("/api/match", files={"image": ("face.png", _png(), "image/png")})
        assert response.status_code == 422
        assert expected in response.json()["error"]


def test_query_processing_leaves_no_files(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        before = _file_snapshot(tmp_path)
        response = client.post("/api/match", files={"image": ("private.png", _png(), "image/png")})
        after = _file_snapshot(tmp_path)
    assert response.status_code == 200
    assert after == before
    assert not any("private" in str(path) for path in after)


def test_three_view_analysis_returns_shape_matches_guidance_and_diagnostics(tmp_path: Path) -> None:
    with _analysis_client(tmp_path) as client:
        before = _file_snapshot(tmp_path)
        response = client.post(
            "/api/analyze",
            files={
                "front": ("front.png", _pattern_png(1), "image/png"),
                "left": ("left.png", _pattern_png(3), "image/png"),
                "right": ("right.png", _pattern_png(6), "image/png"),
            },
            data={"length": "medium", "texture": "wavy", "effort": "moderate", "goal": "add-width"},
        )
        after = _file_snapshot(tmp_path)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["matches"]) == 5
    assert len({match["identity"] for match in payload["matches"]}) == 5
    assert len(payload["recommendations"]) == 3
    assert len(payload["diagnostics"]) == 3
    assert payload["shape"]["primary"] != payload["shape"]["secondary"]
    assert "silhouette" in payload["matches"][0]["breakdown"]
    assert response.headers["cache-control"] == "no-store"
    assert after == before


def test_three_view_analysis_changes_when_the_submitted_faces_change(tmp_path: Path) -> None:
    with _analysis_client(tmp_path) as client:
        first = client.post(
            "/api/analyze",
            files={
                "front": ("first-front.png", _pattern_png(1), "image/png"),
                "left": ("first-left.png", _pattern_png(3), "image/png"),
                "right": ("first-right.png", _pattern_png(6), "image/png"),
            },
        )
        cast(FastAPI, client.app).state.detector = SequenceDetector(
            [dense_detection(90, 0), dense_detection(91, -22), dense_detection(92, 22)]
        )
        second = client.post(
            "/api/analyze",
            files={
                "front": ("second-front.png", _pattern_png(2), "image/png"),
                "left": ("second-left.png", _pattern_png(5), "image/png"),
                "right": ("second-right.png", _pattern_png(8), "image/png"),
            },
        )
    assert first.status_code == second.status_code == 200
    first_payload, second_payload = first.json(), second.json()
    assert first_payload["shape"]["measurements"] != second_payload["shape"]["measurements"]
    assert [item["identity"] for item in first_payload["matches"]] != [
        item["identity"] for item in second_payload["matches"]
    ]


def test_three_view_analysis_rejects_duplicate_and_incorrect_pose_photos(tmp_path: Path) -> None:
    duplicate = _pattern_png(2)
    with _analysis_client(tmp_path / "duplicate") as client:
        before = _file_snapshot(tmp_path)
        response = client.post(
            "/api/analyze",
            files={
                "front": ("front.png", duplicate, "image/png"),
                "left": ("left.png", duplicate, "image/png"),
                "right": ("right.png", _pattern_png(7), "image/png"),
            },
        )
        after = _file_snapshot(tmp_path)
    assert response.status_code == 422
    assert "three different photos" in response.json()["error"]
    assert after == before

    cases = [
        ((18, -22, 22), (0, 0, 0), "straight-ahead"),
        ((0, -40, 22), (0, 0, 0), "15–35°"),
        ((0, -22, -20), (0, 0, 0), "opposite directions"),
        ((0, -22, 22), (13, 0, 0), "tilted too far"),
    ]
    for index, (poses, pitches, expected) in enumerate(cases):
        with _analysis_client(tmp_path / f"pose-{index}", poses, pitches) as client:
            before = _file_snapshot(tmp_path)
            response = client.post(
                "/api/analyze",
                files={
                    "front": ("front.png", _pattern_png(1), "image/png"),
                    "left": ("left.png", _pattern_png(3), "image/png"),
                    "right": ("right.png", _pattern_png(7), "image/png"),
                },
            )
            after = _file_snapshot(tmp_path)
        assert response.status_code == 422
        assert expected in response.json()["error"]
        assert after == before


def test_three_view_analysis_discards_invalid_preferences_and_face_count_failures(
    tmp_path: Path,
) -> None:
    files = {
        "front": ("front.png", _pattern_png(1), "image/png"),
        "left": ("left.png", _pattern_png(3), "image/png"),
        "right": ("right.png", _pattern_png(7), "image/png"),
    }
    with _analysis_client(tmp_path / "preference") as client:
        before = _file_snapshot(tmp_path)
        response = client.post("/api/analyze", files=files, data={"texture": "invented"})
        after = _file_snapshot(tmp_path)
    assert response.status_code == 422
    assert "Unsupported texture preference" in response.json()["error"]
    assert after == before

    for count, expected in ((0, "No face"), (2, "Found 2 faces")):
        with _analysis_client(tmp_path / f"faces-{count}") as client:
            cast(FastAPI, client.app).state.detector = FakeDetector(
                dense_detection(60), face_count=count, model_version="fake-v2"
            )
            before = _file_snapshot(tmp_path)
            response = client.post("/api/analyze", files=files)
            after = _file_snapshot(tmp_path)
        assert response.status_code == 422
        assert expected in response.json()["error"]
        assert after == before
