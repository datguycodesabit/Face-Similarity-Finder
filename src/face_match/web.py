from __future__ import annotations

import mimetypes
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from .config import Settings
from .database import Database
from .detector import Detector, MediaPipeDetector
from .errors import FaceMatchError, IndexNotReadyError, InvalidImageError
from .geometry import (
    LANDMARK_VERSION,
    aggregate_views,
    shape_measurements,
    shape_memberships,
)
from .images import decode_image
from .matcher import rank_matches, rank_shape_matches
from .recommendations import CATALOG_VERSION, recommend_haircuts, validate_preferences

PACKAGE_DIR = Path(__file__).parent


def _difference_hash(image: Image.Image) -> int:
    pixels = np.asarray(image.convert("L").resize((9, 8)), dtype=np.uint8)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return sum(int(bit) << position for position, bit in enumerate(bits))


def _validate_three_views(images: dict[str, Image.Image], detections: dict[str, Any]) -> None:
    hashes = [_difference_hash(images[name]) for name in ("front", "left", "right")]
    if any((hashes[a] ^ hashes[b]).bit_count() <= 3 for a, b in ((0, 1), (0, 2), (1, 2))):
        raise InvalidImageError("The front, left, and right slots need three different photos.")
    for name, detection in detections.items():
        if abs(detection.pitch) > 12.0:
            raise InvalidImageError(
                f"The {name} photo is tilted too far up or down ({detection.pitch:.1f}°)."
            )
    if abs(detections["front"].yaw) > 10.0:
        raise InvalidImageError(
            "The front photo needs a straight-ahead pose; detected "
            f"{detections['front'].yaw:.1f}° yaw."
        )
    left_yaw, right_yaw = detections["left"].yaw, detections["right"].yaw
    if not 15.0 <= abs(left_yaw) <= 35.0:
        raise InvalidImageError("The left photo should be a 15–35° three-quarter view.")
    if not 15.0 <= abs(right_yaw) <= 35.0:
        raise InvalidImageError("The right photo should be a 15–35° three-quarter view.")
    if left_yaw * right_yaw >= 0:
        raise InvalidImageError("The left and right photos need to face opposite directions.")


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    active_settings = settings or Settings.from_env()
    database = Database(active_settings.database_path)
    database.initialize()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.detector = detector
        yield
        current = getattr(app.state, "detector", None)
        close = getattr(current, "close", None)
        if callable(close):
            close()

    app = FastAPI(
        title="Face Structure Finder",
        description="Local landmark-geometry comparison; not identity verification.",
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    app.state.database = database
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")

    @app.middleware("http")
    async def prevent_api_caching(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(FaceMatchError)
    async def face_match_error(_: Request, exc: FaceMatchError) -> JSONResponse:
        status = 409 if isinstance(exc, IndexNotReadyError) else 422
        return JSONResponse({"error": str(exc)}, status_code=status)

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "max_upload_mb": active_settings.max_upload_bytes // 1024 // 1024,
                "max_dimension": active_settings.max_dimension,
                "min_dimension": active_settings.min_dimension,
            },
        )

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        counts = database.status()
        metadata = database.metadata()
        version_compatible = metadata.get("landmark_version", LANDMARK_VERSION) == LANDMARK_VERSION
        indexing_complete = metadata.get("indexing_status", "complete") == "complete"
        return {
            "model_ready": active_settings.model_path.is_file() or detector is not None,
            "index_ready": counts["identities"] >= 5 and version_compatible and indexing_complete,
            "version_compatible": version_compatible,
            "indexing_status": metadata.get("indexing_status", "not_started"),
            "landmark_version": LANDMARK_VERSION,
            "counts": counts,
            "local_only": True,
        }

    @app.post("/api/match")
    async def match(image: UploadFile = File(...)) -> Any:
        try:
            payload = await image.read(active_settings.max_upload_bytes + 1)
        finally:
            await image.close()
        decoded = decode_image(payload, active_settings)
        metadata = database.metadata()
        if metadata.get("indexing_status") == "running":
            raise IndexNotReadyError(
                "The local index is still building. Try again when it completes."
            )
        if metadata.get("landmark_version", LANDMARK_VERSION) != LANDMARK_VERSION:
            raise IndexNotReadyError(
                "The stored landmark vectors are incompatible. Re-run the index command."
            )
        active_detector = getattr(app.state, "detector", None)
        if active_detector is None:
            try:
                active_detector = MediaPipeDetector(active_settings.model_path)
            except FileNotFoundError as error:
                raise IndexNotReadyError(str(error)) from error
            app.state.detector = active_detector
        detection = active_detector.detect_one(decoded)
        results = rank_matches(database, detection.normalized)
        return {
            "notice": "Structural similarity only—not identity verification or a probability.",
            "query_overlay": detection.overlay,
            "matches": [
                {
                    "rank": result.rank,
                    "identity": result.identity,
                    "image_url": f"/api/images/{result.image_id}",
                    "distance": round(result.distance, 8),
                    "similarity": result.similarity,
                    "similarity_label": "Derived structural similarity (not a probability)",
                    "overlay": result.overlay,
                }
                for result in results
            ],
        }

    @app.post("/api/analyze")
    async def analyze(
        front: UploadFile = File(...),
        left: UploadFile = File(...),
        right: UploadFile = File(...),
        length: str = Form("any"),
        texture: str = Form("unsure"),
        effort: str = Form("moderate"),
        goal: str = Form("none"),
    ) -> Any:
        uploads = {"front": front, "left": left, "right": right}
        try:
            payloads = {
                name: await upload.read(active_settings.max_upload_bytes + 1)
                for name, upload in uploads.items()
            }
        finally:
            for upload in uploads.values():
                await upload.close()
        try:
            preferences = validate_preferences(
                {"length": length, "texture": texture, "effort": effort, "goal": goal}
            )
        except ValueError as error:
            raise InvalidImageError(str(error)) from error
        metadata = database.metadata()
        if metadata.get("indexing_status") == "running":
            raise IndexNotReadyError(
                "The local index is still building. Try again when it completes."
            )
        if metadata.get("landmark_version") != LANDMARK_VERSION:
            raise IndexNotReadyError("The shape index needs the v2 dense-landmark rebuild.")
        active_detector = getattr(app.state, "detector", None)
        if active_detector is None:
            try:
                active_detector = MediaPipeDetector(active_settings.model_path)
            except FileNotFoundError as error:
                raise IndexNotReadyError(str(error)) from error
            app.state.detector = active_detector
        images = {
            name: decode_image(payload, active_settings) for name, payload in payloads.items()
        }
        detections = {name: active_detector.detect_one(image) for name, image in images.items()}
        _validate_three_views(images, detections)
        aggregate, agreement = aggregate_views(
            [detections[name].normalized for name in ("front", "left", "right")]
        )
        measurements = shape_measurements(aggregate)
        memberships = shape_memberships(measurements)
        results = rank_shape_matches(database, aggregate, measurements)
        return {
            "notice": (
                "Haircut-oriented structural similarity only—not identity verification "
                "or a probability."
            ),
            "shape": {
                "primary": memberships[0][0],
                "secondary": memberships[1][0],
                "memberships": [
                    {"label": label, "score": score} for label, score in memberships[:2]
                ],
                "measurements": {key: round(value, 4) for key, value in measurements.items()},
                "three_view_agreement": agreement,
                "caveat": (
                    "MediaPipe estimates upper-face and temple structure, not the true hairline."
                ),
            },
            "diagnostics": [
                {
                    "view": name,
                    "yaw": detections[name].yaw,
                    "pitch": detections[name].pitch,
                    "roll": detections[name].roll,
                    "quality": detections[name].quality,
                }
                for name in ("front", "left", "right")
            ],
            "matches": [
                {
                    "rank": result.rank,
                    "identity": result.identity,
                    "image_url": f"/api/images/{result.image_id}",
                    "distance": round(result.distance, 8),
                    "similarity": result.similarity,
                    "breakdown": {
                        "silhouette": round(result.silhouette_distance, 8),
                        "jaw_chin_and_proportions": round(result.proportion_distance, 8),
                    },
                    "overlay": result.overlay,
                }
                for result in results
            ],
            "recommendations": recommend_haircuts(memberships, preferences),
            "preferences": preferences,
            "catalog_version": CATALOG_VERSION,
        }

    @app.get("/api/images/{image_id}")
    async def indexed_image(image_id: int) -> Any:
        metadata = database.metadata()
        root_text = metadata.get("dataset_root")
        if not root_text:
            return JSONResponse({"error": "Dataset path is unavailable."}, status_code=404)
        root = Path(root_text).resolve()
        with database.connect() as connection:
            row = connection.execute(
                "SELECT source_path FROM images WHERE id=? AND status='indexed'", (image_id,)
            ).fetchone()
        if row is None:
            return JSONResponse({"error": "Indexed image not found."}, status_code=404)
        candidate = (root / str(row["source_path"])).resolve()
        if root not in candidate.parents or not candidate.is_file():
            return JSONResponse({"error": "Indexed image not found."}, status_code=404)
        media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        return FileResponse(candidate, media_type=media_type)

    return app


app = create_app()
