from __future__ import annotations

import json
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
from .geometry import LANDMARK_VERSION
from .images import decode_image
from .matcher import rank_matches
from .mica import (
    MicaConfiguration,
    MicaEngine,
    MicaError,
    MicaUnavailableError,
    MicaWorkerClient,
    aggregate_mica_views,
)
from .recommendations import CATALOG_VERSION, recommend_haircuts, validate_preferences
from .shape_v3 import COMPONENT_WEIGHTS, V3_ALGORITHM_VERSION, calibrated_shape_labels
from .v3_database import V3Database
from .v3_matcher import rank_v3_matches

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


def _normalized_contour(points: np.ndarray) -> list[list[float]]:
    plane = np.asarray(points, dtype=np.float64)[:, :2]
    low = np.min(plane, axis=0)
    span = np.maximum(np.max(plane, axis=0) - low, 1e-9)
    normalized = 0.1 + 0.8 * (plane - low) / span
    return [[round(float(x), 6), round(float(y), 6)] for x, y in normalized]


def create_app(
    settings: Settings | None = None,
    detector: Detector | None = None,
    mica_engine: MicaEngine | None = None,
) -> FastAPI:
    active_settings = settings or Settings.from_env()
    database = Database(active_settings.database_path)
    database.initialize()
    v3_database_path = (
        active_settings.v3_database_path
        or active_settings.project_root / "data" / "face_match_v3.sqlite3"
    )
    v3_database = V3Database(v3_database_path)
    v3_database.initialize()
    mica_configuration = MicaConfiguration.from_env(active_settings.project_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.detector = detector
        app.state.mica_engine = mica_engine
        yield
        for name in ("detector", "mica_engine"):
            current = getattr(app.state, name, None)
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
    app.state.v3_database = v3_database
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

    @app.exception_handler(MicaError)
    async def mica_error(_: Request, exc: MicaError) -> JSONResponse:
        return JSONResponse(
            {
                "error": (
                    "The local MICA reconstruction failed. No query photo was retained. "
                    f"Details: {exc}"
                )
            },
            status_code=422,
        )

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
        v3_counts = v3_database.status()
        v3_metadata = v3_database.metadata()
        mica_problems = [] if mica_engine is not None else mica_configuration.problems()
        version_compatible = metadata.get("landmark_version", LANDMARK_VERSION) == LANDMARK_VERSION
        indexing_complete = metadata.get("indexing_status", "complete") == "complete"
        v3_version_compatible = (
            v3_metadata.get("algorithm_version", V3_ALGORITHM_VERSION) == V3_ALGORITHM_VERSION
        )
        calibration_ready = v3_database.active_hybrid_calibration() is not None
        v3_index_complete = v3_metadata.get("indexing_status") == "complete"
        v3_ready = (
            not mica_problems
            and v3_counts["prototypes"] >= 5
            and calibration_ready
            and v3_index_complete
            and v3_version_compatible
        )
        setup_steps = list(mica_problems)
        if v3_counts["identities"] == 0:
            setup_steps.append("Run `face-match prepare-commons --accept-commons-notice`.")
        if not v3_index_complete or v3_counts["prototypes"] < 5 or not calibration_ready:
            setup_steps.append("Run `face-match index-v3` after MICA and Commons are ready.")
        return {
            "model_ready": active_settings.model_path.is_file() or detector is not None,
            "index_ready": v3_ready,
            "version_compatible": v3_version_compatible,
            "indexing_status": v3_metadata.get("indexing_status", "not_started"),
            "analysis_version": V3_ALGORITHM_VERSION,
            "engine": v3_metadata.get("mica_engine_version", "MICA (not configured)"),
            "gallery_version": v3_metadata.get("gallery_version"),
            "mica": {
                "installed": mica_configuration.home.is_dir(),
                "license_accepted": mica_configuration.license_accepted,
                "ready": not mica_problems,
                "problems": mica_problems,
            },
            "commons": {
                "downloaded": v3_counts["indexed"] + v3_counts["invalid"],
                "total": v3_counts["images"],
                "gallery_version": v3_metadata.get("gallery_version"),
            },
            "calibration_ready": calibration_ready,
            "prototype_count": v3_counts["prototypes"],
            "counts": v3_counts,
            "setup_steps": setup_steps,
            "legacy_v2": {
                "version_compatible": version_compatible,
                "indexing_complete": indexing_complete,
                "counts": counts,
            },
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
        metadata = v3_database.metadata()
        counts = v3_database.status()
        calibration = v3_database.active_hybrid_calibration()
        if metadata.get("indexing_status") == "running":
            raise IndexNotReadyError(
                "The V3 Commons prototype index is still building. Try again when it completes."
            )
        if metadata.get("algorithm_version") not in (None, V3_ALGORITHM_VERSION):
            raise IndexNotReadyError(
                "The stored V3 prototype index is incompatible; re-run index-v3."
            )
        if (
            counts["prototypes"] < 5
            or calibration is None
            or metadata.get("indexing_status") != "complete"
        ):
            raise IndexNotReadyError(
                "V3 is not ready. Run `face-match check-mica`, then "
                "`face-match prepare-commons --accept-commons-notice`, then `face-match index-v3`."
            )
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
        active_mica = getattr(app.state, "mica_engine", None)
        if active_mica is None:
            try:
                active_mica = MicaWorkerClient(mica_configuration)
            except MicaUnavailableError as error:
                raise IndexNotReadyError(str(error)) from error
            app.state.mica_engine = active_mica
        mica_views = active_mica.analyze([payloads[name] for name in ("front", "left", "right")])
        aggregate = aggregate_mica_views(mica_views)
        accepted_measurements = [
            detections[("front", "left", "right")[index]].v3_measurements
            for index in aggregate.accepted_views
        ]
        measurements = {
            name: float(np.median([sample[name] for sample in accepted_measurements]))
            for name in accepted_measurements[0]
        }
        prototype_rows = v3_database.prototype_rows()
        population = [json.loads(str(row["descriptor_json"])) for row in prototype_rows]
        labels = calibrated_shape_labels(measurements, population)
        memberships = [(str(item["label"]), float(item["score"]) / 100.0) for item in labels]
        results = rank_v3_matches(v3_database, aggregate, measurements, calibration)
        warnings = []
        if aggregate.rejected_view is not None:
            rejected = ("front", "left", "right")[aggregate.rejected_view]
            warnings.append(f"The {rejected} MICA reconstruction was rejected as a shape outlier.")
        return {
            "analysis_version": V3_ALGORITHM_VERSION,
            "engine": active_mica.engine_version,
            "gallery_version": metadata.get("gallery_version"),
            "warnings": warnings,
            "notice": (
                "Haircut-oriented structural similarity only—not identity verification "
                "or a probability."
            ),
            "shape": {
                "primary": labels[0]["label"],
                "secondary": labels[1]["label"],
                "memberships": labels,
                "measurements": {key: round(value, 4) for key, value in measurements.items()},
                "three_view_agreement": round(len(aggregate.accepted_views) / 3.0 * 100.0, 1),
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
                    "image_url": f"/api/v3/images/{result.image_id}",
                    "distance": round(result.distance, 8),
                    "similarity": result.similarity,
                    "breakdown": {
                        name: round(value, 8) for name, value in result.breakdown.items()
                    },
                    "explanation": (
                        "The largest weighted differences were "
                        + " and ".join(
                            name.replace("_", " ")
                            for name in sorted(
                                result.breakdown,
                                key=lambda component: (
                                    COMPONENT_WEIGHTS[component] * result.breakdown[component]
                                ),
                                reverse=True,
                            )[:2]
                        )
                        + ". Lower component distances are closer."
                    ),
                    "attribution": {
                        "author": result.attribution.author,
                        "license": result.attribution.license_id,
                        "license_url": result.attribution.license_url,
                        "commons_url": result.attribution.commons_url,
                    },
                    "jaw_comparison": {
                        "query": _normalized_contour(aggregate.regions["jaw_chin"]),
                        "reference": _normalized_contour(
                            v3_database.prototype_regions(result.identity_id)["jaw_chin"]
                        ),
                    },
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

    @app.get("/api/v3/images/{image_id}")
    async def v3_indexed_image(image_id: int) -> Any:
        metadata = v3_database.metadata()
        root_text = metadata.get("dataset_root")
        if not root_text:
            return JSONResponse({"error": "Commons dataset path is unavailable."}, status_code=404)
        root = Path(root_text).resolve()
        with v3_database.connect() as connection:
            row = connection.execute(
                "SELECT local_path FROM images WHERE id=? AND status='indexed'", (image_id,)
            ).fetchone()
        if row is None:
            return JSONResponse({"error": "Commons reference image not found."}, status_code=404)
        candidate = Path(str(row["local_path"])).resolve()
        if root not in candidate.parents or not candidate.is_file():
            return JSONResponse({"error": "Commons reference image not found."}, status_code=404)
        media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        return FileResponse(candidate, media_type=media_type)

    return app


app = create_app()
