#!/usr/bin/env python3
"""JSON-lines MICA adapter. Run only inside a separately licensed MICA environment."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# The adapter lives in this project while official MICA remains a separate,
# unvendored checkout. The worker runs with MICA_HOME as its cwd, but Python sets
# sys.path[0] to this script's directory, so add only that explicit checkout root.
sys.path.insert(0, str(Path.cwd()))

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np
import torch  # type: ignore[import-not-found]

# FLAME 2020 pickles may import Chumpy 0.70, which still calls the removed
# inspect.getargspec API. Its return fields are compatible with getfullargspec
# for this deserialization path.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
for legacy_name, builtin_type in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("unicode", str),
    ("str", str),
):
    if legacy_name not in np.__dict__:
        setattr(np, legacy_name, builtin_type)

from configs.config import get_cfg_defaults  # type: ignore[import-not-found]  # noqa: E402
from utils import util  # type: ignore[import-not-found]  # noqa: E402

ARCFACE_TEMPLATE = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _load_checkpoint(path: Path, model: Any, device: str) -> None:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.arcface.load_state_dict(checkpoint["arcface"])
    model.flameModel.load_state_dict(checkpoint["flameModel"])


def _similarity_error(source: np.ndarray, target: np.ndarray) -> float:
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    denominator = float(np.sum(source_centered**2))
    if denominator <= 1e-12:
        return 1.0
    left, singular, right = np.linalg.svd(source_centered.T @ target_centered)
    rotation = left @ right
    scale = float(np.sum(singular) / denominator)
    fitted = source_centered @ rotation * scale + target.mean(axis=0)
    eye_distance = max(float(np.linalg.norm(target[0] - target[1])), 1e-6)
    return float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))) / eye_distance)


class Worker:
    def __init__(self) -> None:
        self.device = "cpu"
        cfg = get_cfg_defaults()
        cfg.model.testing = True
        checkpoint_path = Path(
            os.environ.get("MICA_MODEL", str(Path("data/pretrained/mica.tar").resolve()))
        )
        # Upstream MICA loads its checkpoint inside the constructor with torch.load()
        # defaults that are not portable to newer CPU-only PyTorch releases.  Point
        # the constructor at a deliberately absent file, then load the same official
        # checkpoint ourselves with an explicit CPU map location.
        cfg.pretrained_model_path = str(checkpoint_path.with_name(".face-match-worker-load"))
        cfg.model.flame_model_path = os.environ.get(
            "FLAME_MODEL", str(Path("data/FLAME2020/generic_model.pkl").resolve())
        )
        factory = util.find_model_using_name(model_dir="micalib.models", model_name=cfg.model.name)
        self.model = factory(cfg, self.device)
        _load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()
        model_path = Path(
            os.environ.get(
                "FACE_MATCH_MODEL",
                str(Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"),
            )
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.region_indices = self._build_region_indices()
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:16]
        self.engine_version = f"mica-eccv2022-cpu-{digest}-worker-v1"

    def _build_region_indices(self) -> dict[str, np.ndarray]:
        template = np.asarray(self.model.flame.v_template.detach().cpu(), dtype=np.float64)
        if template.ndim == 3:
            template = template[0]
        centered = template - np.median(template, axis=0)
        horizontal = centered[:, 0]
        vertical = centered[:, 1]
        depth = centered[:, 2]
        front = depth >= np.quantile(depth, 0.45)
        jaw = np.flatnonzero(front & (vertical <= np.quantile(vertical[front], 0.30)))
        outline = np.flatnonzero(
            front
            & (np.abs(horizontal) >= np.quantile(np.abs(horizontal[front]), 0.52))
            & (vertical <= np.quantile(vertical[front], 0.80))
        )
        upper = np.flatnonzero(front & (vertical >= np.quantile(vertical[front], 0.55)))
        middle = np.flatnonzero(
            front & (np.abs(horizontal) <= np.quantile(np.abs(horizontal[front]), 0.35))
        )
        if min(len(jaw), len(outline), len(upper), len(middle)) < 10:
            raise RuntimeError("Could not derive stable FLAME face regions from the template")
        jaw = jaw[np.argsort(horizontal[jaw])]
        outline = outline[np.argsort(horizontal[outline])]
        return {
            "jaw_chin": jaw,
            "outline_cheeks": outline,
            "global_proportions": np.flatnonzero(front)[::8],
            "eye_brow_geometry": upper[::4],
            "nose_midface_geometry": middle[::4],
        }

    def analyze_one(self, encoded: str) -> tuple[dict[str, Any], np.ndarray]:
        payload = base64.b64decode(encoded, validate=True)
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("MICA could not decode an input image")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if len(detection.face_landmarks) != 1:
            raise ValueError(
                f"MICA expected exactly one face; found {len(detection.face_landmarks)}"
            )
        landmarks = detection.face_landmarks[0]
        width, height = image.shape[1], image.shape[0]

        def midpoint(first: int, second: int) -> tuple[float, float]:
            return (
                (landmarks[first].x + landmarks[second].x) * width / 2.0,
                (landmarks[first].y + landmarks[second].y) * height / 2.0,
            )

        keypoints = np.asarray(
            [
                midpoint(33, 133),
                midpoint(362, 263),
                (landmarks[1].x * width, landmarks[1].y * height),
                (landmarks[61].x * width, landmarks[61].y * height),
                (landmarks[291].x * width, landmarks[291].y * height),
            ],
            dtype=np.float32,
        )
        transform, _ = cv2.estimateAffinePartial2D(keypoints, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if transform is None:
            raise ValueError("MICA could not align the detected face")
        aligned = cv2.warpAffine(image, transform, (112, 112), borderValue=0)
        arcface = ((aligned[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)
        display = cv2.resize(aligned, (224, 224))[:, :, ::-1].transpose(2, 0, 1)
        image_tensor = torch.from_numpy(display.copy()).float()[None].to(self.device) / 255.0
        arcface_tensor = torch.from_numpy(arcface.copy()).float()[None].to(self.device)
        with torch.no_grad():
            encoded_shape = self.model.encode(image_tensor, arcface_tensor)
            output = self.model.decode(encoded_shape)
            vertices = output["pred_canonical_shape_vertices"]
            shape_code = output["pred_shape_code"]
            landmarks = self.model.flame.compute_landmarks(vertices)[0].detach().cpu().numpy()
            identity_code = encoded_shape["arcface"][0].detach().cpu().numpy()
        mesh = vertices[0].detach().cpu().numpy()
        predicted_five = np.stack(
            (
                landmarks[36:42, :2].mean(axis=0),
                landmarks[42:48, :2].mean(axis=0),
                landmarks[30, :2],
                landmarks[48, :2],
                landmarks[54, :2],
            )
        )
        aligned_keypoints = cv2.transform(keypoints[None], transform)[0]
        return (
            {
                "shape_code": shape_code[0].detach().cpu().tolist(),
                "regions": {
                    name: mesh[indices].tolist() for name, indices in self.region_indices.items()
                },
                "reprojection_error": _similarity_error(predicted_five, aligned_keypoints),
            },
            identity_code,
        )

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        operation = request.get("operation")
        if operation == "hello":
            return {"engine_version": self.engine_version}
        if operation != "analyze":
            raise ValueError(f"Unsupported MICA worker operation: {operation}")
        if request.get("return_arcface_embedding") is not False:
            raise ValueError("ArcFace embeddings must remain disabled")
        images = request.get("images")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError("Exactly three images are required")
        analyzed = [self.analyze_one(str(image)) for image in images]
        if request.get("validate_identity") is True:
            anchor_index = int(request.get("anchor_index", 0))
            if anchor_index not in range(3):
                raise ValueError("Identity anchor index must be 0, 1, or 2")
            anchor = analyzed[anchor_index][1]
            minimum_similarity = float(os.environ.get("MICA_IDENTITY_MIN_COSINE", "0.20"))
            for index, (_, embedding) in enumerate(analyzed):
                if index == anchor_index:
                    continue
                similarity = float(np.dot(anchor, embedding))
                if similarity < minimum_similarity:
                    raise ValueError(
                        "Commons identity validation rejected image "
                        f"{index + 1}: transient MICA encoder cosine {similarity:.3f} "
                        f"is below {minimum_similarity:.3f}"
                    )
        # Identity encodings are deliberately discarded inside this worker and are
        # never serialized back to the app, database, or manifest.
        return {"results": [result for result, _ in analyzed]}


def main() -> None:
    worker = Worker()
    for line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(line)
            response = worker.handle(request)
        except Exception as error:
            response = {"error": str(error)}
        if isinstance(request, Mapping) and "request_id" in request:
            response["request_id"] = request["request_id"]
        print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
