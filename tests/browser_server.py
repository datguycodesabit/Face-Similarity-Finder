from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import uvicorn
from PIL import Image, ImageDraw, ImageOps

from face_match.config import Settings
from face_match.database import Database
from face_match.web import create_app
from tests.helpers import SequenceDetector, dense_detection


def build_app() -> Any:
    root = Path(os.environ["BROWSER_FIXTURE_ROOT"]).resolve()
    dataset = root / "lfw"
    dataset.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        root, root / "model.task", root / "index.sqlite3", dataset, 8 * 1024 * 1024, 1000, 8
    )
    detector = SequenceDetector(
        [dense_detection(10, 0), dense_detection(10, -22), dense_detection(10, 22)]
    )
    database = Database(settings.database_path)
    database.initialize()
    error_fixture = root / "unsupported.gif"
    Image.new("RGB", (120, 120), "#8b7064").save(error_fixture, "GIF")
    Image.radial_gradient("L").resize((260, 310)).convert("RGB").save(root / "query_front.png")
    gradient = Image.linear_gradient("L").rotate(90, expand=True).resize((260, 310))
    gradient.convert("RGB").save(root / "query_left.png")
    ImageOps.invert(gradient).convert("RGB").save(root / "query_right.png")
    palette = ["#977264", "#7d665e", "#ae7e62", "#6d5147", "#b98b72", "#8a705c"]
    for index, color in enumerate(palette):
        path = dataset / f"reference_{index}.png"
        if not path.exists():
            image = Image.new("RGB", (260, 310), color)
            draw = ImageDraw.Draw(image)
            draw.ellipse((60, 35, 200, 250), fill="#d5aa8f")
            draw.ellipse((92, 112, 105, 121), fill="#2d2826")
            draw.ellipse((155, 112, 168, 121), fill="#2d2826")
            draw.arc((110, 165, 150, 190), 0, 180, fill="#744b43", width=3)
            image.save(path)
        image_id = database.register_image(
            f"Reference Person {index + 1}", path.name, detector.model_version
        )
        database.save_vector(
            image_id,
            dense_detection(20 + index).normalized,
            dense_detection(20 + index).overlay,
            detector.model_version,
            dense_detection(20 + index).measurements,
        )
    database.set_metadata("dataset_root", str(dataset))
    database.set_metadata("indexing_status", "complete")
    database.set_metadata("landmark_version", "structural-v2-dense-weighted-three-view")
    return create_app(settings, detector)


if __name__ == "__main__":
    uvicorn.run(build_app(), host="127.0.0.1", port=8765)
