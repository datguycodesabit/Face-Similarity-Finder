from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    model_path: Path
    database_path: Path
    dataset_path: Path
    max_upload_bytes: int = 8 * 1024 * 1024
    max_dimension: int = 6000
    min_dimension: int = 96

    @classmethod
    def from_env(cls) -> Settings:
        root = Path(os.getenv("FACE_MATCH_ROOT", Path.cwd())).resolve()
        return cls(
            project_root=root,
            model_path=Path(
                os.getenv("FACE_MATCH_MODEL", root / "models" / "face_landmarker.task")
            ).resolve(),
            database_path=Path(
                os.getenv("FACE_MATCH_DB", root / "data" / "face_match.sqlite3")
            ).resolve(),
            dataset_path=Path(
                os.getenv("FACE_MATCH_DATASET", root / "data" / "lfw_funneled")
            ).resolve(),
        )
