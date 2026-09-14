from __future__ import annotations

import base64
import json
import os
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .shape_v3 import COMPONENT_NAMES

FloatArray = NDArray[np.float64]

MICA_LICENSE_URL = "https://raw.githubusercontent.com/Zielon/MICA/master/LICENSE"


class MicaError(RuntimeError):
    pass


class MicaUnavailableError(MicaError):
    pass


@dataclass(frozen=True)
class MicaConfiguration:
    home: Path
    worker: Path
    model: Path
    flame_model: Path
    license_accepted: bool
    python_executable: Path | None = None

    @classmethod
    def from_env(cls, project_root: Path) -> MicaConfiguration:
        home = Path(os.getenv("MICA_HOME", project_root / "models" / "MICA")).resolve()
        acceptance_marker = project_root / "models" / ".mica-license-accepted"
        return cls(
            home=home,
            worker=Path(
                os.getenv("MICA_WORKER", project_root / "scripts" / "mica_worker.py")
            ).resolve(),
            model=Path(
                os.getenv("MICA_MODEL", home / "data" / "pretrained" / "mica.tar")
            ).resolve(),
            flame_model=Path(
                os.getenv("FLAME_MODEL", home / "data" / "FLAME2020" / "generic_model.pkl")
            ).resolve(),
            license_accepted=(
                os.getenv("MICA_LICENSE_ACCEPTED") == "1" or acceptance_marker.is_file()
            ),
            python_executable=Path(
                os.getenv("MICA_PYTHON", home / ".venv" / "bin" / "python")
            ).absolute(),
        )

    def problems(self) -> list[str]:
        problems: list[str] = []
        if not self.license_accepted:
            problems.append(
                "Set MICA_LICENSE_ACCEPTED=1 only after reading and accepting the personal, "
                f"noncommercial MICA license at {MICA_LICENSE_URL}."
            )
        upstream_flame_path = (self.home / "data" / "FLAME2020" / "generic_model.pkl").resolve()
        if self.flame_model != upstream_flame_path:
            problems.append(
                "The official MICA masking code requires FLAME at "
                f"{upstream_flame_path}; move or copy the licensed file there and unset "
                "FLAME_MODEL."
            )
        for label, path in (
            ("MICA checkout", self.home),
            ("Face Match MICA worker", self.worker),
            ("MICA weights", self.model),
            ("FLAME model", self.flame_model),
            (
                "MICA Python executable",
                self.python_executable or self.home / ".venv" / "bin" / "python",
            ),
        ):
            exists = path.is_dir() if label == "MICA checkout" else path.is_file()
            if not exists:
                problems.append(f"{label} is missing at {path}.")
        return problems

    @property
    def ready(self) -> bool:
        return not self.problems()


@dataclass(frozen=True)
class MicaViewResult:
    shape_code: FloatArray
    regions: dict[str, FloatArray]
    reprojection_error: float


@dataclass(frozen=True)
class MicaAggregate:
    shape_code: FloatArray
    regions: dict[str, FloatArray]
    accepted_views: tuple[int, ...]
    rejected_view: int | None


class MicaEngine(Protocol):
    engine_version: str

    def analyze(
        self,
        images: Sequence[bytes],
        *,
        validate_identity: bool = False,
        anchor_index: int = 0,
    ) -> list[MicaViewResult]: ...

    def close(self) -> None: ...


def _decode_result(payload: Mapping[str, Any]) -> MicaViewResult:
    shape_code = np.asarray(payload.get("shape_code"), dtype=np.float64)
    if shape_code.ndim != 1 or shape_code.size == 0 or not np.isfinite(shape_code).all():
        raise MicaError("MICA worker returned an invalid identity-shape vector")
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, Mapping):
        raise MicaError("MICA worker did not return FLAME regions")
    regions: dict[str, FloatArray] = {}
    for name in COMPONENT_NAMES:
        region = np.asarray(raw_regions.get(name), dtype=np.float64)
        if region.ndim != 2 or region.shape[1] != 3 or not np.isfinite(region).all():
            raise MicaError(f"MICA worker returned an invalid {name} region")
        regions[name] = region
    error = float(payload.get("reprojection_error", np.inf))
    if not np.isfinite(error) or error < 0.0:
        raise MicaError("MICA worker returned an invalid reprojection error")
    return MicaViewResult(shape_code, regions, error)


def aggregate_mica_views(results: Sequence[MicaViewResult]) -> MicaAggregate:
    if len(results) != 3:
        raise ValueError("exactly three MICA view results are required")
    shape = results[0].shape_code.shape
    region_shapes = {name: results[0].regions[name].shape for name in COMPONENT_NAMES}
    if any(result.shape_code.shape != shape for result in results):
        raise MicaError("MICA shape-vector dimensions differ between views")
    if any(
        result.regions[name].shape != region_shapes[name]
        for result in results
        for name in COMPONENT_NAMES
    ):
        raise MicaError("MICA FLAME topology differs between views")

    standardized = np.stack([result.shape_code for result in results])
    coefficient_scale = np.maximum(np.median(np.abs(standardized), axis=0), 1e-6)
    standardized = standardized / coefficient_scale
    pairwise = np.asarray(
        [
            np.linalg.norm(standardized[first] - standardized[second])
            for first, second in ((0, 1), (0, 2), (1, 2))
        ],
        dtype=np.float64,
    )
    closest_pair_index = int(np.argmin(pairwise))
    pairs = ((0, 1), (0, 2), (1, 2))
    closest_pair = pairs[closest_pair_index]
    remaining = next(index for index in range(3) if index not in closest_pair)
    distances_to_pair = [
        float(np.linalg.norm(standardized[remaining] - standardized[index]))
        for index in closest_pair
    ]
    pair_distance = float(pairwise[closest_pair_index])
    is_outlier = min(distances_to_pair) > max(3.0 * pair_distance, 1e-6)
    accepted = closest_pair if is_outlier else (0, 1, 2)
    rejected = remaining if is_outlier else None

    shape_code = np.median(np.stack([results[index].shape_code for index in accepted]), axis=0)
    regions = {
        name: np.median(np.stack([results[index].regions[name] for index in accepted]), axis=0)
        for name in COMPONENT_NAMES
    }
    return MicaAggregate(
        shape_code,
        regions,
        tuple(accepted),
        rejected,
    )


class MicaWorkerClient:
    """Persistent JSON-lines adapter around a separately installed MICA environment."""

    def __init__(
        self,
        configuration: MicaConfiguration,
        python_executable: Path | None = None,
    ) -> None:
        problems = configuration.problems()
        if problems:
            raise MicaUnavailableError(" ".join(problems))
        executable = (
            python_executable
            or configuration.python_executable
            or configuration.home / ".venv" / "bin" / "python"
        )
        if not executable.is_file():
            raise MicaUnavailableError(f"MICA Python executable is missing at {executable}.")
        self._process = subprocess.Popen(
            [str(executable), str(configuration.worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=configuration.home,
            env={
                **os.environ,
                "MICA_MODEL": str(configuration.model),
                "FLAME_MODEL": str(configuration.flame_model),
            },
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise MicaUnavailableError("MICA worker pipes could not be created")
        self._input: IO[str] = self._process.stdin
        self._output: IO[str] = self._process.stdout
        hello = self._request({"operation": "hello"})
        self.engine_version = str(hello.get("engine_version", ""))
        if not self.engine_version:
            self.close()
            raise MicaError("MICA worker did not report an engine version")

    def _request(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise MicaError(f"MICA worker exited unexpectedly: {stderr[-1000:]}")
        request_id = uuid.uuid4().hex
        request = {"request_id": request_id, **body}
        self._input.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._input.flush()
        line = self._output.readline()
        if not line:
            with suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=1)
            stderr = self._process.stderr.read() if self._process.stderr else ""
            detail = stderr[-2000:].strip()
            message = "MICA worker closed its output without a response"
            if detail:
                message += f": {detail}"
            raise MicaError(message)
        response = json.loads(line)
        if not isinstance(response, Mapping) or response.get("request_id") != request_id:
            raise MicaError("MICA worker returned a mismatched response")
        if response.get("error"):
            raise MicaError(str(response["error"]))
        return response

    def analyze(
        self,
        images: Sequence[bytes],
        *,
        validate_identity: bool = False,
        anchor_index: int = 0,
    ) -> list[MicaViewResult]:
        if len(images) != 3 or any(not image for image in images):
            raise ValueError("exactly three nonempty encoded images are required")
        if anchor_index not in range(3):
            raise ValueError("MICA identity anchor index must be 0, 1, or 2")
        response = self._request(
            {
                "operation": "analyze",
                "images": [base64.b64encode(image).decode("ascii") for image in images],
                "return_arcface_embedding": False,
                "validate_identity": validate_identity,
                "anchor_index": anchor_index,
            }
        )
        raw_results = response.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != 3:
            raise MicaError("MICA worker did not return three view results")
        return [_decode_result(result) for result in raw_results]

    def close(self) -> None:
        if getattr(self, "_process", None) is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

    def __enter__(self) -> MicaWorkerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
