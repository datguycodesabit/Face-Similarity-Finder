from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path
from urllib.error import URLError

import typer

from .commons_candidates import discover_candidates, write_candidates
from .commons_curator import CommonsServiceError, curate_gallery, load_candidate_catalog
from .commons_pack import download_manifest_images, load_manifest
from .config import Settings
from .database import Database
from .detector import MediaPipeDetector
from .geometry import LANDMARK_VERSION
from .indexer import index_dataset
from .mica import MICA_LICENSE_URL, MicaConfiguration, MicaWorkerClient
from .v3_database import V3Database
from .v3_indexer import index_v3_gallery

app = typer.Typer(no_args_is_help=True, help="Local facial-structure similarity finder.")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
LFW_URL = "https://ndownloader.figshare.com/files/5976015"
LFW_SHA256 = "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"
LFW_NOTICE = """
LFW is a research dataset of named people collected from the web. The repository does
not redistribute it and cannot grant rights to the underlying photographs. Use it only
for lawful personal/research purposes, review the official LFW documentation, and cite:
Huang, Ramesh, Berg & Learned-Miller, Labeled Faces in the Wild (UMass, 2007), plus
Huang, Jain & Learned-Miller, Unsupervised Joint Alignment of Complex Images
(ICCV, 2007) for the funneled images. Source: https://vis-www.cs.umass.edu/lfw/
""".strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual_sha256 = _sha256(temporary)
        if actual_sha256 != expected_sha256:
            raise OSError(f"SHA-256 mismatch: expected {expected_sha256}, received {actual_sha256}")
        temporary.replace(destination)
    except (OSError, URLError) as error:
        typer.echo(
            f"Download failed for {url}. Check the network connection and try again.",
            err=True,
        )
        raise typer.Exit(1) from error
    finally:
        temporary.unlink(missing_ok=True)


@app.command("setup-model")
def setup_model() -> None:
    """Download the pinned MediaPipe Face Landmarker model bundle."""
    settings = Settings.from_env()
    if settings.model_path.is_file():
        digest = _sha256(settings.model_path)
        if digest != MODEL_SHA256:
            typer.echo(
                f"Existing model checksum is invalid: {settings.model_path}\n"
                f"Expected {MODEL_SHA256}; received {digest}.",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Model already present: {settings.model_path}")
        return
    typer.echo("Downloading the MediaPipe Face Landmarker model locally…")
    _download(MODEL_URL, settings.model_path, MODEL_SHA256)
    digest = _sha256(settings.model_path)
    typer.echo(f"Saved {settings.model_path}\nSHA-256: {digest}")


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise typer.BadParameter("Dataset archive contains an unsafe path.")
        if member.issym() or member.islnk():
            raise typer.BadParameter("Dataset archive contains unsupported links.")
    archive.extractall(destination, filter="data")


@app.command("prepare-lfw")
def prepare_lfw(
    accept_dataset_notice: bool = typer.Option(
        False, "--accept-dataset-notice", help="Acknowledge the displayed LFW usage notice."
    ),
) -> None:
    """Acknowledge the notice, then download and unpack complete funneled LFW."""
    typer.echo(LFW_NOTICE)
    if not accept_dataset_notice:
        raise typer.BadParameter("Re-run with --accept-dataset-notice after reviewing the notice.")
    settings = Settings.from_env()
    if settings.dataset_path.is_dir():
        typer.echo(f"Dataset already present: {settings.dataset_path}")
        return
    archive_path = settings.dataset_path.parent / "lfw-funneled.tgz"
    typer.echo("Downloading checksum-pinned complete funneled LFW…")
    _download(LFW_URL, archive_path, LFW_SHA256)
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract(archive, settings.dataset_path.parent)
    archive_path.unlink(missing_ok=True)
    typer.echo(f"Prepared {settings.dataset_path}")


@app.command("check-mica")
def check_mica() -> None:
    """Report exact setup steps for the separately licensed local MICA dependency."""
    settings = Settings.from_env()
    configuration = MicaConfiguration.from_env(settings.project_root)
    problems = configuration.problems()
    typer.echo(f"MICA license: {MICA_LICENSE_URL}")
    if problems:
        for problem in problems:
            typer.echo(f"- {problem}", err=True)
        raise typer.Exit(1)
    typer.echo("MICA, its weights, and FLAME are configured for local V3 analysis.")


@app.command("prepare-commons")
def prepare_commons(
    manifest: Path | None = typer.Option(None, "--manifest", exists=True, dir_okay=False),
    accept_commons_notice: bool = typer.Option(False, "--accept-commons-notice"),
) -> None:
    """Validate and download the checksum-locked, per-file-attributed Commons pack."""
    if not accept_commons_notice:
        raise typer.BadParameter(
            "Review each Commons file's attribution/license, then re-run with "
            "--accept-commons-notice."
        )
    settings = Settings.from_env()
    manifest_path = manifest or settings.commons_manifest_path
    if manifest_path is None:
        raise typer.BadParameter("A Commons manifest path is required.")
    gallery = load_manifest(manifest_path)
    destination = settings.v3_dataset_path or settings.project_root / "data" / "commons_v3"

    def progress(done: int, total: int, identity: object, image: object) -> None:
        if done == 1 or done == total or done % 50 == 0:
            typer.echo(f"[{done}/{total}] Commons references")

    summary = download_manifest_images(gallery, destination, progress)
    typer.echo(f"Commons pack ready: {summary}")


@app.command("discover-commons")
def discover_commons(
    output: Path = typer.Option(
        Path("manifests/commons-candidates.json"), "--output", dir_okay=False
    ),
    limit: int = typer.Option(2000, "--limit", min=500, max=5000),
) -> None:
    """Create a surplus Wikidata/Commons candidate list for later local vision curation."""
    payload = discover_candidates(limit)
    write_candidates(output.resolve(), payload)
    typer.echo(
        f"Saved {payload['candidate_count']} unaccepted candidates to {output.resolve()}. "
        "They still require MediaPipe/MICA filtering before entering the 500-person manifest."
    )


@app.command("curate-commons")
def curate_commons(
    candidates: Path = typer.Option(
        Path("manifests/commons-candidates.json"), "--candidates", exists=True, dir_okay=False
    ),
    output: Path = typer.Option(Path("manifests/commons-v3-500.json"), "--output", dir_okay=False),
    accept_commons_notice: bool = typer.Option(False, "--accept-commons-notice"),
) -> None:
    """Build the exact 500-person manifest using local MediaPipe and MICA gates."""
    if not accept_commons_notice:
        raise typer.BadParameter(
            "Review Commons attribution/reuse requirements, then re-run with "
            "--accept-commons-notice."
        )
    settings = Settings.from_env()
    detector = MediaPipeDetector(settings.model_path)
    mica = MicaWorkerClient(MicaConfiguration.from_env(settings.project_root))
    destination = settings.v3_dataset_path or settings.project_root / "data" / "commons_v3"

    def progress(accepted: int, rejected: int, name: str) -> None:
        if accepted == 1 or (accepted > 0 and accepted % 10 == 0) or rejected % 25 == 0:
            typer.echo(f"accepted={accepted}/500 rejected={rejected} latest={name}")

    try:
        try:
            summary = curate_gallery(
                load_candidate_catalog(candidates),
                destination,
                output.resolve(),
                detector,
                mica,
                settings,
                progress=progress,
            )
        except CommonsServiceError as error:
            typer.echo(
                f"{error}. The curation checkpoint is safe; wait for the Commons cooldown "
                "and rerun the same command.",
                err=True,
            )
            raise typer.Exit(code=1) from error
    finally:
        mica.close()
        detector.close()
    typer.echo(f"Commons curation complete: {summary}")


@app.command("index-v3")
def index_v3(
    manifest: Path | None = typer.Option(None, "--manifest", exists=True, dir_okay=False),
) -> None:
    """Build or resume the separate MICA/FLAME identity-prototype index."""
    settings = Settings.from_env()
    manifest_path = manifest or settings.commons_manifest_path
    if manifest_path is None:
        raise typer.BadParameter("A Commons manifest path is required.")
    database_path = (
        settings.v3_database_path or settings.project_root / "data" / "face_match_v3.sqlite3"
    )
    gallery = load_manifest(manifest_path)
    detector = MediaPipeDetector(settings.model_path)
    mica = MicaWorkerClient(MicaConfiguration.from_env(settings.project_root))

    def progress(done: int, total: int, name: str) -> None:
        if done == 1 or done == total or done % 10 == 0:
            typer.echo(f"[{done}/{total}] {name}")

    try:
        summary = index_v3_gallery(
            gallery, V3Database(database_path), detector, mica, settings, progress
        )
    finally:
        mica.close()
        detector.close()
    typer.echo(f"V3 index complete: {summary}")


@app.command("index")
def index(
    dataset: Path = typer.Option(..., "--dataset", exists=True, file_okay=False, resolve_path=True),
) -> None:
    """Build or resume the local structural landmark index."""
    settings = Settings.from_env()
    database = Database(settings.database_path)
    detector = MediaPipeDetector(settings.model_path)

    def progress(done: int, total: int, name: str) -> None:
        if done == 1 or done == total or done % 100 == 0:
            typer.echo(f"[{done}/{total}] {name}")

    try:
        summary = index_dataset(dataset, database, detector, settings, progress)
        database.set_metadata("landmark_version", LANDMARK_VERSION)
    finally:
        detector.close()
    typer.echo(f"Index complete: {summary}")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, min=1, max=65535, help="Local port."),
) -> None:
    """Start the local web application."""
    import uvicorn

    uvicorn.run("face_match.web:app", host=host, port=port, reload=False)
