from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path
from urllib.error import URLError

import typer

from .config import Settings
from .database import Database
from .detector import MediaPipeDetector
from .geometry import LANDMARK_VERSION
from .indexer import index_dataset

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
