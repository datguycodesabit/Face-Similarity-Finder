from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import InvalidImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def decode_image(data: bytes, settings: Settings) -> Image.Image:
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // 1024 // 1024
        raise InvalidImageError(f"Image is too large. The upload limit is {limit_mb} MB.")
    try:
        with Image.open(BytesIO(data)) as opened:
            if opened.format not in ALLOWED_FORMATS:
                raise InvalidImageError("Use a JPEG, PNG, or WebP image.")
            width, height = opened.size
            if min(width, height) < settings.min_dimension:
                message = (
                    "Image is too small. Each side must be at least "
                    f"{settings.min_dimension} pixels."
                )
                raise InvalidImageError(message)
            if max(width, height) > settings.max_dimension:
                message = (
                    "Image dimensions are too large. Maximum side is "
                    f"{settings.max_dimension} pixels."
                )
                raise InvalidImageError(message)
            opened.verify()
        with Image.open(BytesIO(data)) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("The file is not a readable image.") from error
