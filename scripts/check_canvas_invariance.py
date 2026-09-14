from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from face_match.config import Settings
from face_match.detector import MediaPipeDetector


def _padded(image: Image.Image, width: int, height: int) -> Image.Image:
    if width < image.width or height < image.height:
        raise ValueError("padding canvas cannot be smaller than the source image")
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify real MediaPipe V3 canvas invariance.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args()
    source = Image.open(args.image).convert("RGB")
    base = max(source.size) + 150
    variants = {
        "square": _padded(source, base, base),
        "portrait": _padded(source, base, base + 200),
        "landscape": _padded(source, base + 200, base),
    }
    detector = MediaPipeDetector(Settings.from_env().model_path)
    try:
        results = {name: detector.detect_one(image) for name, image in variants.items()}
    finally:
        detector.close()
    baseline = results["square"].v3_measurements
    failures: list[str] = []
    for variant in ("portrait", "landscape"):
        for name, value in results[variant].v3_measurements.items():
            denominator = max(abs(baseline[name]), 1e-6)
            drift = abs(value - baseline[name]) / denominator
            if drift > args.threshold:
                failures.append(f"{variant}:{name}={drift:.2%}")
    if failures:
        raise SystemExit(f"Canvas invariance exceeded {args.threshold:.1%}: " + ", ".join(failures))
    print(f"PASS: all V3 descriptors stayed within {args.threshold:.1%} across three canvases")


if __name__ == "__main__":
    main()
