# Face Shape Studio

A local-first haircut-planning tool that combines front, left three-quarter, and right three-quarter portraits into a pose-corrected facial-shape profile. It returns five structurally similar **Labeled Faces in the Wild (LFW)** references, blended face-shape labels, measurements, and three deterministic haircut directions.

To publish or update this project using GitHub Desktop, follow [GITHUB_DESKTOP.md](GITHUB_DESKTOP.md). The dataset, generated index, MediaPipe model, local environments, and test artifacts are intentionally excluded from Git.

This is a structural comparison, not face recognition, identity verification, or a same-person probability. It uses no identity embeddings and makes no inference about protected traits.

## Quick start (macOS, Python 3.11)

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
uv run face-match setup-model
uv run face-match prepare-lfw --accept-dataset-notice
uv run face-match index --dataset data/lfw_funneled
uv run face-match serve
```

Open <http://127.0.0.1:8000>. After the one-time model and dataset downloads, matching, indexing, and image serving are entirely local. The model, LFW images, SQLite database, and biometric landmark vectors are ignored by Git and must not be committed.

Always open the local server URL. `src/face_match/templates/index.html` is a Jinja source template; opening that file directly with a `file://` URL cannot load the API, stylesheet, or rendered configuration values.

## Dataset notice and attribution

`prepare-lfw` always displays a notice and refuses to download unless `--accept-dataset-notice` is provided. LFW contains named people in photographs collected from the web. This project does not redistribute LFW and cannot grant rights to its underlying images. Review the [official LFW site](https://vis-www.cs.umass.edu/lfw/) before use and restrict use to lawful personal/research purposes.

Please cite:

- Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller, “Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments,” UMass Amherst Technical Report 07-49, 2007.
- Gary B. Huang, Vidit Jain, and Erik Learned-Miller, “Unsupervised Joint Alignment of Complex Images,” ICCV, 2007, for the funneled images.

The complete funneled archive is fetched from the checksum-pinned Figshare mirror used by scikit-learn (`b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a`). The app performs its own translation, full 3D pose, scale, and weighted Procrustes normalization.

## Methodology

MediaPipe Face Landmarker v2 runs locally in image mode with at most two detected faces and transformation matrices enabled. V2 retains 428 of the 468 base-mesh points: all irises and expression-sensitive lip points are excluded. The 36-point outer contour has weight `4.0`, forehead and cheek surface points weight `1.0`, and eyes, brows, and nose are low-weight (`0.25`) alignment anchors. Blendshapes and identity embeddings remain disabled.

Each view is centered, inverse-rotated with the MediaPipe 3D pose, uniformly scaled, and aligned to the front view with weighted Kabsch/Procrustes alignment. The coordinate-wise median forms one three-view mesh. Face-shape distance is:

```text
0.70 × RMS(explicit proportion descriptors)
+ 0.30 × weighted RMS(Procrustes-aligned dense mesh)
```

Descriptors cover length-to-width ratio, forehead/temple/cheek/jaw/chin widths, jaw taper and angularity, and facial thirds. Each identity is represented by its closest eligible frontal photo. Deterministic tie-breaking returns five distinct names. Shape labels are fuzzy styling aids, not scientific categories or probabilities. MediaPipe does not locate the true hairline, so the interface explicitly describes only upper-face and temple structure.

The primary workflow requires a front portrait with at most 10° yaw and opposite left/right three-quarter portraits at 15–35° yaw. Every view permits at most 12° pitch. The app reports corrective pose guidance when a slot is wrong or duplicated.

## Privacy and limits

- All three query uploads are read into memory, explicitly closed, used once, and released. No query path, bytes, vector, preference, request log, or history row is stored—including failure paths.
- No analytics, cookies, authentication, cloud inference, external APIs, or deployment components are present.
- Accepted formats: JPEG, PNG, WebP. Maximum upload: 8 MB. Dimensions: 96–6000 pixels per side.
- Zero faces and multiple faces are rejected with corrective guidance.
- Result images are served only when their database record resolves beneath the recorded local dataset root.

## Index behavior

SQLite schema v2 stores subjects, relative paths, dense float64 vectors, descriptors, overlay coordinates, yaw/pitch/roll, quality, match eligibility, exclusion reasons, model hash, and algorithm version. Schema v1 databases migrate in place without losing image records. The landmark-version change triggers a resumable recalculation. Detected but non-frontal or low-quality references remain indexed with an exclusion reason instead of being mislabeled corrupt.

## HTTP API

- `GET /api/status` — model readiness, version compatibility, and total/indexed/invalid/pending/identity counts.
- `POST /api/analyze` — multipart fields `front`, `left`, and `right`, plus optional `length`, `texture`, `effort`, and `goal`; returns the shape profile, diagnostics, five matches with component distances, and three haircut directions.
- `POST /api/match` — retained compatibility endpoint for one multipart `image`; it is no longer used by the main interface.
- `GET /api/images/{id}` — serves the best-matching local LFW image by database ID.

Every match contains `rank`, `identity`, `image_url`, raw normalized `distance`, derived `similarity`, a warning label, and normalized overlay coordinates.

## Validation

```bash
uv run ruff check .
uv run mypy .
uv run pytest
uv run python scripts/smoke_scale.py
pnpm install
pnpm exec playwright install chromium
pnpm exec playwright test
```

Tests cover full 3D invariance, improvement over legacy roll-only normalization, weighted Procrustes distance, three-view median/agreement, shape-label boundaries, preference filtering, schema migration, eligibility filtering, an independent NumPy oracle, resumption/idempotency, pose and duplicate rejection, malformed uploads, privacy, five unique matches, three haircut cards, dense overlays, and desktop/mobile layouts.

`smoke_scale.py` additionally exercises the real SQLite loading and matching path with 13,233 synthetic vectors across 5,749 unique identities—the published scale of LFW—and requires exact agreement with the independent oracle. It is a scale check, not a substitute for the consent-gated real-LFW smoke test below.

## Real three-view LFW smoke test

Choose one front and two opposite three-quarter photos. For a true held-out check, copy the database to a temporary location, delete those three image rows from the copy, set `FACE_MATCH_DB` to it, and run:

```bash
uv run python scripts/smoke_three_view.py front.jpg left.jpg right.jpg
```

The smoke requires five unique API results, exact API/direct-matcher agreement, identity/image agreement with an independently implemented weighted NumPy oracle, three diagnostics, three recommendations, no project-file changes, and median response time across three complete analyses below five seconds.

## Configuration and troubleshooting

Paths can be overridden with `FACE_MATCH_ROOT`, `FACE_MATCH_MODEL`, `FACE_MATCH_DB`, and `FACE_MATCH_DATASET`.

- **Model setup needed:** run `uv run face-match setup-model`.
- **Fewer than five identities:** finish indexing and check `/api/status` for invalid counts.
- **Required reindex:** rerun `index`; incompatible vector versions are never ranked.
- **MediaPipe install failure:** use Python 3.11 or 3.12 on a MediaPipe-supported macOS architecture and recreate `.venv` with `uv sync`.
- **Wrong pose:** use a straight-ahead front view and 15–35° opposite three-quarter views; keep pitch within 12°.
- **No face:** use a sharper portrait with the face unobscured and hair away from the jaw and temples.
- **Multiple faces:** crop to exactly one visible face.
