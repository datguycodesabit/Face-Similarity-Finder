# Face Shape Studio

A local-first haircut-planning tool that combines front, left three-quarter, and right three-quarter portraits into a calibrated 3D facial-shape profile. V3 compares one robust three-photo prototype per named person from a license-locked Wikimedia Commons pack, with 80% of the score assigned to outer structure and 20% to stable inner features.

To publish or update this project using GitHub Desktop, follow [GITHUB_DESKTOP.md](GITHUB_DESKTOP.md). The dataset, generated index, MediaPipe model, local environments, and test artifacts are intentionally excluded from Git.

This is a structural comparison, not face recognition, identity verification, or a same-person probability. It uses no identity embeddings and makes no inference about protected traits.

## V3 quick start (macOS, Python 3.11)

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
uv run face-match setup-model
uv run face-match check-mica
uv run face-match discover-commons
uv run face-match prepare-commons --accept-commons-notice
uv run face-match index-v3
uv run face-match serve
```

Open <http://127.0.0.1:8000>. `/api/status` lists each missing V3 setup step. The app never silently falls back to V2. After one-time setup, matching, indexing, and image serving are entirely local. Models, portraits, SQLite databases, and geometric vectors are ignored by Git.

Always open the local server URL. `src/face_match/templates/index.html` is a Jinja source template; opening that file directly with a `file://` URL cannot load the API, stylesheet, or rendered configuration values.

## Separately licensed MICA and FLAME setup

MICA is not a dependency of the distributable Python package. Read its [personal/noncommercial license](https://raw.githubusercontent.com/Zielon/MICA/master/LICENSE), obtain FLAME 2020 under its own account/license, and keep both outside Git. The Apple-CPU worker is `scripts/mica_worker.py`; it returns only FLAME shape coefficients, selected canonical vertices, and reprojection error. It never returns or persists MICA's internal ArcFace embedding.

The expected local layout is:

```text
models/MICA/
  .venv/bin/python
  data/pretrained/mica.tar
  data/FLAME2020/generic_model.pkl
```

For Apple Silicon, clone the official checkout, create a separate environment, and obtain the official weights:

```bash
git clone --depth 1 https://github.com/Zielon/MICA.git models/MICA
uv venv models/MICA/.venv --python 3.11
uv pip install --python models/MICA/.venv/bin/python torch torchvision mediapipe opencv-python-headless loguru trimesh yacs
mkdir -p models/MICA/data/pretrained
uvx gdown 1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt -O models/MICA/data/pretrained/mica.tar
export MICA_LICENSE_ACCEPTED=1
```

Download `generic_model.pkl` from the official FLAME 2020 portal into the path above. Do not put credentials or a license document in this repository. License acknowledgement can be supplied per terminal with `MICA_LICENSE_ACCEPTED=1`, or kept locally by creating the Git-ignored marker `models/.mica-license-accepted`. `uv run face-match check-mica` verifies readiness without downloading restricted assets.

## Commons pack and attribution

`discover-commons` queries Wikidata for a deterministic surplus of named people with primary portraits and Commons categories. This candidate list is not treated as accepted gallery data. Local MediaPipe/MICA curation must still produce a V3 manifest containing exactly 500 identities and three immutable Commons revisions per identity. Every final entry records the name, QID, page/revision, original upload URL, author, allowlisted license and URL, SHA-256, and display-image selection. `prepare-commons` remotely revalidates the locked metadata, refuses download without `--accept-commons-notice`, and reuses checksum-valid files on restart. Result cards link the author/license and Commons source.

Allowed licenses are public domain, CC0, CC BY 2.0–4.0, and CC BY-SA 2.0–4.0. Images must have one face, a short edge of at least 768 px, adequate jaw visibility, pitch within 12°, yaw within 35°, and at least one near-frontal view per identity.

Build the local pack and separate V3 index with:

```bash
uv run face-match curate-commons --accept-commons-notice
uv run face-match prepare-commons --accept-commons-notice
uv run face-match index-v3
```

Curation uses Wikimedia's standard API-provided 1440-pixel thumbnails for screening and downloads full originals only for accepted triplets. Its checkpoint under `data/commons_v3` is resumable and Git-ignored. If Commons returns a rate-limit or server error, the command stops without recording that infrastructure failure as a rejected person; wait for the cooldown and rerun the same command.

## Legacy V2/LFW recovery

The existing V2 database remains untouched and recoverable. Its setup commands are:

```bash
uv run face-match prepare-lfw --accept-dataset-notice
uv run face-match index --dataset data/lfw_funneled
```

`prepare-lfw` displays the legacy dataset notice and requires explicit acknowledgement. LFW is no longer used by the primary V3 workflow.

Please cite:

- Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller, “Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments,” UMass Amherst Technical Report 07-49, 2007.
- Gary B. Huang, Vidit Jain, and Erik Learned-Miller, “Unsupervised Joint Alignment of Complex Images,” ICCV, 2007, for the funneled images.

The complete funneled archive is fetched from the checksum-pinned Figshare mirror used by scikit-learn (`b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a`). The app performs its own translation, full 3D pose, scale, and weighted Procrustes normalization.

## V3 methodology

MediaPipe first validates face count, pose, crop quality, and stable landmarks. V3 converts its mixed normalized coordinates into isotropic camera space (`x × width`, `y × height`, `z × width`) and reruns structural extraction on a standardized 512 px face crop. The same real face padded onto square, portrait, and landscape canvases passes the 2% descriptor-invariance gate.

MICA reconstructs canonical FLAME identity geometry for each view. One-view shape outliers are rejected and the remaining two or three results are coordinate-wise median-combined. Comparison permits translation and rotation only—never scale or non-rigid fitting. Population descriptors use median/MAD normalization and 10% shrinkage-covariance whitening.

The fixed score is 45% jaw/chin FLAME surface, 20% cheek/temple/upper-face FLAME surface, 15% global proportions, 10% eyes/brows, and 10% nose/midface. Mouth geometry is excluded. All components are population-calibrated before weighting. Face-shape styling labels and their evidence come from reference-pack percentiles, not hard-coded cutoffs.

## Legacy V2 methodology

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

## V3 index behavior

V3 uses `data/face_match_v3.sqlite3`, separate from V2. It stores per-image MICA shape data, five canonical vertex regions, pose/quality, source-license metadata, identity prototypes, and calibration versions. Indexing is resumable by image hash, Commons revision, MICA engine version, and V3 algorithm version. Changing any of them rebuilds only affected identities. Adding extra photos cannot improve a score because ranking reads one fixed prototype per identity.

## Legacy V2 index behavior

SQLite schema v2 stores subjects, relative paths, dense float64 vectors, descriptors, overlay coordinates, yaw/pitch/roll, quality, match eligibility, exclusion reasons, model hash, and algorithm version. Schema v1 databases migrate in place without losing image records. The landmark-version change triggers a resumable recalculation. Detected but non-frontal or low-quality references remain indexed with an exclusion reason instead of being mislabeled corrupt.

## HTTP API

- `GET /api/status` — MediaPipe, MICA/license, Commons download/index, calibration, gallery version, and prototype readiness plus exact setup steps.
- `POST /api/analyze` — retains `front`, `left`, and `right` plus optional haircut preferences; returns `analysis_version`, MICA engine/gallery versions, warnings, percentile labels, diagnostics, five calibrated component distances, Commons attribution, jaw contours, and haircut directions.
- `POST /api/match` — retained compatibility endpoint for one multipart `image`; it is no longer used by the main interface.
- `GET /api/v3/images/{id}` — serves a checksum-locked local Commons display image. `/api/images/{id}` remains the legacy V2 route.

Every match contains `rank`, `identity`, `image_url`, raw normalized `distance`, derived `similarity`, a warning label, and normalized overlay coordinates.

## Validation

```bash
uv run ruff check .
uv run mypy .
uv run pytest
uv run python scripts/check_canvas_invariance.py data/lfw_funneled/George_W_Bush/George_W_Bush_0001.jpg
uv run python scripts/smoke_scale.py
pnpm install
pnpm exec playwright install chromium
pnpm exec playwright test
```

Tests cover isotropic canvas invariance, rigid-only FLAME alignment, outlier-resistant prototypes, calibrated fixed weights, descriptor independence, monotonic jaw sensitivity, manifest licensing/checksums, resumability, attribution, malformed uploads, no-retention success/failure paths, five unique results, and desktop/mobile V3 layouts. Legacy V2 regression tests remain enabled.

`smoke_scale.py` additionally exercises the real SQLite loading and matching path with 13,233 synthetic vectors across 5,749 unique identities—the published scale of LFW—and requires exact agreement with the independent oracle. It is a scale check, not a substitute for the consent-gated real-LFW smoke test below.

## Real V3 acceptance

After MICA/FLAME and the 500-person Commons pack are ready, use front and opposite 15–35° three-quarter photos:

```bash
uv run python scripts/smoke_three_view.py front.jpg left.jpg right.jpg
```

The real V3 smoke requires five unique prototypes, verifies the five-part/attribution contract, compares the API order with a fresh direct database ranking, hashes source and the V3 database across both success and failure paths, requires median three-view analysis below 20 seconds on the current Apple M5, and requires ranking below one second. The broader multi-identity freshness, same-person Jaccard, pose, feature-independence, and photo-count-bias gates run in the automated acceptance suite.

For real held-out validation, create at least five identity folders outside `data/`, with two independent triplets per person named `a-front`, `a-left`, `a-right`, `b-front`, `b-left`, and `b-right` (any supported extension), then run:

```bash
uv run python scripts/validate_v3_heldout.py /path/to/heldout-identities
```

This requires at least 90% distinct cross-identity top-five sets, mean same-person top-five Jaccard of 0.60, and a dimensionless within-person/between-person MICA pose ratio at least 40% below V2. These real held-out gates remain unproven until FLAME, the Commons index, and independent held-out photos are available; synthetic unit gates are regression coverage, not a substitute.

## Configuration and troubleshooting

Paths can be overridden with `FACE_MATCH_ROOT`, `FACE_MATCH_MODEL`, `FACE_MATCH_V3_DB`, `FACE_MATCH_V3_DATASET`, `FACE_MATCH_COMMONS_MANIFEST`, `MICA_HOME`, `MICA_WORKER`, `MICA_MODEL`, and `MICA_PYTHON`. The official MICA masking code hardcodes FLAME beneath its checkout, so `generic_model.pkl` must remain at `MICA_HOME/data/FLAME2020/generic_model.pkl`.

- **Model setup needed:** run `uv run face-match setup-model`.
- **Fewer than five identities:** finish indexing and check `/api/status` for invalid counts.
- **Required reindex:** rerun `index`; incompatible vector versions are never ranked.
- **MediaPipe install failure:** use Python 3.11 or 3.12 on a MediaPipe-supported macOS architecture and recreate `.venv` with `uv sync`.
- **Wrong pose:** use a straight-ahead front view and 15–35° opposite three-quarter views; keep pitch within 12°.
- **No face:** use a sharper portrait with the face unobscured and hair away from the jaw and temples.
- **Multiple faces:** crop to exactly one visible face.
