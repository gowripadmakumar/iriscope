# IRISCOPE

**Because someone had to count them.**

> *"How many visible lines are in your iris?"*

IRISCOPE is an intentionally useless but technically real computer-vision
application, built for a college hackathon. It takes a photo of your eye,
finds your iris, and counts the approximate number of visible radial
texture structures in it.

Nobody asked for this. That's exactly why it exists. And it actually works.

---

## Table of contents

- [What IRISCOPE actually measures](#what-iriscope-actually-measures)
- [How it works](#how-it-works)
- [System architecture](#system-architecture)
- [Technology stack](#technology-stack)
- [Installation](#installation)
- [Running the application](#running-the-application)
- [Using IRISCOPE](#using-iriscope)
- [IRIS MATCH](#iris-match)
- [Privacy](#privacy)
- [Screenshots](#screenshots)
- [Known limitations](#known-limitations)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Hackathon context](#hackathon-context)

---

## What IRISCOPE actually measures

There is no scientifically standardized quantity called "the number of
lines in an iris." IRISCOPE does not claim to count exact anatomical iris
structures, diagnose anything, or identify anyone.

What it actually computes is:

> **The number of detectable radial texture structures crossing a defined
> annular region of the visible iris, under fixed image-processing
> parameters.**

This is a real, deterministic computer-vision measurement — just not a
medical or biometric one. Every count, quality score, and similarity
percentage shown by the app comes directly from the image processing
pipeline described below; nothing is randomized, hardcoded, or invented
for effect.

---

## How it works

Every photo — whether uploaded or captured from a camera — goes through
the same nine-stage pipeline:

```
ORIGINAL
   │
   ▼
FACE + EYE DETECTION      Haar cascades locate the face, then the eyes
   │                      within the upper 60% of the face box
   ▼
EYE REGION                A padded crop around each detected eye
   │
   ▼
IRIS REGION               Pupil found via percentile-based darkness
   │                      thresholding; iris boundary found via a
   │                      simplified Daugman-style radial brightness
   │                      search
   ▼
MASK                      Reflections and eyelid/eyelash occlusion are
   │                      masked out, isolating usable iris texture
   ▼
NORMALIZED                The annular iris is unwrapped into a fixed
   │                      64×360 polar (rectangular) representation
   ▼
ENHANCED                  Bilateral denoising + CLAHE contrast
   │                      enhancement, then a masked Scharr edge map
   ▼
DETECTED STRUCTURES       360 angular columns are scored for edge
   │                      coverage; qualifying columns are grouped into
   │                      circular runs (with wraparound handling) and
   │                      merged/split into final structure candidates
   ▼
FINAL RESULT              Headline count + supporting metrics
```

Each stage's output image is real — nothing in the visualization is
staged or synthetic. If a photo fails partway through (no face, no
pupil, too much reflection, etc.), the pipeline stops at that stage and
returns an honest, specific error message instead of a fabricated
result.

### Metrics

- **Structure count** — the headline number of detected radial structures.
- **Analysis quality** — a 0–100% score reflecting how suitable the
  detected structures are (never called "accuracy," since this isn't a
  claim about anatomical correctness).
- **Usable iris area** — the fraction of the iris ring that survived
  masking (i.e. wasn't reflection, eyelid, or eyelash).
- **Structure density** — structures per 100° of *usable* iris arc, so
  partially occluded eyes aren't penalized for arc they never had a
  chance to contribute detections from.

---

## System architecture

```
Browser (upload or camera)
        │  one image, one HTTP POST
        ▼
Flask app (backend/app.py)
        │  decodes bytes in memory, never touches disk
        ▼
analyzer.analyze_iris()  ──────────────────────────────┐
        │                                               │
        ▼                                               │
face_eye_detection → iris_segmentation →                │
iris_normalization → line_detection                     │
        │                                               │
        ▼                                               │
JSON response (metrics + base64 result images)  ────────┘
        │
        ▼
Browser renders the result panel
```

Upload and camera capture are indistinguishable to the backend — both
arrive as one image file in a multipart POST body and are handed to the
exact same `analyze_iris()` function. There is no separate code path for
either input method. IRIS MATCH is built the same way: it simply calls
`analyze_iris()` twice (once per photo) and compares the results — it
introduces no new detection logic of its own.

---

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| CV / image processing | OpenCV (Haar cascades, no ML models) | Works fully offline, bundled with `opencv-python`, testable immediately, no GPU or network dependency — fits the target 8 GB / integrated-graphics laptop |
| Numerics | NumPy | Array operations for masks, polar unwrapping, gradient analysis |
| Backend | Flask | One synchronous endpoint is all this needs; no ASGI server or extra dependencies |
| Frontend | Vanilla HTML / CSS / JS | No build step, no framework — a static-file frontend served directly by Flask |

Full pinned versions are in [`requirements.txt`](requirements.txt):

```
opencv-python==4.13.0.92
numpy==2.4.4
flask==3.1.3
gunicorn==23.0.0
```

`mediapipe` was evaluated early on (for precise iris landmarks via
FaceMesh) but isn't a project dependency — see
[Known limitations](#known-limitations) for why, and what it would take
to switch.

---

## Installation

Requires Python 3.10+ (developed and tested on 3.12).

```bash
git clone <this-repo>
cd IRISCOPE
pip install -r requirements.txt
```

No GPU, no model downloads, no external API keys — everything runs
locally with what `pip install` gives you.

---

## Running the application

```bash
python run.py
```

Then open **http://127.0.0.1:5000/** in a browser.

By default the server also listens on your machine's network address
(useful for testing the camera flow from a phone on the same Wi-Fi —
Chrome/Safari require `localhost` or HTTPS for camera access, so
`127.0.0.1` works locally but a phone will need your machine's LAN IP
over plain HTTP to *not* have camera access, only upload). Flask's debug
mode is **off** by default, since leaving it on while bound to a network
address would expose Werkzeug's interactive debugger to anyone on the
network. Set `IRISCOPE_DEBUG=1` if you want it on for local-only
development:

```bash
IRISCOPE_DEBUG=1 python run.py
```

---

## Hosting

### Render

This repository includes `render.yaml`. Create a new Blueprint on Render and
select the repository; Render will install the Python dependencies and start
the Flask app with Gunicorn. The service listens on Render's `PORT` and uses
`/` as its health check.

### Vercel

This repository includes `vercel.json` and `api/index.py`. Import the
repository as a Vercel project with the default settings. Vercel routes both
the static frontend and `/api/*` endpoints through the Flask application.

The computer-vision work is CPU-bound and synchronous. Render is the better
fit for sustained or larger image workloads; Vercel is useful for a simple
serverless deployment and demonstrations subject to Vercel function limits.

---

## Using IRISCOPE

**Iris Analysis** — upload a JPG/PNG of a face, or use your camera to
capture one. IRISCOPE detects your eyes, analyzes each iris
independently, and shows the full stage-by-stage visualization plus a
final structure count for each eye that could be analyzed.

**Supported input**: JPG and PNG, either uploaded or captured live via
`getUserMedia`. Camera capture takes a single still frame — the app
never streams continuous video to the backend.

If an eye can't be analyzed (too much reflection, eyelid obstruction, no
face detected, etc.), IRISCOPE says so directly rather than guessing.

---

## IRIS MATCH

A second, explicitly fictional feature: submit two photos and get a
deterministic "compatibility" score based purely on how similar the two
irises' detected texture metrics are.

- **Texture Similarity**, **Radial Harmony**, and **Density Harmony**
  are each an average of two underlying metric-similarity scores
  (structure count, structure density, detection quality, usable area,
  etc.), all already computed by the main analysis pipeline.
- **Iris Compatibility** is the average of those three sub-scores.
- The score maps to a fixed, dryly humorous verdict string (e.g.
  "Suspiciously compatible.").

IRIS MATCH is **fully deterministic** — the same two photos always
produce the same score — and makes **no claim whatsoever** about
romantic compatibility, personality, relationships, health, or identity.
It's a fictional entertainment feature built on real image
measurements, and the UI says so directly.

---

## Privacy

- Uploaded and captured images are decoded and processed **entirely in
  memory** — they are never written to disk at any point in the request
  lifecycle.
- No external AI APIs, no third-party image-processing services, and no
  biometric database are used anywhere in the pipeline.
- IRISCOPE does not perform identity recognition or authentication of
  any kind — it has no concept of "whose eye this is," only "what does
  this eye's texture look like."
- The app's footer states this plainly to anyone using it.

---

## Screenshots

Not included in this repository yet. The development environment used to
build IRISCOPE has no browser binary available to capture real screenshots
in an automated way, and this project's own honesty principle rules out
staging fake ones. Before submission, add 3 real screenshots here (landing
page, an analysis result with the stage viewer open, and an IRIS MATCH
result) taken from a real run of the app on a normal machine.

## Known limitations

Documented honestly, not smoothed over:

- **Dark-iris/pupil separation.** The percentile-based pupil detector can
  merge a very dark iris and pupil into one blob, which sometimes narrowly
  fails the circularity check used to confirm a real pupil. This is the
  single biggest driver of the segmentation failure rate below.
- **No light-iris test coverage.** All test photos used during
  development have dark brown/brown irises. The pipeline's thresholds are
  brightness-relative by design (should generalize to hazel/green/blue/
  grey), but this has never actually been verified on a light-colored
  iris — flagged consistently since Module 1 and never dropped from this
  list.
- **Haar cascades struggle with non-frontal, motion-blurred, or very
  small faces.** They fail *safely* (report no eyes rather than a wrong
  box) rather than silently, but coverage on difficult angles is weak.
  MediaPipe FaceMesh (with `refine_landmarks=True`) was evaluated as a
  more precise alternative but couldn't be used in the original
  development sandbox — its Tasks API needs to download a model file at
  runtime, and that network request was blocked there. This remains a
  reasonable upgrade path on a machine with normal internet access.
- **No blur or lighting quality detector.** A naive whole-image
  Laplacian-variance blur score was tested and produced the *wrong*
  answer on the project's own test set (a busy background scored
  "sharper" than a genuinely blurred subject) — so it was deliberately
  left out rather than shipped broken. Real blur/lighting cases are
  currently only caught indirectly, if they also degrade segmentation
  enough to fail at a later stage.
- **Camera capture is code-reviewed but not device-tested.** This
  sandbox has no camera hardware, so `getUserMedia` itself has never been
  exercised against a real camera — only its backend-facing half (a
  captured frame is just a JPEG, indistinguishable server-side from an
  uploaded one). **This should be the first thing checked on a real
  device before a live demo.**
- **Measured detection rate on the project's own 6-photo test set**:
  4/6 photos have a detectable face+eyes, 3 of the resulting 8 eyes
  fully segment and produce a structure count (10, 12, and 12
  structures respectively). This reflects real, mostly-hard test cases
  (extreme gaze angles, near-closed eyes, double reflections) rather
  than typical selfie conditions — see `DEVELOPMENT_STATE.md` for the
  full per-image breakdown.
- **IRIS MATCH's similarity-scale constants** were tuned by inspection
  of this project's own observed metric ranges, not a formal
  calibration — reasonable for the current test set, but would benefit
  from a wider variety of real eyes if the project continued.
- **The "Analysis Quality" score doesn't detect blur.** Confirmed
  directly during Module 11's adversarial testing: a heavily blurred copy
  of a known-good photo still scored 95–99% quality while its structure
  count dropped from 12 to 6–7. This is the same gap as the missing blur
  detector above, made concrete with a specific measured example rather
  than left as a general statement.
- **No publicly redistributable test dataset yet.** See "Testing" above —
  the current `data/test_images/` photos can't ship in a public repo, and
  a replacement licensed set hasn't been built. This is a documentation/
  licensing gap, not a code defect.

None of these are hidden — they're the same limitations tracked in
`DEVELOPMENT_STATE.md` throughout development, carried forward here
rather than glossed over for the final writeup.

---

## Project structure

```
IRISCOPE/
├── README.md                    (this file)
├── DEVELOPMENT_STATE.md         # full module-by-module development history
├── requirements.txt
├── run.py                       # entry point: python run.py
├── backend/
│   ├── config.py                 # every tunable constant, centralized
│   ├── utils.py                  # image load/save/resize/decode helpers
│   ├── face_eye_detection.py    # Module 1 — face/eye detection
│   ├── iris_segmentation.py     # Module 2 — pupil + iris boundary + masks
│   ├── iris_normalization.py    # Module 3 — polar unwrap + texture enhancement
│   ├── line_detection.py        # Module 4 — radial structure counting
│   ├── analyzer.py              # Module 5 — orchestration + visualization
│   ├── app.py                   # Module 6 — Flask app, /api/analyze, /api/match
│   └── compatibility.py         # Module 7 — IRIS MATCH scoring
├── frontend/
│   ├── index.html                # upload/camera UI, both themes, IRIS MATCH tab
│   ├── style.css                 # IRISCOPE design system (MEDICAL CHART / AFTER-HOURS LAB)
│   └── script.js                 # upload flow, camera flow, result rendering
├── data/
│   └── test_images/              # real test photos + per-module debug output
└── tests/
    ├── test_module1.py           # manual visual test runner
    ├── test_module2.py
    ├── test_module3.py
    ├── test_module4.py
    ├── test_module5.py
    ├── test_module6.py           # Flask test-client runner
    └── test_module7.py           # Flask test-client runner (IRIS MATCH)
```

`iris_normalization.py` covers both polar unwrapping *and* texture
enhancement in one file rather than two, since project knowledge's
suggested `texture_analysis.py` split had no natural seam — normalization
and enhancement share the same short pipeline and the same per-eye data,
so a second file would just pass the same arrays back and forth.

---

## Testing

IRISCOPE uses manual, visual-verification test scripts rather than a
pytest-style assertion suite. Every module's script runs the real
pipeline against real photos and saves debug images that were actually
opened and inspected during development — not just checked for the
absence of exceptions. This matches the project's core testing
principle: **pass/fail counts alone are not sufficient; visual inspection
of the actual output is authoritative.**

A full Module 11 final testing pass — regression results, a determinism
check (same image run three times, all metrics compared), adversarial/
degenerate-input testing (blank, noise, oversized, blurred, washed-out
images), and a live-server verification — is written up in full in
[`TEST_REPORT.md`](TEST_REPORT.md).

**Important — test image licensing.** The images referenced below in
`data/test_images/` were carried over from early development and are
**not included in the public repository**. An audit during Module 11
found that three of them are official White House photographs whose
embedded copyright notice explicitly forbids manipulation and
redistribution — exactly what running them through this pipeline does —
and the remaining two are unlicensed OpenCV tutorial sample images. See
[`data/test_images/LICENSING_NOTE.md`](data/test_images/LICENSING_NOTE.md)
for the full finding. A properly licensed, publicly redistributable
`good/ challenging/ invalid/` test set (with a documented source and
license for every file, per `test_images/IMAGE_SOURCES.md`) is the one
piece of Module 11 not yet finished — see that note for what's needed to
close it out.

Run any module's test directly:

```bash
python tests/test_module1.py    # face/eye detection
python tests/test_module2.py    # iris segmentation
python tests/test_module3.py    # normalization + texture
python tests/test_module4.py    # radial structure counting
python tests/test_module5.py    # full pipeline + visualization
python tests/test_module6.py    # Flask app (upload, camera-equivalence, error handling)
python tests/test_module7.py    # IRIS MATCH (scoring, determinism, error handling)
```

Expected results against the bundled 6 test photos (deterministic —
re-running should reproduce these exactly):

| Stage | Result |
|---|---|
| Face + eye detection | 4/6 photos succeed |
| Iris segmentation | 3/8 detected eyes succeed |
| Normalization / structure counting | 3/3 of those eyes succeed, counts **[10, 12, 12]** |
| Web app (`/api/analyze`) | Matches the CLI results exactly over real HTTP |
| IRIS MATCH (`/api/match`) | Deterministic; self-match always scores 100 on every sub-score |

During Module 8, the full app was additionally exercised end-to-end in a
real headless browser (Playwright + Chromium): real file upload → real
network request to a real running server → rendered result, confirmed
identical to the CLI test output. Layout was checked at 320px, 375px,
420px, and 1280px viewport widths in both themes. Camera *permission and
capture* could not be tested against real hardware in this environment —
see [Known limitations](#known-limitations).

---

## Hackathon context

IRISCOPE isn't trying to win on the strength of its AI. It's trying to
win because:

- The premise is ridiculous — nobody needs to know how many lines are in
  their iris.
- The implementation is completely real — every number on screen comes
  from actual, inspectable, deterministic computer vision, not a
  placeholder or a random draw.
- The failures are as honest as the successes — when a photo can't be
  analyzed, IRISCOPE says exactly why instead of faking a result.

The ideal reaction is a judge asking "why would anyone build this?"
followed by "...wait, it actually works?"
