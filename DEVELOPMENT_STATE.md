# IRISCOPE — Development State

Last updated: after Module 11 (Final Hackathon Polish, Testing, Test
Dataset & Release). This header was stale for a while — it said "after
Module 8" even though Module 9 and Module 10's full sections were already
appended below it. Fixed during Module 11's audit; see that section for
what else the audit found.

This file is a factual snapshot of what's actually been built and tested.
It exists so development can resume in a fresh session without needing the
full conversation history. It is not a spec — see project knowledge for that.

---

## Current Status

**Module 1 (Computer Vision Foundation): COMPLETE and tested.**
**Module 2 (Iris Segmentation): COMPLETE and tested.**
**Module 3 (Normalization + Texture Analysis): COMPLETE and tested.**
**Module 4 (Iris Line Counter): COMPLETE and tested.**
**Module 5 (Results + Visualization): COMPLETE and tested.**
**Module 6 (Web App + Upload + Camera): COMPLETE and tested.**
**Module 7 (Design + IRIS MATCH): COMPLETE and tested.**
**Module 8 (Final Polish + README + Testing): COMPLETE.**
**Module 9 (Frontend Architecture Upgrade): EVALUATED, NOT MIGRATED — vanilla stack kept.**
**Module 10 (Interactive Visual Experience): COMPLETE and tested.**
**Module 11 (Final Hackathon Polish, Testing, Test Dataset & Release): COMPLETE, with one open item.**

IRISCOPE's implementation is finished and this is the final planned
module. The one thing not finished is a publicly redistributable test
image dataset — see Module 11's section below and
`data/test_images/LICENSING_NOTE.md`. Everything else (code audit,
regression/determinism/adversarial testing, cleanup, README) is done.

---

## Environment

- Python 3.12.3
- opencv-python 4.13.0.92 (also opencv-contrib-python and opencv-python-headless present, but code only depends on `cv2` generically)
- numpy 2.4.4
- mediapipe 0.10.33 is installed but **not currently used** (see "Rejected approach" below)
- Flask 3.1.3 (added in Module 6, for the web app — see that module's section)
- Playwright + Chromium were used **only** during Module 7 development, to
  actually render the redesigned frontend and click through it headlessly
  (no real browser exists in this sandbox otherwise). Not a project
  dependency — not in `requirements.txt`, nothing in the app imports it.
- Target machine: Windows, AMD Ryzen 3, 8 GB RAM, integrated graphics — no GPU dependency exists in the code

## Dependencies (requirements.txt)

```
opencv-python==4.13.0.92
numpy==2.4.4
flask==3.1.3
```

---

## Project Structure (actual, current)

```
IRISCOPE/
├── DEVELOPMENT_STATE.md         (this file)
├── requirements.txt
├── run.py                       # entry point: python run.py (Module 6)
├── backend/
│   ├── config.py                # all tunable constants (detection + segmentation + normalization + line detection + results/viz + web app)
│   ├── utils.py                  # image load/save/resize/decode helpers
│   ├── face_eye_detection.py    # Module 1 core logic
│   ├── iris_segmentation.py     # Module 2 core logic
│   ├── iris_normalization.py    # Module 3 core logic
│   ├── line_detection.py        # Module 4 core logic
│   ├── analyzer.py              # Module 5 core logic (orchestration + visualization)
│   └── app.py                   # Module 6 — Flask app + /api/analyze endpoint
├── frontend/
│   ├── index.html                # Module 6 — upload/camera UI (functional only, no final design)
│   ├── style.css                 # Module 6 — minimal layout CSS only
│   └── script.js                 # Module 6 — upload flow, camera flow, shared analyze() call
├── data/
│   └── test_images/             # 6 real test photos + output/, output_module2/, output_module3/, output_module4/, output_module5/ debug results
└── tests/
    ├── test_module1.py          # manual visual test runner for Module 1
    ├── test_module2.py          # manual visual test runner for Module 2
    ├── test_module3.py          # manual visual test runner for Module 3
    ├── test_module4.py          # manual visual test runner for Module 4
    ├── test_module5.py          # manual visual test runner for Module 5
    └── test_module6.py          # Flask test-client runner for Module 6 (frontend serving, valid/invalid uploads, upload==camera pipeline check)
```

No `texture_analysis.py` (folded into `iris_normalization.py` — see note
below) or `compatibility.py` yet — those belong to later modules (Module 7
IRIS MATCH) and have not been created.

Note on structure vs. project knowledge's suggested layout: project
knowledge lists `iris_normalization.py` and `texture_analysis.py` as
separate files. In practice, normalization (polar unwrapping) and texture
enhancement (denoise + CLAHE + edge map) are a single short pipeline with
no natural seam between them, sharing the same config section and the same
per-eye data flow — splitting them into two files would mean passing the
same intermediate arrays back and forth for no organizational benefit. They
were kept in one file (`iris_normalization.py`) per the project's stated
preference to simplify the suggested structure when appropriate.

---

## What Module 1 Actually Implemented

A pipeline that takes a BGR image and returns face + eye region locations.
Entry point: `detect_eyes(image)` in `backend/face_eye_detection.py`.

Pipeline steps:
1. Grayscale conversion + histogram equalization (`cv2.equalizeHist`) — chosen
   to normalize contrast across both dark and light skin/iris tones and
   uneven lighting, before detection.
2. Face detection via Haar cascade (`haarcascade_frontalface_default.xml`),
   largest detected face is selected as the subject if multiple are found.
3. Eye detection via Haar cascade (`haarcascade_eye.xml`), restricted to the
   **upper 60%** of the face bounding box only (`EYE_SEARCH_REGION_HEIGHT_RATIO`)
   to avoid false positives from nostrils/mouth.
4. Duplicate-eye merging: if two detections' centers are closer than
   `EYE_DUPLICATE_DISTANCE_RATIO * face_width`, the smaller one is discarded.
5. Left/right labeling: based on x-position **as seen in the image**
   (not the subject's anatomical left/right).
6. Padded crop box computed per eye (`EYE_CROP_PADDING_RATIO = 0.35`) for
   downstream iris work — this is wider than the raw eye detection box.

Returns a dict:
```python
{
  "success": bool,
  "error": str or None,          # "I can't find a face..." / "I can't find an eye..."
  "face_box": (x, y, w, h) or None,
  "eyes": [
    {"position": "left"|"right"|"unknown", "box": (x,y,w,h), "crop_box": (x,y,w,h)},
    ...
  ]
}
```

Helper functions also implemented:
- `get_eye_crop(image, eye)` → actual cropped pixel array using `crop_box`
- `draw_debug_overlay(image, result)` → annotated copy of the image using the
  IRISCOPE palette (coral = face box, teal = eye box) — dev/debug tool only,
  not part of the production output path.

`backend/utils.py` also implements `resize_if_large()`, which downscales any
image whose longest side exceeds `MAX_IMAGE_DIMENSION = 1024px` before
processing (performance/memory decision for the 8GB target machine).

---

## Important Technical Decisions

### CV approach: OpenCV Haar cascades (not MediaPipe)
MediaPipe FaceMesh was considered first because it would provide precise
landmarks (including iris landmarks with `refine_landmarks=True`), which
would have set up Module 2 nicely. It was **not used** because:
- The installed mediapipe 0.10.33 build only ships the `tasks` submodule
  (no legacy `solutions.face_mesh`).
- The Tasks API requires downloading a `.task` model file at runtime from
  `storage.googleapis.com`, which was unreachable in the dev sandbox
  (network policy blocked it — confirmed via direct test, HTTP 403 /
  `host_not_allowed`).
- Haar cascades are bundled with opencv-python, work fully offline, and
  could be fully tested immediately — consistent with the project's "test
  before declaring something works" principle and the privacy goal of
  local-only processing.

**This can be revisited.** On the actual Windows dev machine (with normal
internet access), MediaPipe FaceMesh may be worth adopting for Module 2
onward, since it would give iris/pupil landmarks directly instead of
requiring separate Hough-circle-based iris localization. This is an open
option, not a rejected one — it just couldn't be verified in this session.

### Eye labeling convention
Eyes are labeled "left"/"right" by their on-screen position, not the
subject's anatomical left/right (i.e. mirror convention, like looking at a
photo). Anyone modifying Module 2+ should keep this convention consistent
or explicitly change it everywhere at once.

### Centralized config
All thresholds (scale factors, min neighbors, size ratios, padding ratios,
colors) live in `backend/config.py`. No magic numbers elsewhere in the
codebase. New Module 2 constants should follow the same pattern.

---

## Tests Actually Performed

Manual visual test via `tests/test_module1.py`, run against 6 real (not
synthetic) photos in `data/test_images/`:

| Image | Face found | Eyes found | Notes |
|---|---|---|---|
| obama_small.jpg | Yes | 2 | Clean frontal headshot — best case |
| obama.jpg | Yes | 2 | Full-body shot, small face relative to frame |
| obama2.jpg | Yes | 2 | Close-up, head turned, mouth open — still succeeded |
| biden.jpg | Yes | 2 | Distant, head tilted — still succeeded |
| messi5.jpg | Yes (false positive on background) | 0 | Action shot, motion blur, turned head — failed safely (correctly reported no eyes rather than guessing) |
| fruits.jpg | No | 0 | No face present — correctly reported "I can't find a face" |

Result: **4/6 images fully succeeded.** Debug overlays and individual eye
crops were visually inspected (not just checked for exceptions) and
confirmed to be correctly positioned. Eye crops were confirmed to have
usable resolution/margin for a future iris pipeline (checked at both close
and distant face sizes).

No automated unit-test framework (pytest etc.) is in place yet — testing so
far is the manual visual script only.

---

## Known Limitations / Bugs / Weaknesses

- Haar eye cascade fails on non-frontal / motion-blurred / small faces (see
  messi5.jpg case). It fails *safely* (reports no eyes) rather than
  producing a wrong box, but coverage on difficult angles is weak.
- Face cascade can false-positive on background clutter/texture when the
  real face is small or blurry (also seen in messi5.jpg) — the face box
  itself is not always reliable in poor conditions, even though the overall
  result correctly failed at the eye stage.
- Eye crop resolution is lower for distant/small faces — not a bug, but
  Module 3 (texture analysis) will need to account for this when judging
  usable iris detail.
- No handling yet for: glasses, closed eyes, strong reflections/glare,
  multiple faces in frame (currently just picks the largest) — none of
  this was in Module 1's scope, but Module 2+ should expect these cases in
  real user photos.
- No automated test suite — only the manual script in `tests/test_module1.py`.

---

## What Module 2 Needs to Know

- Build on `detect_eyes()`'s output — specifically use `eye["crop_box"]` via
  `get_eye_crop(image, eye)` to get the padded eye-region image to work on.
  Do not re-detect faces/eyes from scratch.
- The crop already has ~35% padding around the raw eye detection, so the
  iris/pupil should be fully inside it for well-detected eyes, but Module 2
  should not assume the iris is perfectly centered.
- Keep using `backend/config.py` for new constants (e.g.
  `INNER_IRIS_RATIO`, `OUTER_IRIS_RATIO` per the project spec) rather than
  hardcoding values in a new module.
- Do not assume every image will have a usable face_box/eyes — Module 2's
  input will sometimes be a low-quality crop from a marginal detection
  (per the limitations above), so segmentation logic should degrade
  gracefully rather than crash.
- The MediaPipe-vs-Haar decision above is still open — if Module 2 starts
  and the developer has real internet access to test with, it's worth
  reconsidering FaceMesh's iris landmarks before committing further to a
  pure Hough-circle-based iris localization approach.

---

## File Change Log (this session)

Created:
- `backend/config.py`
- `backend/utils.py`
- `backend/face_eye_detection.py`
- `tests/test_module1.py`
- `requirements.txt`
- `data/test_images/*.jpg` (6 test photos)
- `DEVELOPMENT_STATE.md` (this file)

No files were modified or deleted from a prior state — this was the first
implementation session.

---

## Verification Session (new chat, before Module 2)

Re-ran `tests/test_module1.py` from a fresh environment against the existing
codebase (no code changes made — verification only).

**Result: identical to the recorded 4/6 success rate.** Re-inspected all 6
debug overlays and 2 sample eye crops visually (not just the pass/fail
counts):

- `obama_small`, `obama`, `obama2`, `biden` — face box, eye boxes, and the
  padded crop_box are all visually correct and sensibly positioned, including
  on the tilted/turned-head cases. Eye crops (`obama_eye_left.jpg` inspected
  directly) have enough resolution and margin around the iris/eyelid to be
  usable for Module 2 segmentation.
- `fruits.jpg` — correctly reports no face, no spurious boxes drawn.
- `messi5.jpg` — one correction to the state file's prior description: the
  face-cascade false positive lands on his **jersey/torso** (chest area),
  not "background clutter" as previously written. Still fails safely at the
  eye-detection stage (0 eyes reported, no wrong crop produced), so the
  overall behavior is unaffected — this is a documentation correction only.
- Determinism re-confirmed: running `detect_eyes()` twice on the same image
  produces an identical result dict.

**Known gap (not previously flagged):** all 6 test images have dark
brown/brown irises. There is currently no test image with a light iris
(hazel/green/blue/grey) in `data/test_images/`. Module 1 (face/eye box
detection) doesn't look at iris color at all, so this isn't expected to be a
Module 1 problem — but Module 2 onward should get at least one light-iris
test image before claiming light-iris support, per the project's "test
before declaring something works" rule.

**No code changes were made in this session.** Module 1 was confirmed
working as documented and left untouched.

Status after verification: Module 1 remains COMPLETE. Proceeding to
Module 2.

---

## Module 2 (Iris Segmentation) — COMPLETE and tested

Entry point: `segment_iris(eye_crop_bgr, eye_box_in_crop=None)` in
`backend/iris_segmentation.py`. Takes one eye crop from Module 1 and
returns pupil location, iris boundary, and a set of masks isolating usable
iris texture.

### Pipeline steps

1. Grayscale + light Gaussian blur (noise suppression before the
   percentile/gradient steps below).
2. **Pupil estimation**: threshold the darkest N% of pixels
   (`PUPIL_DARK_PERCENTILE`, percentile-based so it adapts to per-image
   brightness rather than assuming a fixed dark value), clean up with
   morphological open+close, then pick the most circular resulting blob
   within a plausible size range.
3. **Iris boundary estimation**: a simplified, single-axis version of the
   Daugman integro-differential operator. Assumes the iris is concentric
   with the pupil (documented simplification) and searches outward for the
   radius where average brightness (sampled around the circle) increases
   most sharply — the iris→sclera transition.
4. **Reflection masking**: small, very bright compact blobs (fixed
   brightness threshold — appropriate here since reflections are near-white
   regardless of iris color; size-capped so a light iris isn't masked out
   wholesale).
5. **Eyelid/eyelash masking**: samples "known-good" iris texture directly
   left/right of the pupil (least likely to be occluded regardless of gaze),
   then masks anything elsewhere in the iris ring that falls well outside
   that image-specific brightness range.
6. Final usable mask = (iris disc − pupil disc) − reflection mask − eyelid
   mask. `usable_area_ratio` = usable pixels / iris-ring pixels; below
   `MIN_USABLE_IRIS_AREA_RATIO` the whole segmentation is marked
   unsuccessful.

### Key fix made during this session (important for future sessions)

**Problem found during testing:** the first working version scored 4/8 eyes
as "successful," but visual inspection at full crop resolution showed most
of those were actually wrong — the pupil detector was frequently locking
onto the eyebrow instead of the real pupil (an eyebrow segment can be dark
and, after morphological cleanup, circular enough to beat a small/noisy
real pupil blob), and separately the iris boundary search sometimes latched
onto a stronger, more distant brightness jump (eyelid crease/eyebrow edge)
instead of the true, closer iris→sclera edge — inflating the iris circle
out into skin.

**Fix applied:**
- `estimate_pupil()` now accepts `hint_box`: Module 1's *raw* (unpadded) eye
  detection box, translated into crop-local coordinates. This is used both
  to hard-restrict the pupil search region (expanded by
  `PUPIL_HINT_BOX_MARGIN_RATIO`) and as the reference point for the
  center-distance preference, instead of the padded crop's geometric
  center.
- `estimate_iris_boundary()` now takes the *first* radius where the
  brightness-increase derivative crosses `IRIS_BOUNDARY_PEAK_THRESHOLD_RATIO`
  of the strongest jump found, rather than the single strongest jump
  anywhere in the search range — favoring the nearest plausible transition
  over a farther, sometimes-sharper one.
- `estimate_iris_boundary()` also accepts the same hint box to cap the
  maximum iris radius using the known eye-opening extent, but **only when
  doing so leaves enough radii for the smoothing window to work** — an
  earlier version of this cap caused a false failure on `biden.jpg` (right
  eye) by shrinking the search range to below the minimum usable span. This
  guard (`min_usable_span` check in the code) is important context for
  anyone tuning these constants further.

All fixes were verified by re-rendering pupil/iris circles at 6x crop
resolution and visually confirming pixel-level placement, not just by
re-reading the pass/fail summary count.

### Tests actually performed

Ran `tests/test_module2.py` against all 6 Module 1 test images (8 detected
eyes total across the 4 images where Module 1 finds a face+eyes). For every
eye, both the summary pass/fail *and* the actual debug visuals were
inspected — successes were re-rendered at 6x scale to confirm the pupil dot
and iris circle land in the right place, and failures were inspected by
viewing the raw eye crop directly to judge whether failing was the correct
call.

| Image / eye | Result | Visual verification |
|---|---|---|
| obama_small.jpg / left | Success, 84% usable | Pupil and iris circle both correctly placed at 6x zoom |
| obama.jpg / left | Success, 94% usable | Pupil correctly placed; iris circle slightly overreaches into the eyebrow at the top edge (see limitation below) |
| biden.jpg / right | Success, 98% usable | Pupil and iris circle both correctly placed at 6x zoom |
| obama.jpg / right | Failed (no pupil) | Correct call — two overlapping specular highlights bisect the pupil itself |
| obama2.jpg / left | Failed (no pupil) | Correct call — eye turned far to the side, iris mostly occluded by eyelid corner, only a small non-circular sliver visible |
| obama2.jpg / right | Failed (no pupil) | **Real limitation, not a bug** — dark brown iris + pupil form one merged dark blob under the percentile threshold; that blob's circularity (0.54) narrowly misses the 0.55 cutoff. A cleaner separation of pupil from a very dark iris needs more than a single darkness percentile. |
| biden.jpg / left | Failed (no pupil) | Correct call — eye is nearly closed/squinting, visible dark region is a thin non-circular sliver |
| obama_small.jpg / right | Failed (no iris boundary) | Pupil found correctly, but no reliable brightness transition was found outward from it |

**Result: 3/8 eyes fully segmented, all 3 verified pixel-accurate. The other
5 failures were individually inspected and are legitimate hard cases
(double reflection, extreme gaze angle, near-closed eye, dark-iris/pupil
merging), not silent wrong answers.** This is a meaningfully different (and
more honest) outcome than the pre-fix version, which had a higher pass
count built on wrong detections.

Determinism was preserved (no randomness introduced; same inputs still
produce identical outputs).

No light-iris test image was available this session either (same gap noted
after Module 1) — segmentation logic was designed to be brightness-relative
throughout (percentiles and sampled ranges, not fixed absolute thresholds)
specifically to make light-iris support plausible, but this has not been
visually confirmed on an actual light iris yet.

### Known limitations / bugs (Module 2)

- **Dark iris + pupil separation is unreliable.** When the iris is very
  dark brown, the percentile-based darkness threshold sometimes merges
  pupil and iris into a single blob rather than isolating the pupil alone
  (seen in `obama2.jpg` right eye). This is the most important open problem
  for Module 3 to be aware of — a size/darkness heuristic alone isn't
  always enough to draw the pupil/iris line on dark eyes.
- **Iris boundary can slightly overreach into the eyebrow** even after the
  fixes above, when the raw Module 1 eye box is generous enough that the
  hint-based cap doesn't bind (seen in `obama.jpg` left eye, ~94% usable
  despite this). The eyelid mask's left/right sampling approach doesn't
  reliably catch this because it only samples horizontally, not above/below
  the pupil.
- Pupil/iris are assumed concentric (same center). Real eyes have some
  natural pupil/iris decentration; this project accepts that simplification
  for speed and simplicity.
- No handling for glasses (not tested — no glasses photos available).
- Still no light-iris test image — flagged after Module 1, still true.
- No automated test suite (pytest etc.) — testing remains the manual visual
  script.

### What Module 3 needs to know

- Use `segment_iris()`'s `masks["usable_mask"]`, `iris_center`, and
  `iris_radius` as the input region for normalization/polar unwrapping —
  don't re-segment from scratch.
- Expect `segment_iris()` to fail (`success: False`) on a meaningful
  fraction of real eyes, even well-lit ones — Module 3 should handle that
  gracefully rather than assume it always has a usable region to unwrap.
- The dark-iris/pupil-merging limitation above may resurface in Module 3 as
  texture analysis picking up pupil pixels that leaked past segmentation —
  worth keeping in mind if radial structure counts look inflated on very
  dark irises specifically.
- `iris_center` currently always equals the pupil center (see concentric
  simplification above) — if Module 3's normalization ever needs a true
  iris-only center, that assumption is where to revisit.

---

## File Change Log (Module 2 session)

Created:
- `backend/iris_segmentation.py`
- `tests/test_module2.py`

Modified:
- `backend/config.py` — added the Module 2 configuration section (pupil,
  iris boundary, reflection, eyelid, and usable-area constants, plus Module
  2's debug visualization colors)

Not modified: `backend/face_eye_detection.py`, `backend/utils.py` — Module 1
was left untouched.

---

## Module 3 (Normalization + Texture Analysis) — COMPLETE and tested

Entry point: `normalize_iris(eye_crop_bgr, segmentation_result)` in
`backend/iris_normalization.py`. Takes one eye crop and Module 2's
`segment_iris()` output, and returns a fixed-size, texture-enhanced,
polar-unwrapped representation of the usable iris ring.

Requires `segmentation_result["success"] == True` — Module 3 has nothing
reliable to work with otherwise, and returns a clean failure dict
(`success: False`) rather than crashing if it's given a failed segmentation.

### Pipeline steps

1. **Polar unwrapping**: the annular iris region (between the pupil and
   iris boundaries found by Module 2) is unwrapped into a fixed
   `NORM_RADIAL_SAMPLES × NORM_ANGLE_SAMPLES` (64×360) rectangular image
   using `cv2.remap` with precomputed coordinate grids — columns are angle
   (0–360°, one column per degree), rows are radius (pupil-ward at the top,
   sclera-ward at the bottom). This is the key geometric property the whole
   module is built around: **a genuine radial iris structure becomes a
   roughly VERTICAL line in the unwrapped image**, which is what makes it
   detectable by a simple horizontal-gradient operator later, and is
   exactly what Module 4 will need.
2. **Annulus margin**: the region actually sampled is not the full
   pupil-to-iris span but `NORM_INNER_MARGIN_RATIO` (10%) to
   `NORM_OUTER_MARGIN_RATIO` (90%) of it, to avoid the blurriest parts of
   each boundary (partial-volume blur at the pupil edge; residual
   eyelash/sclera fringe at the outer edge).
3. **Mask unwrapping**: Module 2's `usable_mask` is unwrapped with the same
   coordinate grids (nearest-neighbor interpolation, since a binary mask
   shouldn't be blended) and re-thresholded back to strict 0/255. This
   carries eyelid/reflection exclusions through into the normalized space
   instead of losing them.
4. **Usability re-check**: `valid_ratio` (fraction of the normalized mask
   that's usable) is computed and checked against `NORM_MIN_VALID_RATIO`
   (0.15) — the narrower annulus sampled here means a segmentation that
   just barely passed Module 2's own check could still fail here.
5. **Texture enhancement**: bilateral filter (denoise while preserving
   edges — a plain Gaussian blur was avoided specifically because it would
   also blur away the texture edges Module 4 needs) followed by CLAHE
   (local contrast boost, chosen over global histogram equalization so it
   doesn't over-flatten already-decent contrast on one part of the ring
   just because another part is duller).
6. **Edge map**: a Scharr derivative in the x/column direction (i.e.
   horizontal gradient) computed on the enhanced image. Because radial
   structures are vertical lines in this representation, a horizontal
   gradient responds strongly to them specifically, while mostly ignoring
   horizontal texture (eyelash shadow bands, etc.) that runs the other way.
   This is texture *preparation*, not detection — no thresholding,
   candidate filtering, or counting happens here, per the module's scope.
   The map is masked (multiplied by the normalized usable mask) so
   Module 4 doesn't need to separately re-check which pixels are valid.

Returns a dict:
```python
{
  "success": bool,
  "error": str or None,
  "normalized_gray": np.ndarray or None,      # raw unwrapped grayscale, (64, 360)
  "normalized_mask": np.ndarray or None,      # unwrapped usable mask, (64, 360), 0/255
  "normalized_enhanced": np.ndarray or None,  # denoised + CLAHE, (64, 360)
  "normalized_edges": np.ndarray or None,     # masked Scharr-x magnitude, (64, 360)
  "valid_ratio": float or None,
}
```

### Tests actually performed

Ran `tests/test_module3.py`, which chains Module 1 → Module 2 → Module 3
across all 6 test images. Module 3 only runs on eyes where Module 2 already
succeeded — that's 3 eyes (`biden.jpg` right, `obama.jpg` left,
`obama_small.jpg` left), since Module 2's pass rate hasn't changed.

**Result: 3/3 normalized successfully.** For each, the debug panel (segmented
crop with circles → raw unwrapped → unwrapped mask → enhanced → edge map)
was visually inspected, not just the pass/fail summary:

| Image / eye | Valid ratio | Visual verification |
|---|---|---|
| biden.jpg / right | 99% | Clean unwrap, near-full valid mask, visible near-vertical texture striations carried through into the enhanced and edge-map stages |
| obama.jpg / left | 95% | The eyebrow overreach flagged in Module 2 (iris circle slightly into the eyebrow) shows up correctly as masked-out black patches in the unwrapped mask — the edge map correctly excludes those patches rather than treating eyebrow hair as iris texture |
| obama_small.jpg / left | 87% | Correctly unwrapped, but visibly lower detail than the other two — this crop comes from a small/distant face (lower source resolution), and that resolution ceiling carries through the whole pipeline; masked regions (eyelid intrusion) also correctly excluded |

Determinism was explicitly re-verified (not assumed): ran `normalize_iris()`
twice on the same segmentation result and confirmed `normalized_gray`,
`normalized_enhanced`, and `normalized_edges` were bit-for-bit identical via
`np.array_equal`.

Also confirmed CLAHE is doing real work, not just running for
show: on the `biden.jpg` right eye, pixel standard deviation went from
~20.2 (raw unwrapped) to ~31.8 (enhanced) — a real, measurable contrast
increase, not a rounding artifact.

Also confirmed graceful handling of Module 2 failures: passing a
`{"success": False}` segmentation result into `normalize_iris()` returns a
clean failure dict immediately rather than raising an exception.

**No light-iris test image was available this session either** — same gap
flagged after Modules 1 and 2. The normalization math (polar unwrap,
percentile-free CLAHE, adaptive bilateral filter) has no dark/light-specific
assumptions baked in, so light-iris support is plausible by design, but this
remains unconfirmed on an actual light iris.

### Known limitations / bugs (Module 3)

- **Output quality is capped by input crop resolution.** `obama_small.jpg`
  (the smallest/most distant face of the three Module-2 successes) produces
  a visibly blurrier normalized representation than the other two. This
  isn't a Module 3 bug — the 64×360 unwrap can't invent detail that wasn't
  in the source crop — but Module 4 should expect radial-structure counts
  to be less reliable on small/distant-face inputs.
- **CLAHE and the edge map operate on the whole normalized image, not
  per-valid-pixel.** Masked-out regions (eyelid/reflection) still
  participate in CLAHE's local contrast calculation for tiles they partially
  overlap, which could mean the tiles bordering a masked patch are slightly
  distorted relative to a tile with no occlusion at all. The final edge map
  is still correctly zeroed in masked regions, so this doesn't leak invalid
  texture into Module 4 — it's a minor, bounded quality concern in the valid
  regions immediately adjacent to a mask boundary, not a correctness bug.
- **The Module 2 dark-iris/pupil-merging limitation still blocks input to
  Module 3** — since Module 3 only runs on eyes Module 2 already segmented,
  the previously-documented 3/8 (now the same 3/3 that reached Module 3)
  bottleneck at Module 2 remains the main limiter on how much of the
  dataset Module 3 gets tested against, not anything new in Module 3 itself.
- No automated test suite (pytest etc.) — testing remains the manual visual
  script pattern established in Modules 1–2.
- Still no light-iris test image — flagged after every module so far.

### What Module 4 needs to know

- Use `normalize_iris()`'s `normalized_edges` as the primary input for
  radial structure detection — it's already denoised, contrast-enhanced,
  and masked, with radial structures appearing as roughly vertical lines.
  `normalized_enhanced` (before the edge/gradient step) is also available
  if Module 4 wants to try a different edge/line-detection approach than
  Scharr-x.
- `normalized_mask` marks which pixels are genuinely valid (0 = eyelid,
  reflection, or outside the sampled annulus). Module 4 should treat any
  detected "structure" that leans heavily on masked-out pixels as
  unreliable, even though `normalized_edges` is already zeroed there.
- Columns = angle (0–359, one per degree), rows = radius (0 = inner
  annulus edge near the pupil, 63 = outer annulus edge near the sclera).
  A real radial structure should span most/all of the 64 rows at a
  consistent column (or a narrow, slowly-drifting column band) — this
  "spans most of the radial extent" property is a natural filter Module 4
  can use to reject short/spurious edge fragments.
- Because the unwrap uses 360 columns (one per degree), Module 4's
  candidate structures will naturally be reported in terms of an angular
  position (0–359°) before merging/counting — convenient for the
  "duplicate merging" step described in the project spec.
- `valid_ratio` is already available per eye if Module 4 wants to factor
  overall image quality into its final structure count or confidence.

---

## File Change Log (Module 3 session)

Created:
- `backend/iris_normalization.py`
- `tests/test_module3.py`

Modified:
- `backend/config.py` — added the Module 3 configuration section (unwrap
  dimensions, annulus margins, mask threshold, minimum valid ratio,
  bilateral filter and CLAHE parameters)

Not modified: `backend/face_eye_detection.py`, `backend/iris_segmentation.py`,
`backend/utils.py` — Modules 1 and 2 were left untouched and re-verified to
still produce identical results (4/6 and 3/8 respectively) before Module 3
work began.

---

## Verification Session (new chat, before Module 4)

Restored the project from a handoff ZIP and re-ran `tests/test_module1.py`,
`test_module2.py`, and `test_module3.py` from a fresh environment before
touching any code.

**Result: identical to the recorded results in every case** — 4/6 (Module
1), 3/8 (Module 2), 3/3 (Module 3), same specific per-eye successes and
failures throughout. Also independently re-verified two claims by direct
computation rather than trusting the prior write-up: Module 3's determinism
(ran `normalize_iris()` twice, confirmed bit-for-bit identical arrays) and
its CLAHE contrast-increase figure (biden.jpg right: std 20.19 → 31.83,
matching the previously documented ~20.2 → ~31.8). Visually re-inspected
`biden_right_normalization.jpg` and `obama_left_segmentation.jpg` against
their written descriptions — both matched exactly, including the documented
eyebrow-overreach artifact on `obama.jpg` left.

**No code changes were made in this session.** Modules 1–3 were confirmed
working as documented and left untouched.

Status after verification: Modules 1–3 remain COMPLETE. Proceeding to
Module 4.

---

## Module 4 (Iris Line Counter) — COMPLETE and tested

Entry point: `count_radial_structures(normalization_result)` in
`backend/line_detection.py`. Takes Module 3's `normalize_iris()` output and
returns an approximate, deterministic count of detectable radial texture
structures, plus per-structure geometry and diagnostic information for
debugging/visualization.

Requires `normalization_result["success"] == True` — returns a clean
failure dict immediately otherwise, consistent with Modules 2 and 3.

### Operational definition of a detectable structure

Per project knowledge, IRISCOPE does not claim to count an anatomical
quantity. Module 4's precise, literal definition of what it counts is:

*A column-span (angular range) of the normalized edge map where, across
most of the visible annulus, the edge magnitude exceeds an adaptively
chosen threshold — and which corresponds to a genuine local intensity peak
if it shares a wide span with other such peaks.*

This is reported to the user as "N detectable radial structures," never as
an anatomical line count.

### Why a column-based approach

Module 3's polar unwrap makes a genuine radial iris structure appear as a
roughly **vertical** line (a column that's strong across most of its
rows), while circular/tangential features (eyelid creases, specular rings,
segmentation-boundary artifacts) appear as **horizontal** bands (strong in
a narrow row range, weak elsewhere). Working column-by-column and
requiring that a column be strong across *most of its rows*, not just
somewhere in the column, rejects the whole class of "this is actually a
circular boundary, not a radial one" false positives by construction, with
no separate orientation/Hough step needed. This was chosen over a general
line detector (Hough transform on the edge map, template matching, etc.)
specifically because it's simpler, deterministic, and this geometric
property already does most of the necessary false-positive rejection for
free.

### Pipeline steps

1. **Adaptive "on" threshold**: computed once per eye as the
   `LINE_EDGE_ON_PERCENTILE`th percentile (70th) of edge-magnitude values
   at valid (unmasked) pixels, floored at `LINE_EDGE_ON_MIN_VALUE` (10) so
   a near-flat, low-texture crop can't manufacture "structures" out of
   percentile math over near-zero noise. Percentile-based rather than a
   fixed brightness cutoff — consistent with Modules 2/3 — so the same
   logic works on both low-contrast and high-contrast iris texture without
   retuning per iris color.
2. **Per-column statistics**: for every one of the 360 angular columns,
   compute (a) `valid_row_fraction` — how much of that column is actually
   usable per Module 3's mask, (b) `coverage_ratio` — of the valid rows,
   what fraction are "on" (this is the length/continuity signal a genuine
   radial structure needs), and (c) `column_intensity` — mean edge value
   over *all* valid rows (a smooth, continuous signal used later for
   telling adjacent structures apart, see step 5).
3. **Column candidate test**: a column only counts as a line candidate if
   `valid_row_fraction >= LINE_MIN_VALID_ROW_FRACTION` (0.5 — don't judge a
   column that's mostly masked-out either way) **and**
   `coverage_ratio >= LINE_MIN_ROW_COVERAGE_RATIO` (0.55 — must be "on" for
   most of its valid rows, not just a short stretch).
4. **Circular run-finding + duplicate merging**: candidate columns are
   grouped into contiguous angular runs, correctly handling the fact that
   angle 359° is adjacent to angle 0° (see "Bug found and fixed" below).
   Runs separated by a gap of at most `LINE_MERGE_GAP_DEG` (2°) are merged
   into one, absorbing the case where a real structure's column briefly
   dips below threshold for a degree or two.
5. **Peak-splitting within wide runs**: a wide merged run isn't
   automatically one structure — dense, closely-packed texture can keep
   every column between two adjacent lines above threshold too, so the run
   never breaks on its own. Each merged run's `column_intensity` profile is
   lightly smoothed (`LINE_PEAK_SMOOTHING_WINDOW`, moving average of 3) and
   searched for local maxima; peaks closer than `LINE_PEAK_MIN_SEPARATION_DEG`
   (4°) apart are treated as one (the weaker one dropped via greedy
   non-max suppression). A run with only one dominant peak — the normal
   case — passes through unsplit. This step is what turns "one 49°-wide
   blob" into several individually-centered structures where the intensity
   profile actually supports it (see "Bug found and fixed" below for why
   this was added).
6. **Result assembly**: each final run becomes one reported structure with
   its angular center, width, average coverage, and average contrast.
   `detection_quality` (0-1) is the mean of Module 3's `valid_ratio` and
   the fraction of columns that were evaluable at all — explicitly a
   suitability/quality indicator, not an accuracy claim (per project
   knowledge, never call this "accuracy").

Returns a dict:
```python
{
  "success": bool,
  "error": str or None,
  "structure_count": int or None,
  "structures": [
      {"start_deg", "end_deg", "angular_width_deg", "center_deg",
       "coverage_ratio", "mean_contrast", "bridged_columns"},
      ...
  ] or None,
  "raw_candidate_count": int or None,     # distinct runs before merge/split
  "detection_quality": float or None,     # 0-1, suitability -- not accuracy
  "edge_threshold_used": float or None,
  "column_diagnostics": {...} or None,    # for the debug visualization only
}
```

### Bugs found and fixed during this session

**Bug 1 — wraparound merge failure.** The first working version found
candidate runs via an internal array rotation (to simplify the
0°/359° circular boundary), but the merge step afterward sorted runs by
their converted-back original-frame start value and merged left-to-right
without re-checking the one pair of runs that wrap through the seam. On
`biden.jpg` right, this produced two separate structures (`352°-358°` and
`1°-32°`) that are only 2° apart circularly — within the merge threshold —
and should have been one. **Fixed** by rewriting `_find_circular_runs` to
resolve the wraparound directly (merging the first and last scanned runs
when the array both starts and ends True) and adding one explicit
wraparound check to the merge step (`_merge_nearby_runs_circular`) for the
one pair the ordinary left-to-right pass can't see. Verified by re-running
the same case: the two runs now correctly merge into one
(`352° → 32°`, spanning the seam), reducing that eye's count from 7 to 6.

**Bug 2 (design gap, not a crash) — wide runs undercounting.** Initial
testing surfaced merged runs as wide as 49° on `obama_small.jpg` — clearly
not one physical radial line, but the coverage-threshold test alone had no
way to tell that apart from one very wide genuine structure, since dense,
closely-packed texture doesn't necessarily dip below the coverage
threshold between adjacent lines. **Fixed** by adding the peak-splitting
step (pipeline step 5 above). Verified this doesn't just add noise: on
`obama_small.jpg` left, the 49°-wide run split into 5 separate peaks whose
centers visually line up with distinct striations in the edge map
(inspected directly, not just trusted from the number); counts moved from
`[6, 7, 4]` to `[10, 12, 12]` across the three testable eyes, and the debug
overlay's radial tick marks visually match the density of visible texture
striations rather than looking arbitrary.

### Tests actually performed

Ran `tests/test_module4.py`, which chains Modules 1→2→3→4 across all 6
test images. Runs on the same 3 eyes Module 3 successfully normalizes
(`biden.jpg` right, `obama.jpg` left, `obama_small.jpg` left) — Module 4
has no new inputs of its own, so it inherits the same upstream bottleneck
documented after Modules 2 and 3.

| Image / eye | Structure count | Raw candidates | Detection quality | Visual verification |
|---|---|---|---|---|
| biden.jpg / right | 10 | 7 | 99% | Tick-mark overlay lines up with visible striations at 6x zoom; wraparound structure (spanning 352°→32°) correctly drawn as one continuous set of nearby ticks, not a gap |
| obama.jpg / left | 12 | 8 | 97% | Ticks cluster in the same angular region as the densest visible texture in the crop; the two black eyebrow-intrusion patches in the edge map correctly produce no ticks nearby |
| obama_small.jpg / left | 12 | 12 | 89% | Densest cluster of ticks corresponds to the same region that showed one 49°-wide run pre-fix; individual sub-peaks visually plausible given the striation density, though this is the lowest-resolution crop of the three and the hardest to be fully certain about at the pixel level |

**Result: 3/3 usable eyes produced a structure count, all inspected
visually via the debug overlay (not just the printed number).**

Determinism was explicitly re-verified: ran `count_radial_structures()`
twice on the same normalization result and confirmed the `structures` list
(not just the count) was identical both times.

Confirmed graceful handling of a Module 3 failure: passing
`{"success": False}` into `count_radial_structures()` returns a clean
failure dict immediately rather than raising an exception.

**No light-iris test image was available this session either** — the same
gap flagged after every prior module. Nothing in Module 4's logic depends
on iris color (it operates entirely on Module 3's already-normalized,
already-color-agnostic edge map), so light-iris support remains plausible
by design but is still not visually confirmed on an actual light iris.

### Known limitations / bugs (Module 4)

- **Peak-splitting is a heuristic, not a validated line-separation
  method.** It correctly turned one implausible 49°-wide blob into several
  more plausible narrower ones, and the result looks visually reasonable,
  but there's no ground-truth iris-line count to check it against (no such
  standardized quantity exists — see project knowledge). Treat the exact
  count as an approximate, order-of-magnitude figure, not a precise one.
- **Still bottlenecked by Module 2's dark-iris/pupil-merging limitation.**
  Module 4 only ever sees the 3 eyes that already made it through Modules
  2 and 3, so the dataset it's been tested against remains small. This
  isn't a new problem, but it does mean Module 4's own robustness across a
  wider variety of real eyes is comparatively less tested than Modules 1-3
  were at their own stage.
- **Column-by-column analysis has no explicit minimum-length-in-pixels
  check beyond the row-coverage ratio.** A very short iris ring (small
  `pupil_r`-to-`iris_r` span, e.g. from a distant/small face) still has 64
  normalized rows by construction (Module 3 always resizes to
  `NORM_RADIAL_SAMPLES`), so "spans most of the 64 rows" doesn't strictly
  guarantee a physically long structure on a low-resolution source crop —
  this compounds with `obama_small.jpg`'s already-lower detail (documented
  after Module 3).
- No automated test suite (pytest etc.) — testing remains the manual
  visual script pattern established in Modules 1–3.
- Still no light-iris test image — flagged after every module so far.

### What Module 5 needs to know

- Use `count_radial_structures()`'s `structure_count` as the headline
  number, `structures` for per-structure geometry if a detailed view is
  wanted, and `detection_quality` if the results page wants to show a
  suitability indicator alongside the count (never label it "accuracy").
- `draw_column_profile()` and `draw_structures_overlay()` in
  `line_detection.py` are debug/dev visualizations only, built for this
  module's own testing — Module 5's "Results + Visualization" work should
  design its own presentation-quality visualization rather than assuming
  these are production-ready as-is, though the underlying color roles
  (teal = accepted/detected, mustard = rejected/annotation, coral = major
  result) already follow the project's palette and can carry over.
- `edge_threshold_used` and `raw_candidate_count` are exposed mainly for
  transparency/debugging — Module 5 doesn't need to surface these to the
  end user unless it wants an "under the hood" detail view.
- Module 4 always operates on Module 3's already-masked `normalized_edges`
  and never needs the original eye crop for its core logic — only the two
  debug-drawing functions take the crop/segmentation result, purely to
  project results back onto the visible eye for human inspection.

---

## File Change Log (Module 4 session)

Created:
- `backend/line_detection.py`
- `tests/test_module4.py`

Modified:
- `backend/config.py` — added the Module 4 configuration section (edge-on
  threshold percentile and floor, valid-row-fraction and coverage-ratio
  minimums, merge-gap degrees, peak-separation and smoothing parameters,
  debug visualization colors)

Not modified: `backend/face_eye_detection.py`, `backend/iris_segmentation.py`,
`backend/iris_normalization.py`, `backend/utils.py` — Modules 1-3 were
re-verified to still produce identical results before Module 4 work began,
and were not changed during it.

---

## Verification Session (new chat, before Module 5)

Restored the project from a handoff ZIP and re-ran `test_module1.py` through
`test_module4.py` from a fresh environment before touching any code.

**Result: identical to the recorded results in every case** — 4/6, 3/8,
3/3, 3/3 (with counts `[10, 12, 12]`), same specific per-eye
successes/failures throughout. Independently re-verified Module 4's
determinism by calling `count_radial_structures()` twice on the same
normalization result and confirming the `structures` list was identical,
not just the count. Also independently re-verified `config.py` matches
every documented Module 1–4 constant.

**No code changes were made in this session.** Modules 1–4 were confirmed
working as documented and left untouched.

Status after verification: Modules 1–4 remain COMPLETE. Proceeding to
Module 5.

---

## Module 5 (Results + Visualization) — COMPLETE and tested

Entry point: `analyze_iris(image_bgr)` in `backend/analyzer.py` (plus
`analyze_iris_file(path)`, a convenience wrapper that loads from disk and
never raises). Runs Modules 1–4 end to end on one image and returns a
single, coherent analysis result: every pipeline stage image that was
actually produced, final metrics, and an honest, IRISCOPE-toned error
message wherever something didn't work.

**Important: Module 5 adds no new computer-vision detection logic.** Every
image, number, and pass/fail outcome in its result comes directly from
Modules 1–4, unchanged. This module's job is orchestration, error-message
translation, and visualization only.

### Result structure

```python
# analyze_iris() / analyze_iris_file()
{
    "success": bool,             # True if at least one eye fully analyzed
    "error": str or None,        # set only when nothing could be analyzed at all
    "stages": {"original", "detection_overlay"},   # shared, whole-image stages
    "eyes": [
        {
            "position": "left" | "right" | "unknown",
            "success": bool,
            "error": str or None,        # friendly message, only set on failure
            "failed_stage": str or None, # "crop" | "segmentation" | "normalization" | "counting"
            "stages": {
                # only the stages actually produced before any failure:
                "eye_region", "iris_region", "mask", "normalized",
                "enhanced", "detected", "final",
            },
            "metrics": {
                "structure_count", "detection_quality", "usable_area_ratio",
                "valid_ratio", "structure_density", "raw_candidate_count",
            },
        },
        ...
    ],
}
```

`build_stage_panel(eye_result, top_level_stages)` assembles whichever
stages exist (shared + per-eye) into one labeled vertical panel, skipping
any stage the pipeline never reached — a partial panel that stops exactly
where the real failure happened, not a padded-out fake one.

### Analysis stages (per project spec's 8-stage list)

1. **ORIGINAL** — the resized input image (`analyze_iris`'s own
   `resize_if_large` call, same as Module 1 already did internally).
2. **FACE + EYE DETECTION** *(not in the spec's numbered list, but kept as
   a shared stage since it's a real, already-existing debug output)* —
   Module 1's `draw_debug_overlay()`, unchanged.
3. **EYE REGION** — Module 1's `get_eye_crop()` output, unchanged.
4. **IRIS REGION** — Module 2's `draw_circles_overlay()` output (pupil +
   iris boundary circles on the crop), unchanged.
5. **MASK** — a new visualization (`_build_mask_visualization()`): the eye
   crop tinted by mask role directly from Module 2's own mask arrays —
   teal for usable iris texture, mustard for eyelid/eyelash exclusion, red
   for reflection exclusion. This is more informative than showing a
   single flat usable-mask, and every color comes from real detected
   pixels, not a stylized guess.
6. **NORMALIZED** — Module 3's `normalized_gray`, converted to BGR for
   display.
7. **ENHANCED** — Module 3's `normalized_enhanced` (denoise + CLAHE),
   converted to BGR.
8. **DETECTED STRUCTURES** — a new presentation version of Module 4's
   tick-mark overlay (`_build_detected_overlay()`): a 1px radial tick per
   detected structure plus both the pupil and iris boundary circles for
   context, drawn at the crop's native resolution.
9. **FINAL RESULT** — the headline card (`_build_final_card()`): the
   detected-structures overlay at a larger, readable size next to a text
   panel with the count and supporting metrics.

(Numbering above follows the project spec's 8-item list with "FACE + EYE
DETECTION" inserted as an extra shared stage — it isn't one of the spec's 8
but was cheap to keep since Module 1 already produces it.)

### Why the presentation overlay needed its own tuning, not just reuse

The first version of `_build_detected_overlay()` reused Module 4's own
`draw_structures_overlay()` styling almost exactly (2px lines + an end-cap
dot). Visual inspection showed this looked like solid teal wedges rather
than fine tick marks — because these eye crops are genuinely tiny (49×49
to 116×135px measured directly across the 3 successful test eyes), and a
panel that scales them up ~3–5× for display balloons any line thicker than
1px into a blob. **Fixed** by drawing at 1px with anti-aliasing and
dropping the end-cap dot, and by switching the panel's resize calls from
nearest-neighbor to linear interpolation (matches what Module 4's own test
script already did, and looked correct there). Re-inspected all 3
successful panels after the fix — ticks now read as fine radial marks
distinct from the pupil/iris circles, matching Module 4's own reference
visualization style. This is a display-only choice; the underlying tick
angles and count are Module 4's untouched output.

**Also fixed:** label bars on tall/narrow images (e.g. `obama2.jpg`'s
portrait crop, scaled down to a very narrow thumbnail) truncated longer
labels like "FACE + EYE DETECTION". `_add_label()` now measures the label
text and widens the whole stage (image + bar, cream-padded on the sides)
whenever the image is narrower than the label needs, instead of shrinking
the font to illegibility. Verified visually on `obama2_left_result.jpg`
before and after.

### Error handling — how technical errors become user-facing messages

Three small translation functions (`_diagnose_segmentation_failure()`,
`_diagnose_normalization_failure()`, `_diagnose_counting_failure()`) map
each stage's existing technical error string (already returned by Modules
2–4, unchanged) onto an IRISCOPE-toned message. These only reword errors
the pipeline already produces — **none of them add a new detection
capability**:

| Technical error (from Modules 2-4) | User-facing message |
|---|---|
| "Could not locate the pupil in this eye region." | "Your eye is here. Your camera is being difficult — I can't pinpoint your pupil in this photo." |
| "Could not locate a reliable iris boundary." | "I can see your pupil, but not a clear edge between your iris and the white of your eye. Try a sharper or closer photo." |
| "The usable iris area is too small..." (reflection-dominant) | "Too much reflection. Try moving away from the light." |
| "The usable iris area is too small..." (eyelid-dominant) | "Your eyelid or eyelashes are covering too much of your iris to inspect it properly." |
| "The usable iris area is too small..." (neither dominant) | "Not enough clear iris texture is visible in this photo." |
| "Iris ring too narrow to normalize reliably." | "The iris is too small to inspect properly." |
| "Not enough usable iris texture survived normalization." | "Not enough clear iris texture survived for reliable analysis." |
| "No valid iris texture to analyze." | "There isn't enough valid iris texture here to count anything reliably." |
| "I can't find a face in this image." / "I can't find an eye in this image." | passed through unchanged — already user-facing |

The reflection-vs-eyelid split for the "usable area too small" case is a
direct read of Module 2's own `reflection_mask` / `eyelid_mask` arrays
(which fraction of the iris ring each one covers) — genuine data, not a
guess — using two new config thresholds
(`RESULT_REFLECTION_DOMINANT_RATIO`, `RESULT_EYELID_DOMINANT_RATIO`, both
0.25).

**Deliberately not implemented:** a generic "excessive blur" or "poor
lighting" detector. Before adding one, I tested the obvious approach
(whole-image Laplacian-variance blur score) against all 6 real test images
and it produced the *opposite* of the correct answer: `messi5.jpg` — the
one image in the set with genuine motion blur on the subject — scored the
**highest** variance (786.7) of all 6 images, because a busy stadium
background has more total high-frequency detail than a clean portrait,
even though the face itself is blurred. Shipping this would have meant
telling users a blurred photo was fine. Per the project's "test before
declaring something works" rule, I did not include it. The spec's "no
face" / "no eye" / "iris too small" / "too much reflection" /
"eyelid obstruction" cases are all covered by genuine signals above;
"excessive blur" and "poor lighting" as standalone diagnoses are left
unimplemented rather than faked. "Unsupported image" is handled directly
by `utils.load_image()`'s existing `ValueError` (bad/corrupt file), caught
in `analyze_iris_file()`. Any other unexpected exception during analysis
is caught by a top-level try/except in `analyze_iris_file()` and turned
into a generic "something went wrong" message rather than crashing.

### Metrics shown

Only metrics the pipeline actually produces:
- **structure_count** — Module 4's headline number, unchanged.
- **detection_quality** ("Analysis quality" in the UI-facing card) —
  Module 4's own 0–1 suitability score, labeled per project knowledge's
  rule (never "accuracy").
- **usable_area_ratio** ("Usable iris area") — Module 2's own metric.
- **structure_density** — a new but simple derived metric:
  `structure_count / (valid_ratio × 360) × 100`, i.e. structures per 100°
  of *usable* iris arc (not per 360° of the whole circle, since occluded
  parts obviously can't contribute detections). Genuinely computed from
  two numbers Module 3/4 already produce, not invented for its own sake.
- **valid_ratio** and **raw_candidate_count** are kept in the `metrics`
  dict for transparency/debugging but are not put on the final card's
  headline stats — consistent with Module 4's own note that these are
  "under the hood" details, not something to always surface.

### Tests actually performed

Ran `tests/test_module5.py`, which calls `analyze_iris_file()` on all 6
test images (the full Module 1→2→3→4→5 chain) and saves a labeled panel
per eye (or per image, for a top-level failure).

**Result: exactly matches Module 4's own 3/3 successful eyes, same counts
(10, 12, 12), same quality/usable-area numbers** — confirms Module 5 didn't
alter any underlying pipeline behavior, only assembled and explained it.
All 10 panels (3 successes + 5 segmentation failures + 2 top-level
no-face/no-eye cases) were visually inspected, not just checked for
exceptions:

| Image / eye | Outcome | Visual verification |
|---|---|---|
| biden.jpg / right | Success, count 10 | Full 8-stage panel correct; final card shows 99% quality, 98% usable area, 2.8/100° density |
| obama.jpg / left | Success, count 12 | Mask stage correctly shows the documented eyebrow-overreach patch in mustard (excluded), matching Module 2/3's own notes exactly |
| obama_small.jpg / left | Success, count 12 | Full 8-stage panel correct despite lower source resolution; final card shows 89% quality |
| biden.jpg / left, obama.jpg / right, obama2.jpg / left+right | Failed at segmentation (pupil) | Panel correctly stops after "IRIS REGION" (circles absent) and jumps to a "FINAL RESULT" error card with the translated message; no "MASK" stage fabricated |
| obama_small.jpg / right | Failed at segmentation (iris boundary) | Panel correctly shows the pupil marker but no iris circle, then the error card |
| fruits.jpg | Top-level failure, no face | Panel shows only ORIGINAL + FACE+EYE DETECTION (no boxes) + error card; matches Module 1's own known behavior |
| messi5.jpg | Top-level failure, no eye | Panel correctly shows the documented face false-positive on his jersey, no eye box, then the error card |

Additional checks performed directly (not just read from the summary):
- **Determinism**: ran `analyze_iris_file()` twice on `obama_small.jpg` and
  confirmed every eye's `metrics` dict was identical (`==`), not just the
  headline count.
- **Unsupported/corrupt file**: wrote a fake `.jpg` (plain text bytes) and
  confirmed `analyze_iris_file()` returns
  `"I can't read this image. Try a JPG or PNG photo instead."` instead of
  raising.
- **Missing file**: confirmed a nonexistent path returns
  `"I can't find that image file."` instead of raising.
- **Untested reflection/eyelid diagnostic branch**: no current test image
  actually triggers Module 2's "usable area too small" failure path (all 5
  segmentation failures are pupil-not-found or iris-boundary-not-found —
  same as documented after Module 2), so
  `_diagnose_segmentation_failure()`'s reflection-vs-eyelid split has never
  run on real data. Sanity-checked it directly with synthetic mask arrays
  (a 36%-reflection case, a 49%-eyelid case, and a
  neither-dominant case) and confirmed all three branches return the
  correct message. **This is a known gap, not a hidden one** — same
  category as the long-standing "no light-iris test image" gap.
- Re-verified Modules 1–4 still produce byte-identical results after
  `config.py`'s Module 5 additions (4/6, 3/8, 3/3, 3/3 with counts
  `[10, 12, 12]`) — nothing in Modules 1–4 was touched.

### Known limitations (Module 5)

- **Reflection-vs-eyelid diagnostic message is untested on real photos** —
  see above. The logic is sound given what the masks represent, but no
  current test image exercises it.
- **No blur or lighting diagnosis** — deliberately not implemented after
  testing showed a naive approach gives wrong answers on this project's
  own test set (see "Deliberately not implemented" above). If a reliable
  signal is found later (e.g., a face-region-only or eye-crop-only blur
  score, rather than whole-image), it could be added then, but only after
  it's tested and confirmed correct on real images with known blur states.
- **Detected-structure tick marks are inherently coarse at native
  resolution** — these eye crops are small by nature of the source photos
  (49–135px). The 1px-line fix makes them read cleanly, but a crop from an
  even smaller/more distant face would still produce a visually blockier
  result. Not a Module 5 bug; a consequence of source image resolution
  Module 5 can't invent detail to fix.
- **Still bottlenecked by Module 2's dark-iris/pupil-merging limitation** —
  Module 5 only ever sees the same 3/8 eyes that already made it through
  Module 2, so most of its own error-path testing above necessarily comes
  from those 5 known failure cases plus the 2 no-face/no-eye images, not a
  wider variety of real photos.
- No automated test suite (pytest etc.) — testing remains the manual
  visual script pattern established in Modules 1–4.
- Still no light-iris test image — flagged after every module so far.

### What Module 6 needs to know

- `analyzer.analyze_iris_file(path)` is the single function Module 6's web
  app should call for the `/api/analyze` endpoint — it already handles
  disk I/O errors, the full Module 1–4 chain, and never raises. Pass it a
  saved upload path; it returns the full result dict described above.
- The result's `stages` images are OpenCV BGR NumPy arrays, not
  web-ready files yet — Module 6 will need to JPEG/PNG-encode them (e.g.
  `cv2.imencode`) before sending to the frontend, or write them to a temp
  location per the project's privacy rules (in-memory preferred; delete
  temp files after use if any are written).
- `build_stage_panel()` is the ready-made "show me everything" debug view
  (useful for an "under the hood" detail view Module 6 might want), but
  Module 6's actual results page should probably show `stages["final"]`
  prominently and let stages be expanded/inspected on demand, rather than
  always dumping the full 8-stage panel on the user.
- Per-eye results are independent — a photo with one good eye and one
  failed eye should probably still show the good result to the user
  (`result["success"]` is `True` if *any* eye succeeded), while the failed
  eye's own `error`/`failed_stage` is available if Module 6 wants to
  surface it too (e.g. "left eye analyzed, right eye: too much
  reflection").
- `result["error"]` at the top level is only meaningful when
  `result["success"]` is `False` — check per-eye `success`/`error` for
  partial-success cases, not just the top-level fields.
- Nothing in Module 5 talks to a browser, handles file uploads, or opens a
  camera — that's still entirely Module 6's job.

---

## File Change Log (Module 5 session)

Created:
- `backend/analyzer.py`
- `tests/test_module5.py`

Modified:
- `backend/config.py` — added the Module 5 configuration section
  (reflection/eyelid dominant-cause ratios, cream background color, panel
  layout constants for the multi-stage debug view and the final result
  card)

Not modified: `backend/face_eye_detection.py`, `backend/iris_segmentation.py`,
`backend/iris_normalization.py`, `backend/line_detection.py`,
`backend/utils.py` — Modules 1–4 were re-verified to still produce
identical results (4/6, 3/8, 3/3, 3/3 with counts `[10, 12, 12]`) before
Module 5 work began, and were not changed during it.

---

## Module 6 (Web App + Upload + Camera) — COMPLETE and tested

### Architecture

Flask, chosen over FastAPI for the simplest possible setup on this
project's target hardware — no ASGI server, no extra dependencies beyond
Flask itself, and the project only needs one synchronous endpoint. Vanilla
HTML/CSS/JS frontend, no build step, no framework — served directly by
Flask as static files.

`backend/app.py` is a thin route layer. It contains **no CV logic** — it
decodes the uploaded bytes (`utils.decode_image_bytes`, new in this
session) and immediately calls `analyzer.analyze_iris()`, the exact same
function `tests/test_module5.py` already exercised. Upload and camera
capture are indistinguishable to the backend: both arrive as one image
file in a multipart POST body.

`run.py` at the project root is the entry point (`python run.py`),
matching the project's suggested top-level structure. It just imports and
runs the Flask app from `backend/app.py`.

### API

**`POST /api/analyze`**

Request: multipart form with one field, `image` (a JPG or PNG file).

Response (200): JSON —
```json
{
  "success": true,
  "error": null,
  "eyes": [
    {
      "position": "left",
      "success": true,
      "error": null,
      "metrics": { "structure_count": 12, "detection_quality": 0.89, ... },
      "panel_image": "data:image/jpeg;base64,..."
    },
    ...
  ]
}
```
`panel_image` is Module 5's existing `build_stage_panel()` output
(the full labeled 8-stage debug/result view), JPEG-encoded and
base64'd for direct use in an `<img src="...">`. If no eye was even
reached (no face/no eye detected), `eyes` is `[]` and a top-level
`panel_image` is included instead, built the same way
`test_module5.py` does it for that case.

Error responses (400/413/500) use the same JSON shape with
`success: false`, an `error` string, and `eyes: []` — the frontend
doesn't need a different code path for "analysis failed on an eye"
vs. "the request itself was invalid."

### Upload flow

`<input type="file" accept="image/jpeg,image/png">` → client-side
preview via `FileReader` → on "Analyze", the `File` object is sent
as-is in a `FormData` to `/api/analyze`. No client-side resizing —
`analyze_iris()` already calls `resize_if_large()` internally, so
resizing only needs to happen once, server-side, matching every
other test path already verified in Modules 1–5.

### Camera flow

`navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })` →
live `<video>` preview → "Capture" draws the current video frame to a
hidden `<canvas>` → `canvas.toBlob("image/jpeg", 0.92)` produces one
still-image `Blob` → camera stream is stopped immediately after capture
(no continuous frame streaming, per spec) → "Analyze" sends that one
`Blob` through the same `FormData` → same `/api/analyze` call as upload.

Handled failure cases: `getUserMedia` unsupported, permission denied
(`NotAllowedError`), no camera device (`NotFoundError`), and a capture
attempted before the video element has produced a frame yet. Each shows
a plain-language message and points the user at the upload option
instead.

### Privacy implementation

- Images are decoded straight from the uploaded bytes in memory
  (`cv2.imdecode` via `utils.decode_image_bytes`) — **never written to
  disk** at any point in the request lifecycle.
- No external services, no third-party image APIs, no biometric
  database, no identity matching — the only computation is the existing
  local OpenCV pipeline.
- The page footer states plainly that images are processed in memory for
  experimental visual-texture analysis only, are not stored, are not used
  for identification/authentication, and this is not a medical tool
  (wording adapted from the project knowledge's suggested privacy text).

### How to run it

```
cd IRISCOPE
pip install -r requirements.txt
python run.py
```
Then open `http://127.0.0.1:5000/` in a browser. (Camera mode needs
either `localhost` or HTTPS — `getUserMedia` is blocked on plain HTTP
non-localhost origins by the browser itself; this doesn't affect local
testing at `127.0.0.1`/`localhost`.)

### Tests actually performed

This development sandbox has **no browser and no camera device**
(`/dev/video*` does not exist here), so `getUserMedia()` itself could not
be exercised end-to-end. Everything server-side and everything that can
be tested without a live browser was tested for real:

1. **`tests/test_module6.py`** (Flask test client, run for real):
   - `GET /` → 200, serves `index.html` with the IRISCOPE title.
   - `POST /api/analyze` for all 6 real test images → identical results
     to Module 5's baseline: `biden.jpg` (right eye 10 structures, left
     fails at segmentation), `obama.jpg` (left eye 12, right fails),
     `obama_small.jpg` (left eye 12, right fails), `fruits.jpg` /
     `messi5.jpg` (top-level detection failures) — same counts, same
     quality/usable-area/density numbers, same error messages as
     `test_module5.py` produced.
   - **Upload-vs-camera pipeline identity check**: the same JPEG bytes
     posted once as `photo.jpg` (upload-shaped) and once as `capture.jpg`
     (camera-shaped) — result dicts (metrics + errors, excluding the
     re-encoded image) compared and confirmed **identical**, directly
     verifying "both inputs eventually enter the same computer-vision
     pipeline."
   - Invalid-input handling, all confirmed:
     - missing `image` field → 400
     - empty filename → 400
     - wrong extension (`.txt`) → 400
     - right extension, corrupt/non-image bytes → 400 (caught by
       `cv2.imdecode` failing, not just the extension check)
     - oversized file (>15 MB) → 413
   - All passed. Full output in the session log.

2. **Real HTTP round trip** (actual `python run.py` process, actual
   `curl` requests over `127.0.0.1:5000`, not just the test client):
   - `GET /`, `GET /style.css`, `GET /script.js` → 200.
   - `POST /api/analyze` with `obama_small.jpg` as a real multipart file
     upload → 200, JSON response matched the test-client result exactly
     (left eye: 12 structures, 89% quality; right eye: fails at
     segmentation). The returned `panel_image` was base64-decoded back
     to a JPEG and visually inspected — it is the correct, real 8-stage
     result panel for that image, not a placeholder.
   - `POST /api/analyze` with a `.txt` file → 400 with the correct error
     message, over real HTTP.

3. **Camera mode**: code was written and reviewed for correctness against
   the MDN `getUserMedia`/`canvas.toBlob` APIs, and its *backend-facing
   half* is proven correct by the upload-vs-camera identity test above
   (a captured frame is just a JPEG `Blob`, indistinguishable server-side
   from an uploaded JPEG `File`). The *browser-facing half* — actually
   granting camera permission, seeing a live preview, and capturing a
   real frame — has **not** been tested in a real browser, because none
   is available in this environment. This should be the first thing
   manually checked in a real browser before relying on it.

### Known limitations (Module 6)

- Camera capture is untested in an actual browser (see above) — the code
  follows the standard `getUserMedia`/`canvas` pattern correctly, but
  hasn't been visually confirmed on a real device/camera.
- No automated test exercises `getUserMedia` failure paths (permission
  denied, no device) since that requires a real browser; those error
  branches are reviewed by inspection only.
- The frontend is deliberately unstyled beyond basic usability — no
  IRISCOPE color palette, typography, or layout polish. That's Module 7's
  job; building it now would mean redoing it once Module 7's design
  direction is applied.
- `build_stage_panel()`'s full 8-stage view is sent back for every result
  (success or failure) rather than a slimmed-down "final card + expandable
  detail" view Module 5 suggested as a possibility — kept simple for this
  module's scope; Module 7 may want to change what the results page
  actually displays.
- Flask's built-in dev server is used (`app.run(...)`) — fine for
  development/demo/hackathon use, explicitly not meant for production
  deployment (Flask itself warns about this on startup).
- No rate limiting or abuse protection on `/api/analyze` — acceptable for
  a local hackathon demo, would need attention before any public deploy.

### What Module 7 needs to know

- The web app is functionally complete: upload and camera both reach
  `analyze_iris()` through one endpoint, tested and confirmed identical.
  Module 7's job is **visual design only** — applying the IRISCOPE
  medical-chart palette/typography from project knowledge to
  `frontend/style.css` (and `index.html`'s structure/copy as needed) —
  not touching the CV pipeline or the API contract.
- `frontend/script.js`'s DOM structure (element IDs, the `renderResult()` /
  `buildEyeBlock()` functions) can be restyled freely via CSS; if Module 7
  needs new DOM elements (e.g. a fancier stage-by-stage stepper instead of
  one tall panel image), the `/api/analyze` JSON shape already has
  everything needed (`metrics`, `error`, `panel_image` per eye) — no
  backend change should be required for a pure redesign.
- IRIS MATCH (also Module 7) is a **new feature**, not a restyle — it
  will need its own backend piece (`backend/compatibility.py`, per project
  structure) and its own endpoint (`POST /api/match`, per project
  knowledge), taking two images through the existing per-eye pipeline and
  computing a deterministic similarity score. Nothing in Module 6 blocks
  this; `analyze_eye()`'s metrics (`structure_count`, `structure_density`,
  etc.) are already suitable inputs for a similarity calculation.
- The privacy note in the page footer is a first draft; Module 7 may want
  to move/style it, but the wording itself was carried over from project
  knowledge's suggested text and shouldn't need rewriting.

---

## Module 7 (Design + IRIS MATCH) — COMPLETE and tested

Two independent pieces: a visual redesign of the existing Module 6 app,
and a new IRIS MATCH feature. Neither touched Modules 1–5's CV pipeline.

### Visual design

Full IRISCOPE design system applied to `frontend/style.css` and
`frontend/index.html`, per project knowledge:

- **Both themes implemented as CSS custom properties**, switched by a
  `data-theme` attribute on `<body>`: MEDICAL CHART (light, default) and
  AFTER-HOURS LAB (dark). A theme toggle in the header switches between
  them and persists the choice in `localStorage` (`iriscope-theme`), so it
  survives a page reload. Default (no saved preference) follows the
  browser's `prefers-color-scheme`.
- **Typography**: Fraunces (headings/results), DM Sans (body), IBM Plex
  Mono (labels/measurements/counts), loaded via a Google Fonts `@import`
  with system-font fallbacks in every `font-family` stack (serif / sans /
  monospace) in case the page is ever opened offline.
- **Structural devices, not decoration**: bordered "panel" sections with a
  mono `panel-label` (e.g. "SPECIMEN INPUT", "READING") styled like a
  chart annotation; a "specimen box" preview frame with coral corner
  ticks (crop-mark style) around every photo preview; hairline dashed
  dividers between sections. No rounded corners, no drop shadows, no
  gradients, no glow anywhere — checked directly against project
  knowledge's "strictly avoid" list.
- Palette and both themes' exact hex values are centralized as CSS custom
  properties at the top of `style.css` — nothing hardcoded per-element.

**A real bug was found and fixed during this work**: several `hidden`
attributes (used throughout for mode toggling and result sections)
stopped working once the new stylesheet was in place. Cause: any rule in
an author stylesheet beats the browser's built-in `[hidden] { display:
none }` rule regardless of specificity, because origin (user-agent vs.
author) is resolved before specificity in the CSS cascade — several of
the new layout rules (`.specimen-box img { display: block }`, etc.)
matched hidden elements and this is what caused it. Fixed by restating
`[hidden] { display: none !important; }` explicitly near the top of
`style.css`. Confirmed fixed by screenshotting before/after with a
headless browser (see Testing below) — the broken-image-icon-overlapping-
placeholder-text symptom was visible before the fix and gone after.

### IRIS MATCH

**`backend/compatibility.py` (new)** — `match_irises(image_a_bgr,
image_b_bgr)`. Adds **no new detection logic**: it calls the existing
`analyzer.analyze_iris()` once per photo (the same function `/api/analyze`
already uses), picks each person's best-analyzed eye (highest
`detection_quality` among that photo's successful eyes), and compares six
numbers Module 5 already computes for that eye: `structure_count`,
`structure_density`, `detection_quality`, `usable_area_ratio`,
`valid_ratio`, and `raw_candidate_count` (the last two also used to derive
a "candidate density" — raw pre-filter edge candidates per 100° of usable
arc, the same normalization style Module 5 already uses for
`structure_density`).

**Scoring method** (fully deterministic, no randomness):
- Each of the six numbers gets a pairwise similarity score:
  `100 * exp(-|a - b| / scale)`, where `scale` is a fixed per-metric
  constant in `config.py` (the gap size at which similarity has decayed
  to ~37%). Identical values always score 100; similarity decays smoothly
  rather than dropping off a hard cliff.
- The six similarities combine (plain averages, no hidden weighting) into
  three named sub-scores:
  - **Texture Similarity** = avg(quality similarity, valid-ratio similarity)
  - **Radial Harmony** = avg(structure-count similarity, candidate-density similarity)
  - **Density Harmony** = avg(structure-density similarity, usable-area similarity)
- **Iris Compatibility** (the headline %) = average of the three sub-scores.
- The compatibility score maps to one of five fixed verdict strings from
  project knowledge (`config.MATCH_VERDICTS`), e.g. "Suspiciously
  compatible." for 75–89.
- If either photo has no successfully-analyzed eye, the match fails
  outright with a message identifying which photo (first/second) and why
  (reusing `analyze_iris()`'s own already-friendly error text) — no score
  is fabricated.

**`POST /api/match`** (added to `backend/app.py`): multipart form with
`image_a` and `image_b`. Validation mirrors `/api/analyze` exactly (same
allowed extensions, same in-memory decode), just applied to two fields
with per-field error messages. On success, returns:
```json
{
  "success": true,
  "error": null,
  "scores": {"compatibility": 81, "texture_similarity": 74, "radial_harmony": 87, "density_harmony": 82},
  "verdict": "Suspiciously compatible.",
  "eye_a": {"position": "left", "metrics": {...}, "card_image": "data:image/jpeg;base64,..."},
  "eye_b": {"position": "left", "metrics": {...}, "card_image": "data:image/jpeg;base64,..."}
}
```
`card_image` reuses Module 5's existing compact "final result card" build
(`analyzer._build_final_card`, already produced as part of `analyze_eye()`
— not reimplemented) rather than the full 8-stage debug panel, since a
side-by-side match view only needs the headline result per person.

**Frontend**: a new "Iris Match" tab (alongside "Iris Analysis") with two
photo input widgets ("Person A" / "Person B", each supporting upload and
camera — the same interaction pattern as the main flow) and a single
"Compare Irises" button. The result view shows the compatibility
percentage, the verdict text, three labeled similarity bars, both eyes'
result cards side by side, and a fixed disclaimer sentence stating the
score is fictional and based only on visual texture. To support three
near-identical input widgets (Analysis, Person A, Person B) without
tripling the JavaScript, `script.js` was refactored around one
`createCaptureWidget(prefix, onChange)` factory that each of the three
markup blocks (`""`, `"match-a"`, `"match-b"`) is wired up through — same
upload/camera logic, same failure handling, one implementation.

### Testing performed

1. **Regression — Modules 1–6 unaffected**: re-ran `test_module1.py`
   through `test_module6.py` after every backend change in this module.
   All still produce identical results to their documented baselines
   (4/6, 3/8, 3/3, 3/3, counts `[10, 12, 12]`, all Module 6 checks) —
   confirmed multiple times, most recently after the frontend rewrite.
2. **`tests/test_module7.py`** (new, Flask test client): a valid match
   between two real photos returns a full result with both card images;
   the same pair run twice produces byte-identical scores (determinism);
   the same photo matched against itself scores exactly 100 on every
   sub-score (sanity check on the similarity formula); a photo with no
   detectable face fails honestly with a specific message; invalid inputs
   (missing field, wrong extension, corrupt bytes) all return 400 with
   correct messages; `/api/analyze` and the frontend route are confirmed
   still served correctly through the same Flask app object. All passed.
3. **Direct `compatibility.py` testing** (before the API layer existed):
   confirmed the same determinism, self-match, and failure-path behavior
   at the Python level, independent of Flask.
4. **Real HTTP round trip** (`python run.py` + real multipart POST) for
   both `/api/analyze` and `/api/match`, not just the test client.
5. **Headless-browser visual/functional testing** (Playwright + Chromium,
   installed only for this development session — see Environment):
   - Screenshotted the redesigned app at mobile width (420px) in both
     light and dark theme, and at desktop width (900px) for the Iris
     Match two-column layout — all visually inspected directly (not just
     "it rendered without errors"), confirming the design matches project
     knowledge's palette/typography/no-glow direction in both themes and
     that the layout responds correctly at both widths.
   - Ran the actual browser-side bug described above to failure, then
     confirmed the fix by re-screenshotting.
   - Filled in two real test photos on the Iris Match tab, clicked
     "Compare Irises", and let the real browser make the real network
     request to the real Flask backend — the rendered result (81%,
     "Suspiciously compatible.", 74/87/82 sub-scores, both eye cards)
     matched the direct backend test exactly.
   - Ran the Iris Analysis tab the same way with a real photo — 12
     detectable structures on the left eye and the correct segmentation
     failure message on the right eye, matching every prior module's
     result for that image.
   - Verified the theme choice survives a real page reload
     (`localStorage`).
   - Camera mode's UI *states* (toggle, placeholder text, button
     show/hide) were verified by clicking through them in the headless
     browser. Actually granting camera permission and capturing a live
     frame could not be tested — same limitation as Module 6, no camera
     device exists in this sandbox.

### Known limitations (Module 7)

- Camera capture still hasn't been tested against a real camera in a real
  browser (carried over from Module 6) — now true for three capture
  widgets (Analysis, Person A, Person B) instead of one, since they share
  the same underlying code path. This should be the first manual check
  before a live demo.
- The Google Fonts `@import` requires the *browser* to have internet
  access when the page loads; if unavailable, the page falls back to
  system serif/sans/monospace fonts (tested — layout holds up, it just
  loses the specific typefaces). This has no effect on the Python
  backend, which needs no internet access at any point.
- IRIS MATCH's "best eye" choice (highest `detection_quality`) is a
  reasonable default but not the only defensible one — e.g. it doesn't
  consider whether the two people's chosen eyes are both left or both
  right eyes, which could matter for a more rigorous comparison. Not
  addressed here since project knowledge doesn't specify this level of
  detail and the feature is explicitly fictional/entertainment.
- The similarity-scale constants in `config.py`
  (`MATCH_COUNT_SCALE`, etc.) were chosen by inspection of this project's
  actual observed metric ranges (structure counts of 10–12, etc.) rather
  than any formal calibration — reasonable for the current test set, but
  should be revisited if a much wider variety of eyes gets tested in
  Module 8.
- No automated test drives the frontend's actual `fetch()` calls end to
  end other than the Playwright sessions run manually during this
  development session — there's no headless-browser test wired into the
  repeatable `tests/` suite (Playwright isn't a project dependency; see
  Environment). `tests/test_module7.py` covers the backend contract
  Playwright confirmed the frontend actually calls correctly.

### What Module 8 needs to know

- Both the Iris Analysis and Iris Match flows are fully working and
  styled end to end. Module 8 is about final polish, README, and broader
  testing — not new features.
- Manually test camera capture on a real device/browser first — it's the
  one piece of the whole project that has only ever been code-reviewed,
  never actually exercised (Modules 6 and 7 both).
- Broader test-image coverage (light-colored irises, glasses, different
  lighting) is still the single biggest known gap carried since Module 2
  — see that module's own known limitations. This affects both IRIS
  ANALYSIS and IRIS MATCH equally, since MATCH is built entirely on top
  of the same per-eye pipeline.
- `data/test_images/output_module7/` was not created — Module 7 has no
  batch image-output test in the Module 1–5 style, since its testing is
  API-response-shaped (`test_module7.py`) and visual (Playwright
  screenshots, not saved as project artifacts). If Module 8 wants
  before/after design screenshots for the README, they'd need to be
  captured fresh.

---

## File Change Log (Module 7 session)

Created:
- `backend/compatibility.py` — IRIS MATCH scoring (`match_irises()`)
- `tests/test_module7.py` — Flask test-client runner for `/api/match`
  (valid match, determinism, self-match, no-face failure, invalid inputs,
  regression check on `/api/analyze` + frontend)

Modified:
- `backend/config.py` — added the Module 7 configuration section
  (`MATCH_COUNT_SCALE`, `MATCH_DENSITY_SCALE`, `MATCH_QUALITY_SCALE`,
  `MATCH_USABLE_SCALE`, `MATCH_VALID_SCALE`,
  `MATCH_CANDIDATE_DENSITY_SCALE`, `MATCH_VERDICTS`)
- `backend/app.py` — added `POST /api/match` and its supporting helpers
  (`_load_match_image`, `_eye_summary_to_json`); `/api/analyze` and static
  serving unchanged
- `frontend/style.css` — full rewrite: IRISCOPE design system (both
  themes, typography, specimen-box/panel-label structural devices)
- `frontend/index.html` — full rewrite: restyled Iris Analysis markup +
  new Iris Match section + theme toggle + top-level nav
- `frontend/script.js` — full rewrite: generalized `createCaptureWidget()`
  factory reused by all three input widgets, theme persistence, nav
  switching, Iris Match result rendering; Iris Analysis behavior
  unchanged in substance (still one call to `/api/analyze`, same result
  data used)

Not modified: `backend/face_eye_detection.py`, `backend/iris_segmentation.py`,
`backend/iris_normalization.py`, `backend/line_detection.py`,
`backend/analyzer.py`, `backend/utils.py` — Modules 1–5 were re-verified
to still produce identical results (4/6, 3/8, 3/3, 3/3, counts
`[10, 12, 12]`) after every change in this session.

---

## File Change Log (Module 6 session)

Created:
- `backend/app.py` — Flask app, `/api/analyze` route, error handlers
- `run.py` — project entry point
- `frontend/index.html` — upload/camera UI markup
- `frontend/style.css` — minimal functional layout CSS (no IRISCOPE design system yet)
- `frontend/script.js` — upload flow, camera flow, shared `analyzeImage()` call, result rendering
- `tests/test_module6.py` — Flask test-client runner (frontend serving, valid/invalid uploads, upload==camera identity check)

Modified:
- `backend/utils.py` — added `decode_image_bytes()` (decodes raw bytes to
  a BGR array via `cv2.imdecode`, for in-memory processing of
  uploaded/captured images without touching disk)
- `backend/config.py` — added the Module 6 configuration section
  (`ALLOWED_UPLOAD_EXTENSIONS`, `MAX_UPLOAD_SIZE_BYTES`,
  `API_RESULT_JPEG_QUALITY`)
- `requirements.txt` — added `flask==3.1.3`

Not modified: `backend/face_eye_detection.py`, `backend/iris_segmentation.py`,
`backend/iris_normalization.py`, `backend/line_detection.py`,
`backend/analyzer.py` — Modules 1–5 were re-verified to still produce
identical results (4/6, 3/8, 3/3, 3/3, counts `[10, 12, 12]`) before Module
6 work began, and were not changed during it.

---

## Module 8 (Final Polish + README + Testing) — COMPLETE

### What was actually done

**A real bug was found and fixed, not just polish.** Both `run.py` and
`backend/app.py`'s `if __name__ == "__main__"` block called
`app.run(host="0.0.0.0", debug=True)`. Binding to all network interfaces
while debug mode is on exposes Werkzeug's interactive debugger — which
allows arbitrary code execution — to anyone who can reach the port. Fixed
by defaulting `debug=False` in both places, with an `IRISCOPE_DEBUG=1`
environment variable to opt back in for local-only development.
`host="0.0.0.0"` was kept as-is, since it's genuinely useful for testing
the camera flow from a phone on the same network — only the debug-mode
half of that combination was the actual risk.

**Frontend polish (`frontend/index.html`, `style.css`, `script.js`):**
- Added short camera-positioning guidance text ("fill the frame... avoid
  backlighting") to all three capture widgets (main analysis + both IRIS
  MATCH slots), shown only while the camera preview is live.
- Analyze/compare buttons are now force-disabled for the duration of the
  in-flight `fetch()` call, not just based on "is a file selected" —
  previously a fast double-click could fire two overlapping requests.
- Added `aria-live="polite"` to the status text, error text, and both
  result panels so screen readers announce results and errors as they
  appear.
- Added visible `:focus-visible` outlines (coral, matching the palette)
  on all buttons, tabs, and the theme toggle for keyboard navigation.
- Added one restrained fade/rise animation on result reveal, replayed on
  every new result and disabled entirely under `prefers-reduced-motion`.
- Added a small easter egg: clicking the "IRISCOPE" wordmark five times
  reveals "Count verified. Science-ish." for four seconds, then hides
  itself again. No visual hint that it exists.
- Small-screen refinements: tighter side padding, smaller heading size,
  narrower IRIS MATCH sub-score grid columns below 400px, so the layout
  holds up cleanly at 320px (previously only checked down to phone-width
  in the abstract, never a real narrow viewport).
- Added a `<meta name="description">` and a real `<title>`.

**Cleanup review performed on the backend** (`analyzer.py`,
`compatibility.py`, `face_eye_detection.py`, `iris_segmentation.py`,
`iris_normalization.py`, `line_detection.py`, `utils.py`, `config.py`):
checked every import for actual use, checked for leftover `print()`/
`TODO`/`pdb` debug statements (found and left exactly one intentional
`print()` in `analyzer.py`'s top-level exception handler — this is a
deliberate "this should never happen, but if it does, don't lose it
silently" safety net, not a debug leftover, so it was kept), and
confirmed `requirements.txt` lists only what's actually imported
(`mediapipe` was investigated in Module 2 but never became a dependency
and correctly isn't listed). No dead code, no unused functions, no
duplicated logic found. Nothing was removed.

**README.md created** — full user- and developer-facing documentation:
concept, honest operational definition, pipeline diagram, architecture,
tech stack, installation, running instructions (including the new
`IRISCOPE_DEBUG` flag), IRIS MATCH explanation, privacy statement, a
complete and specific known-limitations section, project structure, and
testing instructions. No accuracy percentages were invented; every
number quoted in the README is one that a test script actually produced
during this or an earlier session.

### Testing actually performed this session

1. Re-ran `test_module1.py` through `test_module5.py` after the backend
   changes (`run.py`, `backend/app.py`) — all five reproduced their
   documented results exactly: 4/6, 3/8, 3/3, counts `[10, 12, 12]`, and
   3/10 full end-to-end results, respectively.
2. Re-ran `test_module6.py` and `test_module7.py` — both still `PASS`,
   identical output to the Module 7 session (including the specific
   per-image failure messages, the invalid-input status codes, and the
   IRIS MATCH determinism/self-match checks).
3. Started the real Flask server (`python run.py`) and confirmed the
   startup log now reads `Debug mode: off` (previously would have read
   `on`), then drove it with Playwright + a real headless Chromium
   instance — the same tool used for design verification in the Module 7
   session, still present in this environment as a dev-only tool, not a
   project dependency:
   - Took full-page screenshots at 320px, 375px, and 1280px, in both
     light and dark theme, on both the Iris Analysis and Iris Match
     tabs. No overflow, clipping, or broken layout at any width.
   - Ran a real end-to-end request: uploaded an actual test photo
     through the real `#file-input` element, clicked the real Analyze
     button, and waited for the real `/api/analyze` response to render.
     Result matched the CLI test output exactly (left eye: 12
     structures, 89% quality; right eye: the same "not a clear edge
     between your iris and the white of your eye" failure message).
   - Ran a real IRIS MATCH request the same way with two different real
     test photos: 81% compatibility, "Suspiciously compatible.",
     matching sub-scores and per-eye panels rendered correctly
     side-by-side.
   - Confirmed via DOM inspection that the Analyze button re-enables
     itself after a request completes.
   - Confirmed the easter egg: clicked the wordmark five times, verified
     the hidden line becomes visible with the expected text.
   - Attempted `getUserMedia` against the camera-start button. This
     sandbox has no camera hardware, so this exercises the
     permission-request code path but not a real capture — see "What
     could not be tested" below.
4. Did **not** re-run the Playwright screenshot session against the
   `obama.jpg`/`obama2.jpg`/`fruits.jpg`/`messi5.jpg`/`biden.jpg` set —
   `test_module6.py`'s existing Flask-test-client run already covers
   those against the live `/api/analyze` route, and its output was
   confirmed unchanged (see point 2).

### What could not be tested (environmental limitation, not skipped)

- **Real camera capture end to end.** This development sandbox has no
  camera device. `getUserMedia`'s permission-request path was exercised
  and failed the way it should with no device present, but an actual
  photo captured from a live camera, on a real phone or laptop, has
  never been taken through this pipeline. This is the single most
  important thing to check by hand before a live demo — everything
  downstream of "you now have a JPEG" is proven to work (a captured
  frame is handled identically to an uploaded file), but the capture
  step itself is not.
- **Physical responsive testing on real devices.** Verified via
  Chromium's viewport emulation (320px/375px/1280px), which is a strong
  signal but not identical to a real phone's browser chrome, safe-area
  insets, or touch behavior.
- **Broad photo variety** (light-colored irises, glasses, varied
  lighting/blur/distance) remains untested beyond the project's original
  6-photo set, as it has been since Module 2. This was flagged, not
  newly discovered, and is called out explicitly in the README rather
  than being left implicit.

### Known limitations (final, as shipped)

All limitations tracked in earlier module sections of this file still
apply and were not resolved in Module 8 (that wasn't Module 8's job —
see the task's own instructions: fix only genuine problems, don't
redesign working code). The complete, current list is written out in
`README.md`'s "Known limitations" section rather than duplicated here;
that is now the authoritative honest-limitations statement for the
finished project. In summary: dark-iris/pupil merging, no light-iris
test coverage yet, Haar cascades' weakness on non-frontal/blurry faces,
no working blur/lighting quality detector (a naive one was tried and
scored the project's own test images backwards, so it was left out),
camera capture never device-tested, and IRIS MATCH's similarity-scale
constants being tuned by inspection rather than formal calibration.

### File Change Log (Module 8 session)

Modified:
- `run.py` — `debug=True` → `debug=False` by default, with
  `IRISCOPE_DEBUG` env var override; docstring updated to explain why
- `backend/app.py` — same debug-mode fix in its own `__main__` block
- `frontend/index.html` — camera guidance text (3 widgets), `aria-live`
  regions, `<meta name="description">`, real `<title>` (already present,
  confirmed unchanged), easter-egg markup
- `frontend/style.css` — `.camera-guidance`, `:focus-visible` rules,
  `.reveal` keyframe animation + `prefers-reduced-motion` override,
  `.easter-egg-line`, sub-400px media query
- `frontend/script.js` — guidance show/hide wired into the existing
  camera state machine, force-disable analyze buttons during in-flight
  requests, `replayRevealAnimation()` helper called from both result
  renderers, `initEasterEgg()`

Created:
- `README.md` — full project documentation (see above)

Not modified: `backend/face_eye_detection.py`, `backend/iris_segmentation.py`,
`backend/iris_normalization.py`, `backend/line_detection.py`,
`backend/analyzer.py`, `backend/compatibility.py`, `backend/utils.py`,
`backend/config.py` — the CV pipeline itself was not touched in Module 8.
Modules 1–7 were re-verified to still produce identical results after
every change in this session (see "Testing actually performed" above).

### What a future session should know

- The project is feature-complete per the original 8-module plan. Any
  further work is genuinely optional follow-up, not a missing module.
- If picked up again, the highest-value next step is manual device
  testing of the camera flow (see "What could not be tested" above) —
  everything else in the known-limitations list is a deeper CV-quality
  problem, not a missing feature.
- `README.md` is now the canonical source for anything a judge or new
  contributor would ask about the project; this file remains the
  canonical source for *how* it was built and what was actually tested
  at each step.

---

## Module 9 (Frontend Architecture Upgrade) — EVALUATED, NOT MIGRATED

### Task

Module 9 asked for an evaluation of whether the frontend should move to
React + Vite + Tailwind + Framer Motion "to support a highly polished,
interactive user experience," with an explicit instruction not to
rewrite anything unless the migration was genuinely justified.

### What was inspected

- `frontend/index.html` (195 lines), `frontend/script.js` (538 lines),
  `frontend/style.css` (668 lines) — re-read in full.
- `backend/app.py` — the two JSON API routes (`/api/analyze`,
  `/api/match`) that the frontend talks to.
- Module 6/7's own recorded architecture rationale (this file, Module 6
  section): Flask serving vanilla static files was a deliberate choice,
  not a placeholder.
- Re-ran `test_module1.py` through `test_module7.py` against a fresh
  extraction of the project archive to confirm nothing had drifted
  before making any decision.

### Decision: do not migrate to React

Reasoning, weighed against what the current frontend actually has to do:

- **No routing.** There are exactly two views (Iris Analysis / Iris
  Match), already toggled with a `hidden` attribute in `initNav()`.
  React Router or any client routing would add a dependency to solve a
  problem that's currently two `if` branches.
- **No non-trivial shared state.** Each capture widget (upload/camera)
  is already a self-contained unit via `createCaptureWidget()`, reused
  three times (analysis input, Match person A, Match person B) with zero
  duplicated logic. This is the same problem React components solve —
  it's already solved here without JSX, hooks, or a build step.
- **The API contract doesn't change.** Both endpoints take one
  `multipart/form-data` POST and return one JSON blob with base64 JPEGs
  embedded. A `fetch()` + `FormData` call is the entire integration
  surface — swapping it for `axios` or React Query would be added
  weight with no added capability.
- **Cost side of the ledger is real.** React + Vite + Tailwind + Framer
  Motion means a `node_modules` tree, a build step, a dev-server-vs-Flask
  split (or a build-and-copy-into-`frontend/`) step, and four new things
  to explain in a hackathon demo to a judge — for a project whose
  explicit design principle (Project Knowledge §20–21, and this file's
  own Module 6 notes) is to avoid frameworks unless they're clearly
  earning their place. None of the "React-shaped" problems (routing,
  shared state, component reuse) are actually present here.
- **Framer Motion specifically was requested for "highly polished,
  interactive" motion.** The existing `replayRevealAnimation()` +
  `prefers-reduced-motion`-aware CSS keyframe (added in Module 8)
  already covers the one motion moment IRISCOPE currently has (results
  fading in). Module 10 (visual polish) is the right place to add more
  motion if it's wanted — that's a CSS/JS question, not a reason to
  adopt a 40+ package animation library and its React dependency.

None of this means the current frontend is finished — `script.js` is a
single 538-line file, and if Module 10's visual polish work grows it
substantially, splitting it into a few plain ES modules (no framework
needed) would be a reasonable, much smaller-scope follow-up. That's a
file-organization question, not an architecture migration.

### What was NOT done (explicitly, per the "no unjustified rewrite" rule)

- No `package.json`, `vite.config.js`, or `node_modules` were created.
- No frontend files were rewritten, restructured, or moved.
- No backend/API changes — `app.py`'s two routes are untouched.
- No new ZIP was packaged, since no code changed as a result of this
  module; the existing project archive remains current.

### Testing performed this session

Fresh extraction of the project archive into a clean directory, then:
`test_module1.py` (4/6), `test_module2.py` (3/8), `test_module3.py`
(3/3), `test_module4.py` (counts `[10, 12, 12]`), `test_module5.py`
(3/10) — all matching this file's existing recorded results exactly, so
nothing regressed while sitting untouched between sessions.
`test_module6.py` and `test_module7.py` both reported `Overall: PASS`.

### Next module

Module 10 (visual/design polish) is next, per the original 8-module
plan's follow-on numbering. Camera testing on a real device remains the
single highest-value manual check outstanding (unchanged from Module 8).

---

## Module 10 (Interactive Visual Experience) — COMPLETE and tested

### Task

Turn the existing, working IRISCOPE frontend into a more polished,
interactive "eccentric scientific instrument" experience: landing-page
framing, a camera alignment guide, an honest in-progress indicator, a
strong result "readout," an interactive stage-by-stage viewer built from
real backend data, IRIS MATCH polish, restrained micro-interactions and
easter eggs — without redesigning the CV pipeline itself.

### Backend change (small, additive, no CV logic touched)

The one genuine backend change: the API previously returned each eye's
8 pipeline stages only as a single flattened composite JPEG
(`panel_image`, built by `build_stage_panel()`), which is unusable for
building tabs, a zoom view, or a stage-by-stage compare — the individual
images already existed inside `analyze_eye()`'s `stages` dict, just
never surfaced separately.

- **`backend/analyzer.py`**: added `get_stage_entries(eye_result,
  top_level_stages=None)`, which returns the same real per-stage images
  `build_stage_panel()` already assembles, individually, as
  `[{"key", "label", "image"}, ...]` in the documented stage order.
  Stages that don't exist because the pipeline stopped early are
  skipped — same honesty rule `build_stage_panel()` already followed.
  No new image processing; this only exposes data that already existed.
- **`backend/app.py`**: `_eye_to_json()` now also encodes and returns
  `stages` (via a new `_encode_stage_entries()` helper) alongside the
  existing `panel_image` (kept for backward compatibility). The
  top-level no-face/no-eye failure branch does the same with whatever
  stages exist at that point (typically just `original` +
  `detection_overlay`).
- **`tests/test_module6.py`**: `strip_images()` now also excludes the
  new `stages` field from the upload-vs-camera byte comparison, for the
  same reason it already excluded `panel_image` (base64 JPEG re-encode).

This is the only backend edit. `analyze_iris()`, `analyze_eye()`, and
every actual computer-vision function are unchanged.

### Frontend — what was built

Deliberately kept to vanilla HTML/CSS/JS, per Module 9's already-recorded
decision not to adopt a framework. Module 10's brief mentioned Framer
Motion "where it meaningfully improves transitions" — since that library
is React-only and the project is intentionally staying framework-free
(Module 9), the same transition effects (cross-fades, count-up, bar
fills) were built directly with CSS transitions + `requestAnimationFrame`
instead. Same visual result, zero new dependencies.

**Landing / instrument framing**
- Added an instrument eyebrow line above the wordmark: "Experimental
  iris texture analysis · Instrument No. 01."

**Camera experience**
- Added an SVG alignment guide (dashed coral ellipse + teal corner
  ticks) overlaid on the live video feed in all three camera widgets
  (Iris Analysis, Specimen A, Specimen B), plus a small status pill
  ("Camera live — align eye" → "Frame captured"). Purely visual
  (`aria-hidden`); the existing text guidance paragraph still carries
  the actual instructions for screen-reader users.

**Analysis-in-progress indicator**
- The single `/api/analyze` request is one round trip with no
  server-sent progress — there is nothing to stream. Built a client-side
  cycler (`startStageCycler`) that steps through the pipeline's real,
  documented stage names ("01 Locating eye" → ... → "07 Finalizing
  count") once every 550ms while that one request is in flight: an
  honest "here's the procedure" indicator, not a claim that the backend
  is reporting live per-stage completion.
- Added the suggested aside line ("Counting things nobody asked us to
  count.") beneath it.

**Result experience**
- Replaced the old one-line text summary with a proper instrument
  readout: a large Fraunces "Iris Structure Count" hero number that
  counts up to the real `metrics.structure_count` value
  (`requestAnimationFrame`-driven, skipped entirely under
  `prefers-reduced-motion`, and skipped/set-instantly for `0`/falsy
  values), plus a metrics block (Analysis quality, Usable iris area,
  Structure density) — all real values already in the JSON response,
  never invented.
- Added the easter egg "Congratulations. You now know something
  completely unnecessary." beneath the hero number, shown once per page
  load (not repeated per eye/per re-analysis, to keep it sparing).

**Interactive stage viewer (the main addition)**
- Built `buildStageViewer()`: a labeled tab row (one tab per stage that
  actually exists for that eye, from the new `eye.stages` array) plus a
  single image display with a cross-fade on tab switch and a caption in
  "Fig. 0N — LABEL" form. Defaults to the "Final Result" tab.
- Added a shared zoom modal (`#stage-modal`): clicking the image or its
  "Enlarge" button opens a larger view with real image scaling (not just
  the native pixel size), a caption, Escape-to-close, backdrop-click-to-
  close, and focus moved to the close button on open and back to the
  trigger element on close.
- The top-level failure case (no face/eye found) now also uses this same
  viewer against whatever stages exist (usually just Original + Face/Eye
  Detection), instead of one static composite image — more useful even
  when only 1–2 stages were produced.
- Per-eye failures (segmentation/normalization/counting) show the same
  reduced-stage viewer, so a partial failure is visibly partial rather
  than hidden behind one flattened image.
- Added the suggested aside "The iris has declined to participate." next
  to every failure message (top-level and per-eye), alongside — never
  replacing — the real, specific error text.

**IRIS MATCH**
- Added the "Iris Match" / "A completely unnecessary comparison of two
  irises." heading block above the existing (still-accurate) intro
  paragraph.
- Renamed "Person A" / "Person B" to "Specimen A" / "Specimen B"
  throughout (panel labels and the JS-built result cards), matching the
  instrument framing used everywhere else.
- Subscore bars (Texture/Radial/Density) now animate from 0 to their
  real value with a CSS transition (double-`requestAnimationFrame` to
  guarantee the 0% state paints before the transition starts — the same
  technique the existing `replayRevealAnimation()` reflow trick already
  used elsewhere in this file), instead of appearing pre-filled.
- Added the suggested closing line: "Scientific significance:
  questionable. Entertainment value: considerable. No peer-reviewed
  relationship conclusions were reached." — appended after the existing
  factual disclaimer, not replacing it.

**Accessibility / performance**
- All new animations (count-up, stage cross-fade, subscore bar fill,
  status-dot pulse) are gated on `prefers-reduced-motion` and fall back
  to an instant, correct end state.
- New interactive elements are real `<button>`s (keyboard-operable by
  default); the stage tab row uses `role="tablist"`/`role="tab"` with
  `aria-selected` kept in sync; the zoom modal uses
  `role="dialog"`/`aria-modal` and manages focus on open/close.
- No new dependencies, no new network requests beyond the two existing
  API calls; the modal and stage viewer reuse images already present in
  the same JSON response.

### Files changed

- `backend/analyzer.py` — added `get_stage_entries()`.
- `backend/app.py` — `_eye_to_json()` and the no-eyes branch now also
  return `stages`; added `_encode_stage_entries()`.
- `tests/test_module6.py` — `strip_images()` now also excludes `stages`.
- `frontend/index.html` — instrument tag, camera guide/status markup
  (×3 widgets), restructured status block, Iris Match heading, Specimen
  A/B labels, shared zoom-modal markup.
- `frontend/style.css` — styles for all of the above (camera guide,
  status block, result hero/metrics, stage viewer + tabs, zoom modal,
  match heading, animated subscore bar, failure aside), all theme-aware
  and `prefers-reduced-motion`-aware.
- `frontend/script.js` — camera guide/status wiring in
  `createCaptureWidget()`; `startStageCycler()`; rebuilt
  `renderAnalyzeResult()` / `buildEyeBlock()` / `buildFailureBlock()`
  around the new hero, metrics, and `buildStageViewer()`;
  `animateCountUp()`; `openStageModal()` / `closeStageModal()` /
  `initStageModal()`; animated `buildSubscoreRow()`; Specimen A/B
  relabeling and the new scientific-note line in `renderMatchResult()`.

### Tests performed

- Re-ran `test_module1.py` through `test_module7.py` after every
  backend edit — all seven still match this file's recorded baselines
  exactly (4/6, 3/8, 3/3, `[10, 12, 12]`, 3/10, PASS, PASS). No
  regressions from adding the `stages` field.
- `node --check frontend/script.js` — syntax-checked clean.
- Live-server + Playwright/Chromium visual and functional testing
  (headless, real HTTP requests against the actual Flask app — not
  mocked):
  - Landing page renders correctly with the new instrument tag.
  - Full upload → analyze flow (`obama.jpg`, a real two-eye photo with
    one eye that fails segmentation): stage cycler visible during the
    request; on success, the hero count-up reached the correct real
    value (12) with correct metrics; the left eye's stage viewer showed
    all 9 tabs (Original, Face+Eye Detection, Eye Region, Iris Region,
    Mask, Normalized, Enhanced, Detected Structures, Final Result); the
    right eye (real segmentation failure) correctly showed only 5 tabs
    (stopped before Mask) plus the real error text and the failure
    aside — confirming stages are genuinely per-eye and never faked.
  - Switching stage tabs (e.g. to "Mask") correctly swapped to that
    stage's real image with the correct "Fig. 05 — Mask" caption.
  - Clicking "Enlarge" opened the zoom modal with the image correctly
    scaled up (fixed a bug here — see below); Escape correctly closed it
    and returned focus.
  - Dark theme (After-Hours Lab) checked against the full result view —
    all new elements (hero number, metrics, stage tabs, active-tab
    styling) correctly pick up the theme's CSS variables.
  - Mobile viewport (390×844): layout holds, stage tabs wrap correctly,
    nothing overflows.
  - IRIS MATCH: verified the new heading/subtitle, "Specimen A/B"
    labels, and — using a known-successful test pair
    (`obama.jpg`/`obama_small.jpg`) — the full result view: real
    compatibility score, animated subscore bars settled at their correct
    real widths, per-eye "Specimen A — Left eye" cards, and both
    disclaimer lines. Also exercised the genuine-failure path (a pair
    where the second photo doesn't segment) and confirmed the real
    backend error still surfaces correctly.
  - Camera flow tested with Chromium's fake video device
    (`--use-fake-device-for-media-stream`): alignment guide and "Camera
    live — align eye" status appear on start, both correctly disappear
    and the status updates to "Frame captured" after capture; the full
    capture → analyze round trip was exercised end to end and correctly
    produced an honest "I can't find a face in this image" result (the
    fake device's synthetic pattern isn't a face) with the correctly
    reduced 3-tab stage viewer. This exercises the whole camera code
    path but is not a substitute for a real device/real eye — that
    remains outstanding, as before.
  - Checked the browser console throughout: no JavaScript errors. The
    only console message seen was a `403` for the Google Fonts CDN
    request, which is this sandbox's network allowlist blocking
    `fonts.googleapis.com` — not an application bug. Fonts fall back to
    the declared system-font stack (Georgia/Times New Roman,
    system-ui/Arial, Consolas/monospace) cleanly; on a normally deployed
    host with internet access this will load the real Fraunces/DM
    Sans/IBM Plex Mono as before.

### Problems found and fixed

- **Zoom modal wasn't actually zooming.** The first version capped the
  modal image at `max-width: 100%` with no minimum, so small native
  crops (e.g. the ~150×100px mask stage) displayed at their tiny native
  size instead of enlarging — defeating the point of an "Enlarge"
  button. Fixed by changing the modal image to `width: 100%; height:
  auto; max-height: 72vh; object-fit: contain;`, so it now scales up to
  fill the modal (bounded by the modal's own max width/height) while
  preserving aspect ratio. Verified visually before and after.
- One duplicate event-listener bind was caught and removed while writing
  `buildStageViewer()` (a leftover from an early draft that would have
  double-fired stage-tab clicks) — caught by code review before testing,
  not by a failing test, but worth recording since it's the kind of bug
  that wouldn't have thrown an error, just silently double-run the
  cross-fade.
- No other genuine defects found. No regressions in Modules 1–9.

### Known limitations (carried over, unchanged)

- Manual camera testing on a real device with a real eye is still
  outstanding — this sandbox has no camera hardware. The fake-device
  test above confirms the code path works mechanically, not that a real
  capture will look good on a real webcam.
- Segmentation still fails on some real eyes (see Modules 2/5's honestly
  recorded failure rates, 3/8 and 3/10) — this is a CV-quality limitation
  of the current segmentation approach, not something Module 10 touched
  or was asked to fix.
- The interactive stage viewer's tabs are plain focusable buttons, not a
  full roving-tabindex ARIA tabs pattern (arrow-key navigation between
  tabs). They're keyboard-operable via Tab/Enter, which meets basic
  accessibility, but a full tabs pattern would be a reasonable small
  follow-up if this gets more polish time.

### Current status

**Working.** All 7 existing test suites still pass with their original
recorded baselines. The new interactive experience was verified visually
and functionally end to end (upload, camera with fake device, both
success and honest-failure paths, both themes, mobile layout) with no
console errors beyond an expected sandbox network restriction.

### Next module

All 8 modules from the original plan, plus Module 9's evaluation and
this Module 10 polish pass, are complete. Per the "no build ahead"
principle, nothing further should be implemented without explicit
instruction. If further work is wanted, the two standing items are:
manual camera testing on a real device (unchanged since Module 8), and
optionally a full ARIA-tabs keyboard pattern for the stage viewer (minor,
noted above).


---

## Module 11 (Final Hackathon Polish, Testing, Test Dataset & Release) — COMPLETE, one open item

### What was actually done

**Full project audit** (code, not just documentation):
- Compiled every backend file (`python -m py_compile`) — clean.
- AST-checked every backend module for unused imports — none found.
- Grepped for `TODO`/`FIXME`/`pdb`/leftover `print()` — only the one
  intentional safety-net `print()` in `analyzer.py`'s top-level exception
  handler (documented since Module 8) still present, nothing new.
- Grepped for secrets/API keys/passwords/tokens — none found.
- Confirmed `MAX_CONTENT_LENGTH` is enforced and returns a proper 413.
- Confirmed the camera stream is explicitly stopped via
  `getTracks().forEach(track => track.stop())`.
- Scientific-honesty grep across frontend/backend/README for prohibited
  claims (diagnosis, identification, authentication, personality/health
  prediction) — every hit was either the disclaimer text itself or an
  internal function name like `_diagnose_segmentation_failure` (an error
  classifier, not a user-facing medical claim). No violations found.
- Discovered `DEVELOPMENT_STATE.md`'s header was stale (said "after
  Module 8" while Modules 9 and 10 were already fully written below it).
  Fixed — see the top of this file.

**Regression / determinism / adversarial testing** — full results in
`TEST_REPORT.md`, summarized:
- Re-ran all 7 automated suites: identical to every previously recorded
  baseline (4/6, 3/8, 3/3, `[10,12,12]`, 3/10, PASS, PASS).
- Determinism: called `analyze_iris_file()` three times in a row on the
  same image in the same process and diffed every metric field (not just
  the final count) — identical across all three runs, for two different
  test images.
- Adversarial/degenerate inputs run directly through the pipeline: a 1×1
  pixel image, all-black and all-white 1200×1200 images, random noise at
  1500×1500 and 4000×4000, a heavily blurred copy of a known-good photo, a
  washed-out/low-contrast copy, a downscaled 80×60 copy, and a copy with a
  simulated bright reflection. **No crash, no hang, no fabricated result
  in any case** — every failure returned a specific, honest error string,
  and the largest adversarial image (4000×4000 noise) still completed in
  under a second.
- This testing surfaced one genuine, previously-general limitation made
  concrete: a heavily blurred image doesn't get rejected — it produces a
  *lower* structure count (6–7 vs. the sharp image's 12) while the
  "Analysis Quality" score stays at 95–99%, i.e. quality doesn't currently
  reflect blur. This confirms the Module 8 note ("no working blur
  detector — a naive one scored backwards") with an actual measured
  example rather than leaving it as a general statement. Not a new bug,
  and not fixed here — Module 11's brief is to test and document, not to
  redesign the CV pipeline, and a real fix would need the blur detector
  Module 8 already tried and rejected as unreliable.

**Frontend/UX/accessibility audit**: code-reviewed (not re-screenshotted —
see "What could not be completed" below) `frontend/index.html`,
`style.css`, `script.js`, none of which were modified in this module.
Confirmed `aria-live` regions, `:focus-visible` styles, `prefers-reduced-
motion` gating, and the 560px/400px responsive breakpoints are all still
present and unchanged from the Module 10 session that verified them live
in a real headless browser.

**Repository cleanup**:
- Deleted all `backend/__pycache__/` directories.
- Created `.gitignore` (previously absent) covering Python cache files,
  virtual environments, `.env`, OS/editor cruft, and — importantly — the
  non-redistributable test images described below.

**Live verification**: started `python run.py`, confirmed `Debug mode:
off`, hit `GET /` (200), `POST /api/analyze` with a real photo (200,
results matching CLI output exactly, including the Module 10 `stages`
field), and `POST /api/match` (200).

**README finalized**: added a "Screenshots" section (honestly stating none
are included yet, rather than staging fake ones — no browser is available
in this environment to capture real ones), added two new entries to
"Known limitations" (the blur/quality gap made concrete, and the test
dataset licensing gap), and added a prominent licensing caveat to the
"Testing" section pointing at `TEST_REPORT.md` and the new
`LICENSING_NOTE.md`.

### A genuine, previously-unnoticed problem this audit found

**The existing `data/test_images/` photos cannot legally be published in
a public repository.** Running `identify`/EXIF inspection on them (never
done before — they'd been treated as generic "sample images" since Module
1) revealed that `obama.jpg`, `obama2.jpg`, and `biden.jpg` are official
White House photographs by Pete Souza with an embedded copyright notice
that explicitly forbids manipulation and redistribution without written
permission from the White House Photo Office — which directly conflicts
with what IRISCOPE does to every image it processes. `messi5.jpg` and
`fruits.jpg` are recognizable as OpenCV's own tutorial sample images,
which don't carry a confirmed reuse license either. Full detail and a
per-file table is in the new `data/test_images/LICENSING_NOTE.md`.

**This is not something this session could fully resolve.** Building
Module 11's requested `test_images/good/ challenging/ invalid/` public
dataset needs either the developer's own (or a consenting teammate's) eye
photos, or individually-verified CC0/public-domain images with a checkable
per-file source — and this development sandbox has no general internet
access (only a small allowlist of package/code-hosting domains), so
sourcing and downloading new verified-license images from the wider web
couldn't be completed unattended here. `.gitignore` was updated to keep
the current non-redistributable images (and everything derived from them
in `output*/`) out of version control in the meantime, so the risk is
contained but the deliverable itself — the actual public dataset,
`test_images/IMAGE_SOURCES.md`, and 3 real README screenshots — is still
open.

### Files changed

Created:
- `.gitignore`
- `TEST_REPORT.md`
- `data/test_images/LICENSING_NOTE.md`

Modified:
- `DEVELOPMENT_STATE.md` (this file — stale header fixed, this section added)
- `README.md` (Screenshots section added, Known limitations expanded,
  Testing section given a licensing caveat)

Deleted:
- `backend/__pycache__/` (all `.pyc` files — regenerated automatically,
  now gitignored)

Not modified: every backend `.py` file's actual logic, and all of
`frontend/index.html` / `style.css` / `script.js` — no code defect was
found that justified a change, per Module 11's own instruction to fix
only genuine problems.

### Known limitations (final, as shipped)

Everything listed in Module 8's section still applies. Module 11 adds:
- Analysis Quality score doesn't detect blur (now backed by a measured
  example — see above).
- No publicly redistributable test image dataset yet (see above).
- Fresh live-browser (Playwright) verification could not be re-run this
  session — the browser binary isn't cached in this sandbox and the
  Playwright download CDN isn't in this sandbox's network allowlist.
  Module 10's live browser results still describe the current frontend
  code exactly, since none of it changed in this module.

### Current status

**Working**, code- and audit-complete. The single open deliverable is the
public test dataset (`test_images/good|challenging|invalid/` +
`IMAGE_SOURCES.md`) and 3 real README screenshots — both blocked on the
same thing: this session has no way to source new external images or
capture live screenshots. Once the developer supplies eye photos (their
own, or specifically-checked CC0/public-domain ones) and runs the app
once to grab screenshots, both can be dropped in without touching any
code.

### Next step

None planned — this is the final module per the project's own scope. The
one remaining task (test dataset + screenshots) is an asset-gathering step
for the developer, not a development module.
