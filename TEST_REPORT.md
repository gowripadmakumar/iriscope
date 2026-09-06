# IRISCOPE — Test Report (Module 11)

This report records the results of the final pre-submission testing pass.
Every number in this file was produced by actually running the code in this
repository — nothing here is invented or estimated by hand.

A note on scope: the images referenced below live in `data/test_images/`,
which is IRISCOPE's internal development test set (used since Module 1).
**These images are not bundled for public redistribution** — see
`data/test_images/LICENSING_NOTE.md` for why, and see "Public test dataset"
at the end of this report for the current status of a redistributable set.

---

## 1. Regression test suite (automated, `tests/`)

Run via `python tests/test_module{1..7}.py`. All seven suites reproduce
their documented baselines exactly, with no code changes required:

| Suite | Result | Baseline |
|---|---|---|
| test_module1.py (face/eye detection) | 4/6 images | 4/6 |
| test_module2.py (segmentation) | 3/8 eyes | 3/8 |
| test_module3.py (normalization) | 3/3 eyes | 3/3 |
| test_module4.py (line counting) | counts `[10, 12, 12]` | `[10, 12, 12]` |
| test_module5.py (full pipeline) | 3/10 eyes/images | 3/10 |
| test_module6.py (web app) | PASS | PASS |
| test_module7.py (IRIS MATCH) | PASS | PASS |

## 2. Per-image results (manual verification)

| Test Image | Category | Expected Behavior | Actual Behavior | Result |
|---|---|---|---|---|
| obama.jpg | good (brown eyes, portrait) | left eye analyzes, right eye may fail | Left: 12 structures, quality 97%. Right: pupil not located. | PASS |
| obama_small.jpg | good (brown eyes, low-res 320×240) | left eye analyzes despite low resolution | Left: 12 structures, quality 89%. Right: iris/sclera edge not found. | PASS |
| biden.jpg | good (brown eyes, group photo, off-angle face) | one eye should analyze | Left: pupil not located. Right: 10 structures, quality 99%. | PASS |
| obama2.jpg | challenging (motion/podium angle, smaller face) | likely full failure — genuine hard case | Both eyes: pupil not located | PASS (honest failure) |
| messi5.jpg | invalid (face angle too extreme / eyes not both usable) | no usable eye detected | "I can't find an eye in this image." | PASS (honest failure) |
| fruits.jpg | invalid (no face — bowl of fruit) | no face detected | "I can't find a face in this image." | PASS (honest failure) |

Every "failure" above is a genuine result of the real segmentation and
normalization checks — none are hardcoded by filename. `fruits.jpg`
containing no face at all, and `messi5.jpg`'s eyes not passing the usable-eye
check, are two independently useful **demo failure cases** per Module 11's
requirement for a reliable, non-fabricated rejection example.

## 3. Determinism test

Ran `analyze_iris_file()` **three times in a row** on the same image, same
process, comparing every metric (not just the final count):

| Image | 3 runs identical? |
|---|---|
| obama.jpg | Yes |
| obama_small.jpg | Yes |

No randomness is used anywhere in the counting or scoring path — confirmed
by direct repeated execution, not just by reading the code.

## 4. Adversarial / degenerate input testing

Beyond the automated suite's invalid-input tests (wrong file type, corrupt
bytes, missing field, oversized file — all still passing), the following
synthetic images were run directly through `analyze_iris_file()`:

| Input | Result | Time | Notes |
|---|---|---|---|
| 1×1 pixel image | Honest failure, no crash | 1 ms | "I can't find a face in this image." |
| All-black 1200×1200 | Honest failure, no crash | 30 ms | — |
| All-white 1200×1200 | Honest failure, no crash | 35 ms | — |
| Random noise, 1500×1500 | Honest failure, no crash | 182 ms | — |
| Random noise, 4000×4000 | Honest failure, no crash | 770 ms | No hang even at a large resolution |
| Heavily blurred version of obama_small.jpg | Succeeded, but with a **lower** structure count (7, 6 vs. the sharp image's 12, 12) | — | See "known limitation" below — this is disclosed, not hidden |
| Washed-out / low-contrast version | Honest failure on both eyes | — | Pupil/iris boundary correctly rejected as unreliable |
| Downscaled to 80×60 | Honest failure ("can't find an eye") | — | — |
| Simulated bright reflection on iris | One eye still succeeds (10 structures), other fails on iris/sclera edge | — | Reasonable — reflection only affected one eye's boundary in this synthetic test |

**No image, however degenerate, caused a crash, a hang, or a fabricated
result.** Every failure path returns a specific, honest error string.

## 5. Known limitation surfaced by this testing (not a new bug)

Heavy blur produces a *lower* structure count rather than an outright
rejection, and the reported "Analysis Quality" score does not drop to
reflect the blur (it stayed at 95–99%). This is the same known gap
documented since Module 8: a naive sharpness/blur detector was tried
earlier in development and scored the project's own test images
*backwards*, so it was deliberately left out rather than shipped as a
misleading signal. This testing pass confirms the gap is real and still
present, not a regression — it's called out here and in the README rather
than quietly ignored.

## 6. Frontend / UX / accessibility audit

Verified by code review of `frontend/index.html`, `style.css`, `script.js`
(unchanged since the live Playwright verification performed during Module
10 — see `DEVELOPMENT_STATE.md`'s Module 10 section for that session's full
browser-based results at 320px/375px/1280px, both themes, and with a fake
camera device):

- `aria-live="polite"` present on all status, error, and result regions
- `:focus-visible` outlines defined for buttons, tabs, and the theme toggle
- All animations gated behind `prefers-reduced-motion`, with an instant
  correct end-state fallback
- Responsive breakpoints at 560px and 400px; layout previously confirmed
  down to 320px via Playwright viewport emulation in Module 10
- Camera stream is explicitly stopped via `getTracks().forEach(track =>
  track.stop())` when capture ends or the widget is torn down
- No secrets, API keys, or credentials anywhere in the codebase (checked)

**This session could not re-run a fresh Playwright browser session** — this
sandbox has no cached Chromium binary and the Playwright browser-download
CDN is outside this sandbox's network allowlist, so a new download wasn't
possible. Since none of the frontend files were modified in this module,
Module 10's live browser results still describe the current code exactly.

## 7. Live server verification

Started `python run.py`, confirmed `Debug mode: off` (the Module 8 security
fix), then exercised the real endpoints:

- `GET /` → 200
- `POST /api/analyze` (obama.jpg) → 200, left eye 12 structures / 9 stage
  images, right eye honest failure / 5 stage images — matches CLI output
  exactly
- `POST /api/match` (obama.jpg vs obama_small.jpg) → 200, correct JSON shape

## 8. Public test dataset

**Not yet included in this repository.** See `data/test_images/LICENSING_NOTE.md`
for the reason and the recommended path forward.
