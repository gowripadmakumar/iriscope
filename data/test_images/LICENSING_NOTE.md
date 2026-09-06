# Licensing status of images in this folder

**These images must not be published as part of the public IRISCOPE
repository as-is.** This note explains why, so a future session doesn't
accidentally ship them.

## What's actually in here

| File | Actual source (found via EXIF metadata) | License |
|---|---|---|
| `obama.jpg` | Official White House Photo by Pete Souza, Dec. 6, 2012 | Embedded copyright notice explicitly restricts use: "may not be manipulated in any way," "may not otherwise be reproduced, disseminated or broadcast... without written permission of the White House Photo Office," personal-use-only for the photo's subjects. |
| `obama2.jpg` | Official White House Photo by Pete Souza, Sept. 9, 2009 | Same restriction, embedded in EXIF: publication by news organizations or personal printing by the subject only; no manipulation; no commercial/political use. |
| `biden.jpg` | Official White House Photo by Pete Souza, May 10, 2010 | Same restriction as above. |
| `obama_small.jpg` | No EXIF present — provenance unknown | Cannot be verified. Treated as unlicensed. |
| `messi5.jpg` | Filename and JPEG encoder metadata match OpenCV's well-known `samples/data/messi5.jpg` tutorial image | Distributed alongside OpenCV's tutorials for years, but OpenCV's own repository does not attach an explicit reuse license to files under `samples/data/` separate from the code license (Apache 2.0 covers the code, not necessarily the sample images). Not confidently CC0/public-domain. |
| `fruits.jpg` | JPEG encoder metadata ("Handmade Software, Inc. Image Alchemy") matches OpenCV's classic `samples/data/fruits.jpg` | Same ambiguity as `messi5.jpg`. |

**The White House Photo Office's own text explicitly forbids exactly what
IRISCOPE does to an image** (crop it, mask it, overlay detected structures,
run it through a processing pipeline, and display the modified result
publicly) — so `obama.jpg`, `obama2.jpg`, and `biden.jpg` cannot go into a
public GitHub repository's test dataset, regardless of how useful they've
been for development. `messi5.jpg` and `fruits.jpg` are lower-risk (long
precedent of being freely shared as OpenCV tutorial material) but still
don't meet the "confidently verified redistribution license" bar the
project's own module instructions set.

## Why this wasn't caught until Module 11

Nobody had actually opened the EXIF metadata on these files until this
audit. They were treated as generic "sample images" carried over from
early development. Reading `identify`/EXIF output during the Module 11
audit is what surfaced the actual embedded copyright notices.

## What this means in practice

- **Internal development and this repository's own regression testing**
  can keep using these images locally — they're not being redistributed
  by running `pytest`/the test scripts on your own machine, and they
  produce useful, reproducible results (see `TEST_REPORT.md`).
- **They must not be committed to a public GitHub repository**, and if
  this repository already has git history containing them, that history
  needs to be scrubbed before making the repo public — not just deleted
  from the current working tree.
- **A genuinely public, redistributable `good/ challenging/ invalid/`
  test dataset (as specified in Module 11) has not been built yet.** Doing
  it properly requires either:
  1. The developer's own eye photos (or a consenting friend/teammate's),
     which sidesteps licensing entirely and is the simplest fix, or
  2. Specific, individually-verified CC0/public-domain images with a
     checkable source URL for each one (e.g. a specific Wikimedia Commons
     file with a confirmed CC0/PD tag) — this needs to be done by
     visiting and checking each source directly, since license status
     can't be assumed from a filename or a search snippet.

This repository's automated development sandbox for this session has no
general internet access (only a small allowlist of package/code-hosting
domains), so option 2 could not be completed unattended in this session —
see `TEST_REPORT.md` section 8 and the Module 11 handoff notes in
`DEVELOPMENT_STATE.md` for what's needed to finish this.
