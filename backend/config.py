"""
Centralized configuration for the IRISCOPE computer-vision pipeline.

Keeping every tunable number in one place makes the detector easy to
retune during testing without hunting through multiple files.
"""

import cv2

# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

# Very large photos (e.g. modern phone cameras) are downscaled before
# processing. This keeps memory/CPU usage low on modest hardware and speeds
# up detection without hurting accuracy (faces don't need 4000px of detail).
MAX_IMAGE_DIMENSION = 1024

# ---------------------------------------------------------------------------
# Face detection (Haar cascade)
# ---------------------------------------------------------------------------

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5

# A detected face must be at least this fraction of the image's shorter side.
# Filters out tiny, unreliable false-positive detections in the background.
FACE_MIN_SIZE_RATIO = 0.12

# ---------------------------------------------------------------------------
# Eye detection (Haar cascade)
# ---------------------------------------------------------------------------

EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

EYE_SCALE_FACTOR = 1.05
EYE_MIN_NEIGHBORS = 10  # higher than the OpenCV default (3) to cut false positives

# Eyes only appear in the upper portion of a face bounding box. Restricting
# the search region avoids false positives from nostrils, mouth corners, etc.
EYE_SEARCH_REGION_HEIGHT_RATIO = 0.6

# A detected eye must be at least this fraction of the face width.
EYE_MIN_SIZE_RATIO = 0.12

# When cropping an eye out for later iris analysis, pad the raw detection box
# by this fraction on each side so the iris/eyelid area isn't cut off tight.
EYE_CROP_PADDING_RATIO = 0.35

# If two eye detections overlap/are closer together (as a fraction of face
# width) than this, they're treated as duplicates of the same eye.
EYE_DUPLICATE_DISTANCE_RATIO = 0.15

# ---------------------------------------------------------------------------
# Debug visualization colors (BGR, matching the IRISCOPE palette)
# ---------------------------------------------------------------------------

COLOR_FACE_BOX = (81, 111, 231)    # coral   (#E76F51 in BGR)
COLOR_EYE_BOX = (143, 157, 42)     # teal    (#2A9D8F in BGR)
COLOR_TEXT = (47, 52, 61)          # ink brown (#3D342F in BGR)

# ---------------------------------------------------------------------------
# Iris segmentation (Module 2)
# ---------------------------------------------------------------------------

# --- Pupil estimation ---
# The pupil is found as the most circular blob among the darkest pixels in
# the eye crop. Using a percentile (rather than a fixed brightness value)
# means this adapts per-image instead of assuming a global "dark enough"
# threshold, which matters because a light iris on a dark pupil looks very
# different from a dark iris on a dark pupil.
PUPIL_DARK_PERCENTILE = 5

# Plausible pupil radius, as a fraction of the eye crop's shorter side.
# Filters out both tiny noise specks and implausibly large dark blobs
# (e.g. dark eyeshadow, hair, shadow across the whole crop).
PUPIL_MIN_RADIUS_RATIO = 0.03
PUPIL_MAX_RADIUS_RATIO = 0.25

# Circularity = 4*pi*area / perimeter^2 (1.0 = perfect circle). Eyelashes
# and shadows tend to be elongated/irregular, so this filters them out even
# if they're dark and roughly the right size.
PUPIL_MIN_CIRCULARITY = 0.55

# When multiple dark blobs pass the filters above, prefer the one closest to
# a reference point -- Module 1's raw (unpadded) eye box center when
# available, otherwise the crop's center. This is a soft penalty subtracted
# from circularity, not a hard cutoff.
PUPIL_CENTER_DISTANCE_PENALTY = 0.6

# When Module 1's raw eye box is available, the pupil search is hard-
# restricted to that box expanded by this ratio on each side. Excludes
# clearly-outside candidates (most commonly the eyebrow, which sits in the
# padding Module 1 adds around the raw box) rather than just down-weighting
# them.
PUPIL_HINT_BOX_MARGIN_RATIO = 0.25

# --- Iris boundary estimation ---
# The iris boundary is searched for as a ring around the pupil, using the
# same center (concentric assumption -- see DEVELOPMENT_STATE.md for the
# known limitation this introduces). Search range is defined relative to
# the already-found pupil radius rather than the crop size, since pupil
# size is a much more reliable size reference than the padded crop.
IRIS_RADIUS_MIN_RATIO_OF_PUPIL = 1.8
IRIS_RADIUS_MAX_RATIO_OF_PUPIL = 4.0

# Of all radii where brightness increases sharply moving outward, only the
# FIRST one reaching this fraction of the single strongest jump is used.
# Prevents a stronger but more distant edge (eyelid crease, eyebrow,
# hairline) from being mistaken for the iris boundary.
IRIS_BOUNDARY_PEAK_THRESHOLD_RATIO = 0.6

# Number of angles sampled around each candidate radius when measuring the
# average brightness of that ring (a simplified, lightweight version of the
# classic Daugman integro-differential iris localization operator).
IRIS_BOUNDARY_ANGLE_SAMPLES = 72

# A candidate radius is only considered if at least this fraction of its
# sampled angle points actually land inside the crop (near the crop edges,
# large radii start falling outside the image).
IRIS_BOUNDARY_MIN_VALID_FRACTION = 0.5

# Smooths the radius-vs-brightness profile before taking its derivative, to
# avoid picking a false boundary caused by a single noisy radius.
IRIS_BOUNDARY_SMOOTHING_WINDOW = 5

# --- Reflection / specular highlight masking ---
# Camera/light reflections on the cornea are near-white regardless of iris
# color, so a fixed brightness threshold works here (unlike pupil/eyelid
# detection, which need to adapt per-image).
REFLECTION_BRIGHTNESS_THRESHOLD = 225

# Only small bright blobs are treated as reflections. This cap (relative to
# the iris ring's area) prevents a very light/washed-out iris from being
# mistaken entirely for one giant "reflection".
REFLECTION_MAX_BLOB_AREA_RATIO = 0.15

# --- Eyelid / eyelash occlusion masking ---
# "Known good" iris texture is sampled directly left and right of the pupil
# (within this angle band either side of horizontal), since that region is
# the least likely to be covered by eyelids/eyelashes regardless of gaze or
# iris color. Anything elsewhere in the iris ring that falls far outside
# that sampled brightness range is treated as occluded.
EYELID_SAMPLE_HALF_ANGLE_DEG = 30
EYELID_INTENSITY_TOLERANCE_STD = 2.2

# --- Overall usability ---
# Fraction of the theoretical iris ring (iris disc minus pupil) that must
# survive masking for the segmentation to be considered usable by later
# modules. Below this, there's simply not enough clean texture left.
MIN_USABLE_IRIS_AREA_RATIO = 0.15

# --- Debug visualization colors (BGR, matching the IRISCOPE palette) ---
COLOR_PUPIL = (81, 111, 231)          # coral   (#E76F51) - major detected result
COLOR_IRIS_BOUNDARY = (143, 157, 42)  # teal    (#2A9D8F) - detected structure
COLOR_REFLECTION_MASK = (0, 0, 255)   # pure red - debug-only, high visibility
COLOR_EYELID_MASK = (74, 155, 233)    # mustard (#E9C46A) - annotation/exclusion
COLOR_USABLE_REGION = (143, 157, 42)  # teal    (#2A9D8F) - final usable region

# ---------------------------------------------------------------------------
# Iris normalization (Module 3)
# ---------------------------------------------------------------------------

# Size of the polar-unwrapped ("rectangular") iris representation.
# Columns = angle around the iris, rows = radial distance from pupil to
# sclera. One column per degree gives enough angular resolution to
# eventually distinguish individual radial structures (Module 4), without
# being wastefully large.
NORM_ANGLE_SAMPLES = 360
NORM_RADIAL_SAMPLES = 64

# The annulus actually sampled is slightly INSIDE the pupil/iris boundaries
# found by Module 2, not the full pupil-to-iris span. Expressed as a
# fraction of that span (0 = pupil edge, 1 = iris edge):
#   - Right at the pupil edge, segmentation is noisiest (partial-volume
#     blur between pupil and iris).
#   - Right at the iris edge, the same is true against the sclera, and
#     this is also where eyelash fringe most often survives Module 2's
#     eyelid mask.
# Sampling a slightly narrower band avoids both fringes.
NORM_INNER_MARGIN_RATIO = 0.10
NORM_OUTER_MARGIN_RATIO = 0.90

# Interpolated mask values (from unwrapping a binary 0/255 mask) are
# re-thresholded back to binary at this cutoff, since "half masked" isn't a
# meaningful state for downstream structure detection.
NORM_MASK_VALID_THRESHOLD = 127

# Fraction of the normalized representation that must remain valid (i.e.
# not eyelid/reflection) for the normalization to be considered usable.
# Mirrors MIN_USABLE_IRIS_AREA_RATIO's role but is checked again here since
# the annulus sampled is narrower than Module 2's full iris ring.
NORM_MIN_VALID_RATIO = 0.15

# --- Texture enhancement ---
# Bilateral filtering smooths noise while preserving edges -- important
# here because the *edges themselves* are what Module 4 will eventually
# look for, so a filter that blurs them away (like a plain Gaussian) would
# work against the next module's goal.
NORM_BILATERAL_DIAMETER = 5
NORM_BILATERAL_SIGMA_COLOR = 50
NORM_BILATERAL_SIGMA_SPACE = 50

# CLAHE (contrast-limited adaptive histogram equalization) boosts local
# contrast so faint texture becomes visible on both dark and light irises,
# without the global-contrast overreaction of plain histogram equalization.
NORM_CLAHE_CLIP_LIMIT = 2.0
NORM_CLAHE_TILE_GRID_SIZE = (8, 8)

# ---------------------------------------------------------------------------
# Radial structure detection + counting (Module 4)
# ---------------------------------------------------------------------------
#
# The normalized edge map (Module 3) has one column per degree of iris
# angle, and a genuine radial structure shows up as a roughly vertical line
# spanning most of that column's rows. Module 4 works column-by-column
# rather than a general-purpose line detector (Hough etc.) because that
# geometric property already does most of the false-positive rejection for
# free: circular/tangential edges (eyelid creases, specular rings) show up
# as *horizontal* bands, not vertical ones, so they don't pass the
# per-column coverage test below regardless of how strong they are.

# What fraction of the strongest edge values (within the valid/unmasked
# region) counts as "on" for a given pixel. Percentile-based rather than a
# fixed brightness value so the same logic works on both faint, low-
# contrast texture (small pupils, light irises) and strong, high-contrast
# texture, without retuning per iris color.
LINE_EDGE_ON_PERCENTILE = 55

# Floor for the "on" threshold, in the 0-255 edge-magnitude scale used by
# Module 3's normalized_edges. Without this, an eye with almost no real
# texture (a near-flat, low-contrast crop) could still have its top 30% of
# near-zero values treated as "on" purely by percentile math, manufacturing
# structures out of noise.
LINE_EDGE_ON_MIN_VALUE = 5

# A column (one angular degree) is only judged at all if at least this
# fraction of its rows are valid (not eyelid/reflection/out-of-frame, per
# Module 3's normalized_mask). A column that's mostly masked-out doesn't
# have enough real information to call a candidate or reject it, so it's
# excluded from analysis rather than guessed at either way.
LINE_MIN_VALID_ROW_FRACTION = 0.4

# Of a column's *valid* rows, this fraction must be "on" (see above) for
# that column to be treated as part of a radial structure. This is the
# length/continuity filter: a genuine radial structure runs most of the way
# from the inner to the outer annulus edge, while noise, short scratches,
# and eyelash fragments that dip into the ring typically don't.
LINE_MIN_ROW_COVERAGE_RATIO = 0.30

# Two candidate columns separated by a gap of at most this many degrees are
# merged into a single structure, rather than counted as two. This absorbs
# the case where a real structure's column briefly drops below the coverage
# threshold for a degree or two (blur, a stray reflection pixel, JPEG
# noise) and would otherwise be split into duplicate detections.
LINE_MERGE_GAP_DEG = 1

# After merging, a wide accepted run can genuinely contain more than one
# radial structure -- dense, closely-packed texture doesn't dip below the
# coverage threshold *between* adjacent lines, so the coverage test alone
# can't separate them. Within each merged run, local peaks in per-column
# edge intensity are used to find individual line centers. Two peaks closer
# together than this many degrees are treated as one structure (the weaker
# peak is dropped) rather than two, since at this unwrap resolution that's
# more likely to be one uneven line than two genuinely separate ones.
LINE_PEAK_MIN_SEPARATION_DEG = 2

# Light smoothing (simple moving average, in columns/degrees) applied to
# the per-column intensity profile before peak-picking, so single-column
# noise spikes aren't mistaken for a separate structure.
LINE_PEAK_SMOOTHING_WINDOW = 1

# Colors for the Module 4 debug visualization (BGR, IRISCOPE palette).
COLOR_STRUCTURE_ACCEPTED = (143, 157, 42)   # teal    (#2A9D8F) - accepted structure
COLOR_STRUCTURE_BRIDGED = (185, 208, 148)   # light teal - gap-filled column inside an accepted structure
COLOR_STRUCTURE_REJECTED = (74, 155, 233)   # mustard (#E9C46A) - evaluated but below threshold
COLOR_STRUCTURE_UNEVALUATED = (60, 60, 60)  # dark gray - too masked/occluded to judge
COLOR_RESULT_TEXT = (81, 111, 231)          # coral   (#E76F51) - final count/result

# ---------------------------------------------------------------------------
# Results + visualization (Module 5)
# ---------------------------------------------------------------------------

# When Module 2's segmentation fails specifically on "usable area too small"
# (as opposed to "no pupil found" or "no iris boundary found", which already
# have their own direct explanations), these ratios -- invalid mask pixels
# divided by total iris-ring pixels -- decide which occlusion cause to
# surface to the user. Read directly from Module 2's own mask arrays, not a
# separate guess. NOTE: no test image in this project currently triggers
# this specific failure path (see DEVELOPMENT_STATE.md) -- the logic is
# sound given what the masks represent, but has not been visually verified
# against a real "usable area too small" case yet.
RESULT_REFLECTION_DOMINANT_RATIO = 0.25
RESULT_EYELID_DOMINANT_RATIO = 0.25

# True IRISCOPE cream background (#FFF4DF), for card/panel backgrounds --
# distinct from the plain gray (244,244,244) placeholder used in earlier
# modules' throwaway test-script panels.
COLOR_BACKGROUND_CREAM = (223, 244, 255)

# Multi-stage debug/demo panel layout.
RESULT_CROP_STAGE_HEIGHT = 180   # target height for crop-sized stage images
RESULT_STRIP_STAGE_HEIGHT = 90   # target height for the wide 64-row unwrap strips
RESULT_LABEL_BAR_HEIGHT = 22

# Final result card layout.
RESULT_CARD_EYE_HEIGHT = 260     # height of the detected-structures overlay on the card
RESULT_CARD_TEXT_PANEL_MIN_WIDTH = 260

# ---------------------------------------------------------------------------
# Web application (Module 6)
# ---------------------------------------------------------------------------

# The API only accepts these extensions as a quick, friendly first check.
# cv2.imdecode() (in utils.decode_image_bytes) is the real, authoritative
# validity check -- this just avoids wasting a decode attempt on an
# obviously-wrong file and gives a clearer error sooner.
ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Generous enough for an uncompressed modern phone photo, small enough to
# reject garbage uploads before Flask even buffers the whole body.
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024

# JPEG quality used only when re-encoding Module 5's result panel images to
# send back to the browser as base64. This happens after analysis, so it
# has no effect on detection accuracy -- it only affects how sharp the
# returned debug/result imagery looks.
API_RESULT_JPEG_QUALITY = 85

# ---------------------------------------------------------------------------
# IRIS MATCH scoring (Module 7)
# ---------------------------------------------------------------------------
#
# IRIS MATCH compares the SAME six metrics analyze_eye() (Module 5) already
# computes for each person's best-analyzed eye -- no new detection logic is
# introduced here, only comparison. Each pairwise similarity score is:
#
#     100 * exp(-|a - b| / scale)
#
# so two identical values score 100, and similarity decays smoothly (not a
# hard cliff) as the gap between the two people's numbers grows. "scale" is
# the gap, in that metric's own units, at which similarity has decayed to
# ~37%. These are judgment calls (there's no ground truth for "how much
# structure-count difference should feel noticeable"), picked by looking at
# the actual spread of values Module 4/5 produce across the test set (counts
# in the 10-12 range, density around 3-4/100deg) so real photos don't all
# collapse to either 0 or 100.
MATCH_COUNT_SCALE = 12               # structure_count, in structures
MATCH_DENSITY_SCALE = 4.0            # structure_density, per 100 deg of usable arc
MATCH_QUALITY_SCALE = 25.0           # detection_quality, in percentage points
MATCH_USABLE_SCALE = 30.0            # usable_area_ratio, in percentage points
MATCH_VALID_SCALE = 30.0             # valid_ratio, in percentage points
MATCH_CANDIDATE_DENSITY_SCALE = 5.0  # raw_candidate_count per 100 deg of usable arc

# Fictional verdict bands for the overall compatibility score, per project
# knowledge. Entertainment only -- see the non-negotiables list: this never
# claims to predict personality, relationships, or anything real about
# either person.
MATCH_VERDICTS = [
    (90, 100, "Your irises have chemistry."),
    (75, 89, "Suspiciously compatible."),
    (50, 74, "Your eyes can coexist peacefully."),
    (25, 49, "Your irises have unresolved issues."),
    (0, 24, "Please discuss this with your irises."),
]
