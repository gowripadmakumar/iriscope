"""
Module 2 — Iris Segmentation

Takes a padded eye-region crop from Module 1 and locates the pupil and iris
boundary within it, then builds a mask of the iris texture that's actually
usable for later analysis (excluding the pupil, eyelids/eyelashes, and
reflections).

Deliberately does NOT normalize/unwrap the iris or analyze its texture yet
-- that's Module 3.
"""

import cv2
import numpy as np

import config


def _preprocess_for_segmentation(eye_crop_bgr):
    """Grayscale + light blur. The blur suppresses pixel-level noise before
    we look for the pupil's dark blob and the iris's radial brightness
    transition -- both are sensitive to salt-and-pepper noise otherwise."""
    gray = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def estimate_pupil(gray, hint_box=None):
    """
    Finds the pupil as the most circular blob among the darkest pixels in
    the crop.

    Uses a percentile-based threshold (darkest N% of pixels) rather than a
    fixed brightness cutoff, so this adapts per-image instead of assuming a
    global "dark enough" value -- important since a light iris and a dark
    iris produce very different overall brightness.

    hint_box, if given, is Module 1's *raw* (unpadded) eye detection box in
    this crop's local coordinates: (x, y, w, h). The padded crop around it
    often includes the eyebrow (Module 1 pads symmetrically to give the iris
    pipeline breathing room), and an eyebrow segment can easily be darker
    and more "circular" after cleanup than the actual pupil. The raw box
    brackets the eye opening itself, so it's used both as a hard search
    region (expanded slightly for margin) and as the reference point for the
    center-distance preference below -- this is what actually fixed pupil
    detection landing on eyebrows during testing (see DEVELOPMENT_STATE.md).

    Returns (center_x, center_y, radius) or None if nothing plausible was found.
    """
    h, w = gray.shape[:2]
    min_side = min(h, w)

    if hint_box is not None:
        hx, hy, hw, hh = hint_box
        hint_cx, hint_cy = hx + hw / 2, hy + hh / 2
        margin_x, margin_y = hw * config.PUPIL_HINT_BOX_MARGIN_RATIO, hh * config.PUPIL_HINT_BOX_MARGIN_RATIO
        search_x1 = max(0, hx - margin_x)
        search_y1 = max(0, hy - margin_y)
        search_x2 = min(w, hx + hw + margin_x)
        search_y2 = min(h, hy + hh + margin_y)
    else:
        # No hint available -- fall back to trusting the whole crop, biased
        # toward its geometric center.
        hint_cx, hint_cy = w / 2, h / 2
        search_x1, search_y1, search_x2, search_y2 = 0, 0, w, h

    dark_value = np.percentile(gray, config.PUPIL_DARK_PERCENTILE)
    _, dark_mask = cv2.threshold(gray, dark_value, 255, cv2.THRESH_BINARY_INV)

    # Opening removes thin dark noise (eyelashes, stray pixels); closing
    # fills small gaps so the pupil forms one solid blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    # Hard-restrict the search to the (margin-expanded) hint region when we
    # have one, so candidates like an eyebrow sitting well above the eye
    # opening are excluded outright rather than merely down-weighted.
    if hint_box is not None:
        region_mask = np.zeros_like(dark_mask)
        region_mask[int(search_y1):int(search_y2), int(search_x1):int(search_x2)] = 255
        dark_mask = cv2.bitwise_and(dark_mask, region_mask)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_r = min_side * config.PUPIL_MIN_RADIUS_RATIO
    max_r = min_side * config.PUPIL_MAX_RADIUS_RATIO

    best_score = None
    best_circle = None

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 10 or perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        if radius < min_r or radius > max_r:
            continue
        if circularity < config.PUPIL_MIN_CIRCULARITY:
            continue

        # Prefer blobs close to the reference point (the hint box center
        # when available, otherwise the crop center) -- this breaks ties in
        # favor of the real pupil over a similarly-circular dark patch
        # elsewhere in the search region.
        center_distance = np.hypot(cx - hint_cx, cy - hint_cy)
        normalized_distance = center_distance / (min_side / 2)
        score = circularity - config.PUPIL_CENTER_DISTANCE_PENALTY * normalized_distance

        if best_score is None or score > best_score:
            best_score = score
            best_circle = (int(round(cx)), int(round(cy)), int(round(radius)))

    return best_circle


def estimate_iris_boundary(gray, pupil, hint_box=None):
    """
    Finds the iris's outer radius by searching, at increasing distance from
    the pupil center, for the radius where average brightness jumps most
    sharply -- the transition from (darker) iris to (brighter) sclera.

    This is a simplified version of the classic Daugman integro-differential
    operator: instead of a full 2D search over centers and radii, it assumes
    the iris is concentric with the already-found pupil and only searches
    over radius. That's a real simplification (real irises are sometimes
    slightly decentered from the pupil), but it's fast, deterministic, and
    accurate enough for this project's purpose -- see DEVELOPMENT_STATE.md.

    hint_box, if given, is Module 1's raw eye box in crop-local coordinates.
    It's used to cap how far the iris boundary is allowed to extend: without
    it, a bright eyebrow or eyelid crease sitting further out than the true
    sclera edge can occasionally out-score the real transition and inflate
    the iris radius well past the actual iris (see DEVELOPMENT_STATE.md).

    Returns an integer radius, or None if no reliable boundary was found.
    """
    cx, cy, pupil_r = pupil
    h, w = gray.shape[:2]
    min_side = min(h, w)

    r_min = pupil_r * config.IRIS_RADIUS_MIN_RATIO_OF_PUPIL
    r_max = min(pupil_r * config.IRIS_RADIUS_MAX_RATIO_OF_PUPIL, min_side / 2)

    # A search range needs at least this many radii for the smoothing
    # window + derivative step to produce anything at all.
    min_usable_span = config.IRIS_BOUNDARY_SMOOTHING_WINDOW + 2

    if hint_box is not None:
        hx, hy, hw, hh = hint_box
        margin_x, margin_y = hw * config.PUPIL_HINT_BOX_MARGIN_RATIO, hh * config.PUPIL_HINT_BOX_MARGIN_RATIO
        ex1, ey1 = hx - margin_x, hy - margin_y
        ex2, ey2 = hx + hw + margin_x, hy + hh + margin_y
        hint_r_max = min(cx - ex1, ex2 - cx, cy - ey1, ey2 - cy)
        # Only apply the tighter cap if it still leaves a workable search
        # range -- otherwise a small/tight eye box would block legitimate
        # detections rather than just preventing eyebrow/cheek overreach.
        if hint_r_max > 0 and (hint_r_max - r_min) >= min_usable_span:
            r_max = min(r_max, hint_r_max)

    if r_max <= r_min:
        return None

    radii = np.arange(int(r_min), int(r_max))
    if len(radii) < config.IRIS_BOUNDARY_SMOOTHING_WINDOW + 2:
        return None

    angles = np.linspace(0, 2 * np.pi, config.IRIS_BOUNDARY_ANGLE_SAMPLES, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    min_valid_points = len(angles) * config.IRIS_BOUNDARY_MIN_VALID_FRACTION

    mean_intensities = []
    valid_radii = []

    for r in radii:
        xs = (cx + r * cos_a).astype(int)
        ys = (cy + r * sin_a).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

        if inside.sum() < min_valid_points:
            continue  # too much of this ring falls outside the crop to trust it

        mean_intensities.append(gray[ys[inside], xs[inside]].mean())
        valid_radii.append(r)

    if len(mean_intensities) < config.IRIS_BOUNDARY_SMOOTHING_WINDOW + 2:
        return None

    intensities = np.array(mean_intensities, dtype=float)
    window = config.IRIS_BOUNDARY_SMOOTHING_WINDOW
    smoothing_kernel = np.ones(window) / window
    smoothed = np.convolve(intensities, smoothing_kernel, mode="valid")
    derivative = np.diff(smoothed)

    if len(derivative) == 0:
        return None

    # The iris-to-sclera transition is a brightness *increase* with radius.
    # Take the FIRST jump strong enough to plausibly be that transition,
    # rather than the single strongest jump anywhere in the search range --
    # a further-out edge (eyelid crease, hairline, eyebrow) is sometimes a
    # sharper brightness change than the true iris boundary, which was
    # inflating the iris radius well past the actual iris during testing
    # (see DEVELOPMENT_STATE.md). Scanning from the pupil outward and
    # stopping at the first strong candidate favors the nearest transition.
    peak_derivative = derivative.max()
    if peak_derivative <= 0:
        return None

    threshold = peak_derivative * config.IRIS_BOUNDARY_PEAK_THRESHOLD_RATIO
    candidates = np.where(derivative >= threshold)[0]
    best_step = int(candidates[0]) if len(candidates) > 0 else int(np.argmax(derivative))

    radius_index = min(best_step + window // 2, len(valid_radii) - 1)

    return int(valid_radii[radius_index])


def _build_reflection_mask(gray, iris_radius):
    """Masks small, very bright blobs (camera/light reflections on the
    cornea). A fixed brightness threshold is appropriate here -- unlike
    pupil/eyelid detection, reflections are near-white regardless of iris
    color. The blob-size cap prevents an overall bright/light iris from
    being masked out entirely."""
    bright = (gray >= config.REFLECTION_BRIGHTNESS_THRESHOLD).astype(np.uint8) * 255

    iris_area = np.pi * (iris_radius ** 2)
    max_blob_area = iris_area * config.REFLECTION_MAX_BLOB_AREA_RATIO

    mask = np.zeros_like(gray, dtype=np.uint8)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 0 < area <= max_blob_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask


def _build_eyelid_mask(gray, pupil, iris_radius):
    """
    Masks eyelid/eyelash occlusion using an adaptive brightness range rather
    than a fixed threshold, since "what iris texture looks like" differs a
    lot between a dark brown iris and a light blue one.

    The approach: sample pixels directly left and right of the pupil (within
    a band around the horizontal), since that region is rarely covered by
    eyelids regardless of gaze direction. That gives an expected brightness
    range for genuine iris texture in *this* image. Anything elsewhere in
    the iris ring that falls well outside that range -- eyelid skin
    (usually much brighter) or eyelash shadow (usually much darker) -- gets
    masked out.
    """
    cx, cy, pupil_r = pupil
    h, w = gray.shape[:2]

    half_angle = np.radians(config.EYELID_SAMPLE_HALF_ANGLE_DEG)
    sample_angles = np.concatenate([
        np.linspace(-half_angle, half_angle, 15),                  # right of pupil
        np.linspace(np.pi - half_angle, np.pi + half_angle, 15),   # left of pupil
    ])
    sample_radii = np.linspace(pupil_r * 1.2, iris_radius * 0.95, 8)

    samples = []
    for r in sample_radii:
        xs = (cx + r * np.cos(sample_angles)).astype(int)
        ys = (cy + r * np.sin(sample_angles)).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        samples.extend(gray[ys[inside], xs[inside]].tolist())

    if len(samples) < 10:
        # Not enough reliable samples (e.g. pupil is right at the crop edge)
        # -- rather than guess, skip eyelid masking for this image.
        return np.zeros_like(gray, dtype=np.uint8)

    samples = np.array(samples, dtype=float)
    mean_val = samples.mean()
    # Floor the std so a nearly-flat sample band doesn't produce an
    # unrealistically tight (and over-aggressive) exclusion range.
    std_val = max(samples.std(), 5.0)
    tolerance = config.EYELID_INTENSITY_TOLERANCE_STD * std_val

    gray_f = gray.astype(float)
    occluded = ((gray_f < mean_val - tolerance) | (gray_f > mean_val + tolerance))
    return (occluded.astype(np.uint8)) * 255


def _build_masks(gray, pupil, iris_radius):
    """Builds every intermediate mask plus the final usable iris region."""
    cx, cy, pupil_r = pupil

    iris_disc = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(iris_disc, (cx, cy), iris_radius, 255, thickness=cv2.FILLED)

    pupil_disc = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(pupil_disc, (cx, cy), pupil_r, 255, thickness=cv2.FILLED)

    iris_ring = cv2.subtract(iris_disc, pupil_disc)  # iris texture area, pupil excluded

    reflection_mask = _build_reflection_mask(gray, iris_radius)
    eyelid_mask = _build_eyelid_mask(gray, pupil, iris_radius)

    invalid = cv2.bitwise_or(reflection_mask, eyelid_mask)
    usable_mask = cv2.bitwise_and(iris_ring, cv2.bitwise_not(invalid))

    return {
        "iris_ring": iris_ring,
        "reflection_mask": reflection_mask,
        "eyelid_mask": eyelid_mask,
        "usable_mask": usable_mask,
    }


def segment_iris(eye_crop_bgr, eye_box_in_crop=None):
    """
    Main entry point for Module 2.

    Given a BGR eye-region crop (as produced by Module 1's get_eye_crop),
    returns a dict describing the segmentation.

    eye_box_in_crop, if given, is Module 1's raw (unpadded) eye detection
    box translated into this crop's local coordinates: (x, y, w, h). It
    substantially improves pupil localization by telling the pupil search
    where the actual eye opening is, as opposed to the padded crop as a
    whole (which often includes some eyebrow) -- see estimate_pupil() for
    why this matters. If omitted, pupil search falls back to using the
    whole crop.

    {
        "success": bool,
        "error": str or None,
        "pupil": (cx, cy, radius) or None,
        "iris_center": (cx, cy) or None,   # currently == pupil center, see note below
        "iris_radius": int or None,
        "masks": {
            "iris_ring": binary mask,       # iris disc minus pupil
            "reflection_mask": binary mask, # detected specular highlights
            "eyelid_mask": binary mask,     # detected eyelid/eyelash occlusion
            "usable_mask": binary mask,     # iris_ring minus both of the above
        } or None,
        "usable_area_ratio": float or None,  # usable_mask area / iris_ring area
    }

    Note: iris_center is currently assumed identical to the pupil center
    (concentric approximation). Real eyes have some pupil/iris decentration;
    this is a deliberate simplification for this stage of the project.
    """
    gray = _preprocess_for_segmentation(eye_crop_bgr)

    pupil = estimate_pupil(gray, hint_box=eye_box_in_crop)
    if pupil is None:
        return {
            "success": False,
            "error": "Could not locate the pupil in this eye region.",
            "pupil": None,
            "iris_center": None,
            "iris_radius": None,
            "masks": None,
            "usable_area_ratio": None,
        }

    iris_radius = estimate_iris_boundary(gray, pupil, hint_box=eye_box_in_crop)
    if iris_radius is None or iris_radius <= pupil[2]:
        return {
            "success": False,
            "error": "Could not locate a reliable iris boundary.",
            "pupil": pupil,
            "iris_center": (pupil[0], pupil[1]),
            "iris_radius": None,
            "masks": None,
            "usable_area_ratio": None,
        }

    masks = _build_masks(gray, pupil, iris_radius)

    iris_area = int(np.count_nonzero(masks["iris_ring"]))
    usable_area = int(np.count_nonzero(masks["usable_mask"]))
    usable_ratio = (usable_area / iris_area) if iris_area > 0 else 0.0

    success = usable_ratio >= config.MIN_USABLE_IRIS_AREA_RATIO

    return {
        "success": success,
        "error": None if success else "The usable iris area is too small for reliable analysis.",
        "pupil": pupil,
        "iris_center": (pupil[0], pupil[1]),
        "iris_radius": iris_radius,
        "masks": masks,
        "usable_area_ratio": round(usable_ratio, 3),
    }


def draw_circles_overlay(eye_crop_bgr, result):
    """Draws the detected pupil and iris boundary circles for visual
    inspection. Debug/dev tool only."""
    annotated = eye_crop_bgr.copy()

    if result["pupil"] is not None:
        cx, cy, r = result["pupil"]
        cv2.circle(annotated, (cx, cy), r, config.COLOR_PUPIL, 1)
        cv2.circle(annotated, (cx, cy), 2, config.COLOR_PUPIL, -1)

    if result["iris_radius"] is not None:
        cx, cy = result["iris_center"]
        cv2.circle(annotated, (cx, cy), result["iris_radius"], config.COLOR_IRIS_BOUNDARY, 1)

    return annotated


def draw_mask_overlay(eye_crop_bgr, mask, color, alpha=0.5):
    """Blends a binary mask over the eye crop in the given color, for debug
    visualization. Debug/dev tool only."""
    solid_color = np.full_like(eye_crop_bgr, color, dtype=np.uint8)
    tinted = cv2.addWeighted(eye_crop_bgr, 1 - alpha, solid_color, alpha, 0)

    mask_bool = mask > 0
    blended = eye_crop_bgr.copy()
    blended[mask_bool] = tinted[mask_bool]
    return blended
