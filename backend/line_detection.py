"""
Module 4 — Radial Structure Detection + Counting

Takes Module 3's normalized (polar-unwrapped) iris representation and
produces an approximate, deterministic count of detectable radial texture
structures.

IMPORTANT — scientific honesty (see project knowledge):
There is no standardized anatomical quantity called "the number of lines in
an iris." This module measures a specific, well-defined operational
quantity instead: the number of column-spans in the normalized edge map
that are consistently strong across most of the visible annulus, at a
fixed set of thresholds. That's what "detectable radial structure" means
here, and it's the only thing this module ever claims to count.

Deliberately does NOT touch the web app, results page, or IRIS MATCH --
those are later modules.
"""

import cv2
import numpy as np

import config


def _compute_column_stats(edges, mask_bool):
    """
    For every column (angular degree) of the normalized edge map, computes:

    - valid_row_fraction: how much of that column is actually usable
      (not eyelid/reflection/out-of-frame), per Module 3's mask.
    - coverage_ratio: of the *valid* rows in that column, what fraction are
      "on" (edge value at/above the adaptive threshold). This is the
      length/continuity signal -- a real radial structure runs most of the
      pupil-to-sclera span, so its column has high coverage; a short
      scratch, noise speck, or eyelash fragment usually doesn't.
    - mean_on_value: average edge strength among that column's "on" rows,
      used later as a rough per-structure contrast figure. 0 if no rows
      are "on".

    Returns (valid_row_fraction, coverage_ratio, mean_on_value, edge_threshold),
    each of the first three being a (num_columns,) array, plus the single
    adaptive threshold value used.
    """
    rows, cols = edges.shape

    valid_values = edges[mask_bool]
    if valid_values.size == 0:
        return None

    percentile_value = float(np.percentile(valid_values, config.LINE_EDGE_ON_PERCENTILE))
    edge_threshold = max(percentile_value, config.LINE_EDGE_ON_MIN_VALUE)

    on_mask = (edges >= edge_threshold) & mask_bool

    valid_counts = mask_bool.sum(axis=0).astype(float)          # (cols,)
    on_counts = on_mask.sum(axis=0).astype(float)                # (cols,)

    valid_row_fraction = valid_counts / rows
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage_ratio = np.where(valid_counts > 0, on_counts / np.maximum(valid_counts, 1), 0.0)

    mean_on_value = np.zeros(cols, dtype=float)
    column_intensity = np.zeros(cols, dtype=float)
    edges_f = edges.astype(float)
    for c in range(cols):
        if on_counts[c] > 0:
            mean_on_value[c] = edges_f[on_mask[:, c], c].mean()
        if valid_counts[c] > 0:
            # Mean edge strength over ALL valid rows (not just "on" ones) --
            # a smoother, continuous signal used later for peak-finding
            # within a wide accepted run, where the binary on/off threshold
            # alone can't tell two adjacent structures apart.
            column_intensity[c] = edges_f[mask_bool[:, c], c].mean()

    return valid_row_fraction, coverage_ratio, mean_on_value, column_intensity, edge_threshold


def _find_circular_runs(bool_array):
    """
    Groups a circular boolean array (angle 0..359, wrapping) into runs of
    consecutive True values. Returns a list of (start_index, length) tuples.

    A dedicated circular routine is needed (rather than a plain run-length
    pass) because angle 359 is adjacent to angle 0 -- a real radial
    structure straddling that seam must not be reported as two separate
    structures. This is done directly (scan, then reconcile the two ends)
    rather than via an internal rotation, because a rotation trick applied
    only to *this* step still leaves the later merge step exposed to the
    same wraparound problem -- see _merge_nearby_runs_circular below, which
    handles the wraparound once, in one place, for both raw candidate runs
    and post-merge runs alike.
    """
    n = len(bool_array)
    if not bool_array.any():
        return []
    if bool_array.all():
        return [(0, n)]

    runs = []
    i = 0
    while i < n:
        if bool_array[i]:
            j = i
            while j < n and bool_array[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1

    # If the array both starts and ends True, the first and last scanned
    # runs are actually one run that straddles the index-0 seam.
    if len(runs) >= 2 and bool_array[0] and bool_array[-1]:
        first_start, first_len = runs[0]
        last_start, last_len = runs[-1]
        runs = runs[1:-1]
        runs.append((last_start, last_len + first_len))

    return runs


def _merge_nearby_runs_circular(runs, n, max_gap):
    """
    Merges runs separated by a gap of at most max_gap columns, correctly
    accounting for the circular seam (angle 359 is adjacent to angle 0).

    Sorting by start and merging left-to-right handles every adjacent pair
    except one: the gap between the last run (highest start) and the first
    run (lowest start), which wraps through the 359/0 seam rather than
    sitting between them in sorted order. That pair needs one explicit
    check after the ordinary linear pass.
    """
    if len(runs) <= 1:
        return list(runs)

    runs_sorted = sorted(runs, key=lambda r: r[0])
    merged = [list(runs_sorted[0])]

    for start, length in runs_sorted[1:]:
        prev_start, prev_length = merged[-1]
        prev_end = prev_start + prev_length
        gap = start - prev_end

        if gap <= max_gap:
            new_end = start + length
            merged[-1][1] = new_end - prev_start
        else:
            merged.append([start, length])

    # Wraparound check: does the last merged run connect back to the first
    # through the seam? Skip if merging already collapsed everything into
    # a single run, or if the last run already extends past n (meaning it
    # was itself formed by _find_circular_runs's own seam handling and
    # already accounts for the wrap).
    if len(merged) > 1:
        first_start, first_length = merged[0]
        last_start, last_length = merged[-1]
        if last_start + last_length <= n:
            wrap_gap = (first_start + n) - (last_start + last_length)
            if wrap_gap <= max_gap:
                combined_length = (first_start + first_length + n) - last_start
                merged[0] = [last_start, combined_length]
                merged.pop()

    return [tuple(r) for r in merged]


def _split_run_by_peaks(run, column_intensity, cols, min_separation, smoothing_window):
    """
    Splits a single merged run into one or more sub-runs, one per distinct
    local peak in the run's edge-intensity profile.

    A wide run doesn't necessarily mean one wide structure -- dense,
    closely-packed radial texture can keep every column between two
    adjacent lines above the coverage threshold too, so the run never
    breaks in the first place. This looks for actual local maxima within
    the run (each one a plausible individual line center) and only accepts
    a split when there's more than one, at least min_separation degrees
    apart. A run with a single dominant peak (the normal case) is returned
    unchanged.
    """
    start, length = run
    if length < min_separation:
        return [run]

    indices = [(start + k) % cols for k in range(length)]
    values = np.array([column_intensity[i] for i in indices], dtype=float)

    if smoothing_window > 1 and length >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(values, kernel, mode="same")
    else:
        smoothed = values

    peak_positions = [
        k for k in range(length)
        if smoothed[k] > 0
        and smoothed[k] >= (smoothed[k - 1] if k > 0 else -np.inf)
        and smoothed[k] >= (smoothed[k + 1] if k < length - 1 else -np.inf)
    ]

    if len(peak_positions) <= 1:
        return [run]

    # Greedy non-max suppression: strongest peaks first, drop anything too
    # close to an already-kept, stronger peak.
    peak_positions.sort(key=lambda k: smoothed[k], reverse=True)
    kept = []
    for k in peak_positions:
        if all(abs(k - kk) >= min_separation for kk in kept):
            kept.append(k)

    if len(kept) <= 1:
        return [run]

    kept.sort()
    boundaries = [0] + [(kept[i] + kept[i + 1]) // 2 + 1 for i in range(len(kept) - 1)] + [length]

    sub_runs = []
    for i in range(len(kept)):
        sub_length = boundaries[i + 1] - boundaries[i]
        if sub_length <= 0:
            continue
        sub_start = (start + boundaries[i]) % cols
        sub_runs.append((sub_start, sub_length))

    return sub_runs


def _describe_structure(run, coverage_ratio, mean_on_value, candidate_mask, cols):
    """Builds the reported dict for one final (post-merge) structure."""
    start, length = run
    indices = [(start + k) % cols for k in range(length)]

    end_deg = (start + length - 1) % cols
    center_deg = (start + (length - 1) / 2.0) % cols

    coverages = [coverage_ratio[i] for i in indices]
    on_values = [mean_on_value[i] for i in indices if mean_on_value[i] > 0]
    bridged_count = sum(1 for i in indices if not candidate_mask[i])

    return {
        "start_deg": int(start),
        "end_deg": int(end_deg),
        "angular_width_deg": int(length),
        "center_deg": round(float(center_deg), 1),
        "coverage_ratio": round(float(np.mean(coverages)), 3) if coverages else 0.0,
        "mean_contrast": round(float(np.mean(on_values)), 1) if on_values else 0.0,
        "bridged_columns": int(bridged_count),
    }


def count_radial_structures(normalization_result):
    """
    Main entry point for Module 4.

    Given Module 3's normalize_iris() output, detects and counts
    approximately radial texture structures. Requires
    normalization_result["success"] to be True.

    Returns:
    {
        "success": bool,
        "error": str or None,
        "structure_count": int or None,
        "structures": [
            {
                "start_deg": int, "end_deg": int, "angular_width_deg": int,
                "center_deg": float, "coverage_ratio": float,
                "mean_contrast": float, "bridged_columns": int,
            },
            ...
        ] or None,
        "raw_candidate_count": int or None,   # distinct runs before merging
        "detection_quality": float or None,   # 0-1, image/analysis suitability -- NOT accuracy
        "edge_threshold_used": float or None, # for transparency/debugging
        "column_diagnostics": {                # for the debug visualization only
            "valid_row_fraction": np.ndarray,
            "coverage_ratio": np.ndarray,
            "candidate_mask": np.ndarray,       # per-column, pre-merge
            "evaluable_mask": np.ndarray,
        } or None,
    }
    """
    failure = {
        "success": False,
        "error": None,
        "structure_count": None,
        "structures": None,
        "raw_candidate_count": None,
        "detection_quality": None,
        "edge_threshold_used": None,
        "column_diagnostics": None,
    }

    if not normalization_result.get("success"):
        failure["error"] = "No usable normalized iris to analyze."
        return failure

    edges = normalization_result["normalized_edges"]
    mask = normalization_result["normalized_mask"]
    valid_ratio = normalization_result["valid_ratio"]

    if edges is None or mask is None:
        failure["error"] = "No usable normalized iris to analyze."
        return failure

    mask_bool = mask > config.NORM_MASK_VALID_THRESHOLD
    cols = edges.shape[1]

    stats = _compute_column_stats(edges, mask_bool)
    if stats is None:
        failure["error"] = "No valid iris texture to analyze."
        return failure

    valid_row_fraction, coverage_ratio, mean_on_value, column_intensity, edge_threshold = stats

    evaluable_mask = valid_row_fraction >= config.LINE_MIN_VALID_ROW_FRACTION
    candidate_mask = evaluable_mask & (coverage_ratio >= config.LINE_MIN_ROW_COVERAGE_RATIO)

    raw_runs = _find_circular_runs(candidate_mask)
    merged_runs = _merge_nearby_runs_circular(raw_runs, cols, config.LINE_MERGE_GAP_DEG)

    final_runs = []
    for run in merged_runs:
        final_runs.extend(_split_run_by_peaks(
            run, column_intensity, cols,
            config.LINE_PEAK_MIN_SEPARATION_DEG, config.LINE_PEAK_SMOOTHING_WINDOW,
        ))

    structures = [
        _describe_structure(run, coverage_ratio, mean_on_value, candidate_mask, cols)
        for run in final_runs
    ]
    # Report structures in angular order for a predictable, readable result.
    structures.sort(key=lambda s: s["start_deg"])

    evaluable_fraction = float(evaluable_mask.mean())
    detection_quality = round(float(np.mean([valid_ratio, evaluable_fraction])), 3)

    return {
        "success": True,
        "error": None,
        "structure_count": len(structures),
        "structures": structures,
        "raw_candidate_count": len(raw_runs),
        "detection_quality": detection_quality,
        "edge_threshold_used": round(float(edge_threshold), 1),
        "column_diagnostics": {
            "valid_row_fraction": valid_row_fraction,
            "coverage_ratio": coverage_ratio,
            "candidate_mask": candidate_mask,
            "evaluable_mask": evaluable_mask,
        },
    }


def draw_column_profile(result, bar_height=60):
    """
    Builds a (bar_height, 360, 3) BGR strip visualizing, per angular degree:

    - dark gray  : column too occluded to evaluate (below
                   LINE_MIN_VALID_ROW_FRACTION)
    - mustard    : evaluable but below the coverage threshold (rejected)
    - light teal : below threshold on its own, but bridged into a nearby
                   accepted structure by the merge-gap step
    - teal       : part of a final accepted structure, bar height scaled by
                   that column's own coverage ratio

    This is the "why accepted / why rejected" view: a viewer can see both
    the final decision (color) and the underlying signal strength (bar
    height) for every angle at once, debug/dev tool only.
    """
    if result is None or not result.get("success"):
        return np.zeros((bar_height, 360, 3), dtype=np.uint8)

    diagnostics = result["column_diagnostics"]
    coverage_ratio = diagnostics["coverage_ratio"]
    candidate_mask = diagnostics["candidate_mask"]
    evaluable_mask = diagnostics["evaluable_mask"]

    accepted_cols = np.zeros(360, dtype=bool)
    for structure in result["structures"]:
        start = structure["start_deg"]
        length = structure["angular_width_deg"]
        for k in range(length):
            accepted_cols[(start + k) % 360] = True

    strip = np.full((bar_height, 360, 3), 244, dtype=np.uint8)  # cream background

    for c in range(360):
        if accepted_cols[c]:
            color = config.COLOR_STRUCTURE_ACCEPTED if candidate_mask[c] else config.COLOR_STRUCTURE_BRIDGED
        elif not evaluable_mask[c]:
            color = config.COLOR_STRUCTURE_UNEVALUATED
        else:
            color = config.COLOR_STRUCTURE_REJECTED

        bar_h = max(1, int(coverage_ratio[c] * bar_height))
        strip[bar_height - bar_h:, c] = color

    return strip


def draw_structures_overlay(eye_crop_bgr, segmentation_result, normalization_result, result):
    """
    Draws each accepted structure as a radial tick mark on the actual eye
    crop, mapped from its unwrapped angular position back to image
    coordinates using the same pupil/iris geometry Module 3 used to build
    the normalized representation in the first place. Debug/dev tool only.
    """
    annotated = eye_crop_bgr.copy()

    if result is None or not result.get("success") or not segmentation_result.get("success"):
        return annotated

    cx, cy = segmentation_result["iris_center"]
    pupil_r = segmentation_result["pupil"][2]
    iris_r = segmentation_result["iris_radius"]
    span = iris_r - pupil_r

    r_inner = pupil_r + span * config.NORM_INNER_MARGIN_RATIO
    r_outer = pupil_r + span * config.NORM_OUTER_MARGIN_RATIO

    for structure in result["structures"]:
        angle_rad = np.radians(structure["center_deg"])
        x1 = int(cx + r_inner * np.cos(angle_rad))
        y1 = int(cy + r_inner * np.sin(angle_rad))
        x2 = int(cx + r_outer * np.cos(angle_rad))
        y2 = int(cy + r_outer * np.sin(angle_rad))
        cv2.line(annotated, (x1, y1), (x2, y2), config.COLOR_STRUCTURE_ACCEPTED, 1)

    cv2.circle(annotated, (cx, cy), iris_r, config.COLOR_RESULT_TEXT, 1)

    return annotated
