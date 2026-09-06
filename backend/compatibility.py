"""
Module 7 — IRIS MATCH

A deterministic, fictional "iris compatibility" score between two people's
eye photos.

This module adds NO new computer-vision detection logic. It runs the exact
same full pipeline every other IRISCOPE feature uses --
`analyzer.analyze_iris()` (Modules 1-5, unchanged) -- once per photo, picks
each person's best-analyzed eye, and compares the metrics that pipeline
already produces (`structure_count`, `structure_density`,
`detection_quality`, `usable_area_ratio`, `valid_ratio`,
`raw_candidate_count`). The score is a texture/structure comparison, not a
prediction about the people themselves -- see project knowledge's
non-negotiables.
"""

import math

import config
from analyzer import analyze_iris


def _pick_best_eye(analysis):
    """
    Of an analyze_iris() result's eyes, returns the one to represent this
    person in the match -- the successfully-analyzed eye with the highest
    detection_quality (Module 4's own suitability score). Returns None if
    no eye in the photo was successfully analyzed. max() with a list in a
    fixed order is deterministic: ties resolve to whichever eye appears
    first (left before right, per Module 1's ordering).
    """
    successful_eyes = [eye for eye in analysis["eyes"] if eye["success"]]
    if not successful_eyes:
        return None
    return max(successful_eyes, key=lambda eye: eye["metrics"]["detection_quality"])


def _candidate_density(metrics):
    """
    Raw candidate columns per 100 degrees of usable iris arc -- the same
    style of "per usable arc" normalization Module 5 already uses for
    structure_density, just applied to the pre-filtering candidate count
    instead of the final accepted count. This reflects how much raw edge
    texture the iris had before Module 4's length/continuity filtering,
    independent of how large the photo's usable region happened to be.
    """
    valid_degrees = max(metrics["valid_ratio"] * config.NORM_ANGLE_SAMPLES, 1e-6)
    return metrics["raw_candidate_count"] / valid_degrees * 100


def _extract_features(metrics):
    """Pulls the six comparable numbers out of one eye's existing metrics dict."""
    return {
        "count": metrics["structure_count"],
        "density": metrics["structure_density"],
        "quality_pct": metrics["detection_quality"] * 100,
        "usable_pct": metrics["usable_area_ratio"] * 100,
        "valid_pct": metrics["valid_ratio"] * 100,
        "candidate_density": _candidate_density(metrics),
    }


def _similarity(a, b, scale):
    """100 * exp(-|a-b|/scale), rounded to an int percentage. See config.py."""
    if scale <= 0:
        return 100
    return round(100 * math.exp(-abs(a - b) / scale))


def _score_pair(features_a, features_b):
    """
    Combines six pairwise similarities into the three named sub-scores plus
    an overall compatibility score. Grouping:

    - TEXTURE SIMILARITY: how comparably clean/usable each iris capture
      was (detection quality, valid texture ratio).
    - RADIAL HARMONY: how similar the actual radial structure content was
      (accepted structure count, raw candidate density).
    - DENSITY HARMONY: how similarly dense/well-covered each iris was
      (structures per usable arc, usable area fraction).

    Every sub-score is a plain average of two of the six similarities
    above -- no hidden weighting, no randomness.
    """
    count_sim = _similarity(features_a["count"], features_b["count"], config.MATCH_COUNT_SCALE)
    density_sim = _similarity(features_a["density"], features_b["density"], config.MATCH_DENSITY_SCALE)
    quality_sim = _similarity(features_a["quality_pct"], features_b["quality_pct"], config.MATCH_QUALITY_SCALE)
    usable_sim = _similarity(features_a["usable_pct"], features_b["usable_pct"], config.MATCH_USABLE_SCALE)
    valid_sim = _similarity(features_a["valid_pct"], features_b["valid_pct"], config.MATCH_VALID_SCALE)
    candidate_sim = _similarity(
        features_a["candidate_density"], features_b["candidate_density"], config.MATCH_CANDIDATE_DENSITY_SCALE
    )

    texture_similarity = round((quality_sim + valid_sim) / 2)
    radial_harmony = round((count_sim + candidate_sim) / 2)
    density_harmony = round((density_sim + usable_sim) / 2)
    compatibility = round((texture_similarity + radial_harmony + density_harmony) / 3)

    return {
        "compatibility": compatibility,
        "texture_similarity": texture_similarity,
        "radial_harmony": radial_harmony,
        "density_harmony": density_harmony,
    }


def _verdict(score):
    for low, high, text in config.MATCH_VERDICTS:
        if low <= score <= high:
            return text
    return config.MATCH_VERDICTS[-1][2]  # unreachable given the band coverage, kept as a safe fallback


def match_irises(image_a_bgr, image_b_bgr):
    """
    Main entry point for Module 7.

    Runs analyze_iris() on both photos, picks each person's best eye, and
    returns a deterministic fictional compatibility result:

    {
        "success": bool,
        "error": str or None,        # set only if one photo had no usable eye
        "scores": {"compatibility", "texture_similarity",
                   "radial_harmony", "density_harmony"} or None,
        "verdict": str or None,
        "eye_a": analyze_eye() result for person A's chosen eye, or None,
        "eye_b": analyze_eye() result for person B's chosen eye, or None,
    }

    Same two images in, same result out -- no randomness anywhere in this
    module or in the pipeline it calls.
    """
    analysis_a = analyze_iris(image_a_bgr)
    analysis_b = analyze_iris(image_b_bgr)

    eye_a = _pick_best_eye(analysis_a)
    eye_b = _pick_best_eye(analysis_b)

    result = {"success": False, "error": None, "scores": None, "verdict": None, "eye_a": None, "eye_b": None}

    if eye_a is None or eye_b is None:
        missing_label = "first" if eye_a is None else "second"
        reason = (analysis_a["error"] if eye_a is None else analysis_b["error"]) or (
            "I couldn't get a usable iris reading from that photo."
        )
        result["error"] = f"I couldn't read the {missing_label} photo well enough to compare it. {reason}"
        return result

    features_a = _extract_features(eye_a["metrics"])
    features_b = _extract_features(eye_b["metrics"])
    scores = _score_pair(features_a, features_b)

    result.update({
        "success": True,
        "scores": scores,
        "verdict": _verdict(scores["compatibility"]),
        "eye_a": eye_a,
        "eye_b": eye_b,
    })
    return result
