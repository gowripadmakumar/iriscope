"""
Module 5 — Results + Visualization

Runs the existing Modules 1-4 pipeline end to end on one image and turns
the raw per-module outputs into a single, coherent analysis result: every
stage image that was actually produced, the final metrics, and an honest,
IRISCOPE-toned explanation whenever something didn't work.

This module does NOT add any new computer-vision detection logic. Every
number and image here comes directly from Modules 1-4 -- this module's job
is only to orchestrate them, translate their technical errors into
user-facing messages, and assemble presentation-quality visualizations.

Deliberately does NOT touch the web app, camera/upload UI, or IRIS MATCH --
those are later modules.
"""

import cv2
import numpy as np

import config
from utils import load_image, resize_if_large
from face_eye_detection import detect_eyes, get_eye_crop, draw_debug_overlay
from iris_segmentation import segment_iris, draw_circles_overlay
from iris_normalization import normalize_iris
from line_detection import count_radial_structures


# ---------------------------------------------------------------------------
# Friendly error translation
#
# Each function below only rewords an error Modules 1-4 already produced --
# it never invents a cause the pipeline didn't actually detect. The one
# exception is the reflection-vs-eyelid split below, which is a direct
# read of Module 2's own mask data, not a guess.
# ---------------------------------------------------------------------------

def _diagnose_segmentation_failure(seg_result):
    """Translates Module 2's technical error into an IRISCOPE-toned message."""
    error = seg_result.get("error")

    if error == "Could not locate the pupil in this eye region.":
        return ("Your eye is here. Your camera is being difficult -- "
                "I can't pinpoint your pupil in this photo.")

    if error == "Could not locate a reliable iris boundary.":
        return ("I can see your pupil, but not a clear edge between your iris "
                "and the white of your eye. Try a sharper or closer photo.")

    if error == "The usable iris area is too small for reliable analysis." and seg_result.get("masks"):
        masks = seg_result["masks"]
        iris_ring_area = np.count_nonzero(masks["iris_ring"])
        if iris_ring_area > 0:
            reflection_ratio = np.count_nonzero(masks["reflection_mask"]) / iris_ring_area
            eyelid_ratio = np.count_nonzero(masks["eyelid_mask"]) / iris_ring_area
            if (reflection_ratio >= config.RESULT_REFLECTION_DOMINANT_RATIO
                    and reflection_ratio >= eyelid_ratio):
                return "Too much reflection. Try moving away from the light."
            if eyelid_ratio >= config.RESULT_EYELID_DOMINANT_RATIO:
                return ("Your eyelid or eyelashes are covering too much of your "
                        "iris to inspect it properly.")
        return "Not enough clear iris texture is visible in this photo."

    return error or "I couldn't analyze the iris in this eye."


def _diagnose_normalization_failure(norm_result):
    """Translates Module 3's technical error into an IRISCOPE-toned message."""
    error = norm_result.get("error")

    if error == "Iris ring too narrow to normalize reliably.":
        return "The iris is too small to inspect properly."

    if error == "Not enough usable iris texture survived normalization.":
        return "Not enough clear iris texture survived for reliable analysis."

    return error or "I couldn't process this iris further."


def _diagnose_counting_failure(line_result):
    """Translates Module 4's technical error into an IRISCOPE-toned message."""
    error = line_result.get("error")

    if error == "No valid iris texture to analyze.":
        return "There isn't enough valid iris texture here to count anything reliably."

    return error or "I couldn't count structures in this iris."


# ---------------------------------------------------------------------------
# Per-eye analysis
# ---------------------------------------------------------------------------

def _empty_metrics():
    return {
        "structure_count": None,
        "detection_quality": None,
        "usable_area_ratio": None,
        "valid_ratio": None,
        "structure_density": None,
        "raw_candidate_count": None,
    }


def analyze_eye(image_bgr, eye):
    """
    Runs Modules 2-4 on one detected eye (an entry from Module 1's
    detect_eyes() output) and assembles a full per-eye result:

    {
        "position": "left" | "right" | "unknown",
        "success": bool,
        "error": str or None,          # friendly message, only set on failure
        "failed_stage": str or None,   # "crop" | "segmentation" | "normalization" | "counting"
        "stages": {                    # only the stages actually produced
            "eye_region", "iris_region", "mask", "normalized",
            "enhanced", "detected", "final",
        },
        "metrics": {
            "structure_count", "detection_quality", "usable_area_ratio",
            "valid_ratio", "structure_density", "raw_candidate_count",
        },
    }

    Stops and returns as soon as any stage fails -- later stages are never
    faked or guessed at. Whatever "stages" dict keys exist reflect real
    processing that actually happened.
    """
    result = {
        "position": eye["position"],
        "success": False,
        "error": None,
        "failed_stage": None,
        "stages": {},
        "metrics": _empty_metrics(),
    }

    eye_crop = get_eye_crop(image_bgr, eye)
    if eye_crop.size == 0:
        result["error"] = ("This eye crop came out empty -- the detected region "
                            "was too close to the image edge.")
        result["failed_stage"] = "crop"
        result["stages"]["final"] = _build_error_card(result["error"])
        return result

    result["stages"]["eye_region"] = eye_crop

    bx, by, bw, bh = eye["box"]
    cx, cy, _cw, _ch = eye["crop_box"]
    eye_box_in_crop = (bx - cx, by - cy, bw, bh)

    seg_result = segment_iris(eye_crop, eye_box_in_crop=eye_box_in_crop)
    result["stages"]["iris_region"] = draw_circles_overlay(eye_crop, seg_result)

    if not seg_result["success"]:
        result["error"] = _diagnose_segmentation_failure(seg_result)
        result["failed_stage"] = "segmentation"
        result["stages"]["final"] = _build_error_card(result["error"])
        return result

    result["metrics"]["usable_area_ratio"] = seg_result["usable_area_ratio"]
    result["stages"]["mask"] = _build_mask_visualization(eye_crop, seg_result["masks"])

    norm_result = normalize_iris(eye_crop, seg_result)

    if not norm_result["success"]:
        result["error"] = _diagnose_normalization_failure(norm_result)
        result["failed_stage"] = "normalization"
        if norm_result.get("normalized_gray") is not None:
            result["stages"]["normalized"] = _gray_to_bgr(norm_result["normalized_gray"])
        result["stages"]["final"] = _build_error_card(result["error"])
        return result

    result["metrics"]["valid_ratio"] = norm_result["valid_ratio"]
    result["stages"]["normalized"] = _gray_to_bgr(norm_result["normalized_gray"])
    result["stages"]["enhanced"] = _gray_to_bgr(norm_result["normalized_enhanced"])

    line_result = count_radial_structures(norm_result)

    if not line_result["success"]:
        result["error"] = _diagnose_counting_failure(line_result)
        result["failed_stage"] = "counting"
        result["stages"]["final"] = _build_error_card(result["error"])
        return result

    structure_count = line_result["structure_count"]
    usable_degrees = norm_result["valid_ratio"] * config.NORM_ANGLE_SAMPLES
    structure_density = (
        round((structure_count / usable_degrees) * 100, 2) if usable_degrees > 0 else None
    )

    result["success"] = True
    result["metrics"].update({
        "structure_count": structure_count,
        "detection_quality": line_result["detection_quality"],
        "structure_density": structure_density,
        "raw_candidate_count": line_result["raw_candidate_count"],
    })
    result["stages"]["detected"] = _build_detected_overlay(eye_crop, seg_result, line_result)
    result["stages"]["final"] = _build_final_card(eye_crop, seg_result, line_result, result["metrics"])

    return result


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def analyze_iris(image_bgr):
    """
    Main entry point for Module 5.

    Runs the full IRISCOPE pipeline (Modules 1-4) on a raw BGR image.

    {
        "success": bool,             # True if at least one eye fully analyzed
        "error": str or None,        # set when nothing could be analyzed at all
        "stages": {"original", "detection_overlay"},
        "eyes": [analyze_eye() result, ...],
    }
    """
    image, _scale = resize_if_large(image_bgr)

    result = {
        "success": False,
        "error": None,
        "stages": {"original": image},
        "eyes": [],
    }

    detection = detect_eyes(image)
    result["stages"]["detection_overlay"] = draw_debug_overlay(image, detection)

    if not detection["success"]:
        result["error"] = detection["error"]
        result["stages"]["final"] = _build_error_card(result["error"])
        return result

    eyes_results = [analyze_eye(image, eye) for eye in detection["eyes"]]
    result["eyes"] = eyes_results
    result["success"] = any(eye["success"] for eye in eyes_results)

    if not result["success"] and eyes_results:
        # Every detected eye failed somewhere downstream. Surface the first
        # eye's explanation at the top level too, so callers checking
        # result["error"] alone still get something useful.
        result["error"] = eyes_results[0]["error"]

    return result


def analyze_iris_file(path):
    """
    Convenience wrapper: loads an image from disk and runs analyze_iris()
    on it, translating load failures (missing file, corrupt/unsupported
    format) and any unexpected exception into the same honest result-dict
    shape as everything else, instead of raising or crashing.
    """
    try:
        image = load_image(path)
    except FileNotFoundError:
        return _top_level_error("I can't find that image file.")
    except ValueError:
        return _top_level_error("I can't read this image. Try a JPG or PNG photo instead.")

    try:
        return analyze_iris(image)
    except Exception as exc:  # last-resort safety net -- a bad photo should never crash the app
        print(f"[IRISCOPE] Unexpected analysis failure on '{path}': {exc}")
        return _top_level_error("Something went wrong analyzing this photo. Try a different one.")


def _top_level_error(message):
    result = {"success": False, "error": message, "stages": {}, "eyes": []}
    result["stages"]["final"] = _build_error_card(message)
    return result


# ---------------------------------------------------------------------------
# Visualization builders
#
# These produce presentation-quality images from Modules 1-4's actual
# output. They deliberately don't reuse Modules 2/4's own draw_* helpers
# unchanged -- those are documented as debug/dev tools -- but they draw
# from the same real data, using the same IRISCOPE color roles (coral =
# major result, teal = detected/confirmed, mustard = measurement/exclusion).
# ---------------------------------------------------------------------------

def _gray_to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _build_mask_visualization(eye_crop_bgr, masks):
    """
    Colors the eye crop by mask role: teal = usable iris texture, mustard =
    eyelid/eyelash exclusion, red = reflection exclusion. Everything else
    (pupil, sclera, background) is left as the original crop. Built
    directly from Module 2's own mask arrays -- not a stylized guess.
    """
    vis = eye_crop_bgr.copy()
    tint = np.zeros_like(vis)
    tint[masks["usable_mask"] > 0] = config.COLOR_USABLE_REGION
    tint[masks["eyelid_mask"] > 0] = config.COLOR_EYELID_MASK
    tint[masks["reflection_mask"] > 0] = config.COLOR_REFLECTION_MASK

    colored = tint.astype(bool).any(axis=2)
    blended = cv2.addWeighted(vis, 0.4, tint, 0.6, 0)
    vis[colored] = blended[colored]
    return vis


def _build_detected_overlay(eye_crop_bgr, seg_result, line_result):
    """
    Presentation version of Module 4's tick-mark overlay: a clean radial
    tick per detected structure plus both the pupil and iris boundary
    circles for context, so a viewer can see exactly what was counted
    without needing the raw column profile.

    Drawn at the crop's native resolution with 1px lines -- these eye crops
    are small (often well under 100px across), so anything thicker balloons
    into solid wedges once the panel scales the image up for display, which
    would misrepresent how fine the actual detected structures are.
    """
    annotated = eye_crop_bgr.copy()
    cx, cy = seg_result["iris_center"]
    pupil_r = seg_result["pupil"][2]
    iris_r = seg_result["iris_radius"]
    span = iris_r - pupil_r
    r_inner = pupil_r + span * config.NORM_INNER_MARGIN_RATIO
    r_outer = pupil_r + span * config.NORM_OUTER_MARGIN_RATIO

    cv2.circle(annotated, (cx, cy), iris_r, config.COLOR_RESULT_TEXT, 1, cv2.LINE_AA)
    cv2.circle(annotated, (cx, cy), pupil_r, config.COLOR_RESULT_TEXT, 1, cv2.LINE_AA)

    for structure in line_result["structures"]:
        angle_rad = np.radians(structure["center_deg"])
        x1 = int(round(cx + r_inner * np.cos(angle_rad)))
        y1 = int(round(cy + r_inner * np.sin(angle_rad)))
        x2 = int(round(cx + r_outer * np.cos(angle_rad)))
        y2 = int(round(cy + r_outer * np.sin(angle_rad)))
        cv2.line(annotated, (x1, y1), (x2, y2), config.COLOR_STRUCTURE_ACCEPTED, 1, cv2.LINE_AA)

    return annotated


def _stat_line(panel, label, value, y, ink_color, value_color):
    cv2.putText(panel, label, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ink_color, 1, cv2.LINE_AA)
    (text_w, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(panel, value, (panel.shape[1] - 16 - text_w, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, value_color, 1, cv2.LINE_AA)
    return y + 26


def _build_final_card(eye_crop_bgr, seg_result, line_result, metrics):
    """
    Assembles the "FINAL RESULT" stage: the detected-structures overlay at
    a larger, readable size next to a text panel with the headline count
    and supporting metrics. Deliberately simple -- no custom typography or
    layout system, that's Module 7's web design work -- but it uses the
    IRISCOPE color roles consistently (coral = major result, teal =
    detected/confirmed, mustard = supporting measurement) and only shows
    metrics the pipeline actually produced.
    """
    eye_h, eye_w = eye_crop_bgr.shape[:2]
    scale = config.RESULT_CARD_EYE_HEIGHT / eye_h
    overlay = _build_detected_overlay(eye_crop_bgr, seg_result, line_result)
    overlay_resized = cv2.resize(
        overlay, (max(1, int(eye_w * scale)), config.RESULT_CARD_EYE_HEIGHT),
    )

    panel_w = max(config.RESULT_CARD_TEXT_PANEL_MIN_WIDTH, 260)
    panel = np.full((config.RESULT_CARD_EYE_HEIGHT, panel_w, 3), config.COLOR_BACKGROUND_CREAM, dtype=np.uint8)

    ink = config.COLOR_TEXT
    coral = config.COLOR_RESULT_TEXT
    teal = config.COLOR_STRUCTURE_ACCEPTED

    y = 32
    cv2.putText(panel, "IRIS STRUCTURE COUNT", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ink, 1, cv2.LINE_AA)
    y += 46
    cv2.putText(panel, str(metrics["structure_count"]), (16, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, coral, 3, cv2.LINE_AA)
    y += 24
    cv2.putText(panel, "DETECTABLE RADIAL STRUCTURES", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, ink, 1, cv2.LINE_AA)
    y += 34

    cv2.line(panel, (16, y), (panel_w - 16, y), (200, 210, 216), 1)
    y += 22

    y = _stat_line(panel, "Analysis quality", f"{metrics['detection_quality']:.0%}", y, ink, teal)
    y = _stat_line(panel, "Usable iris area", f"{metrics['usable_area_ratio']:.0%}", y, ink, teal)
    if metrics["structure_density"] is not None:
        y = _stat_line(panel, "Structure density", f"{metrics['structure_density']:.1f} / 100 deg", y, ink, teal)

    return np.hstack([overlay_resized, panel])


def _build_error_card(message, width=440, height=140):
    """
    A simple card rendering a failure explanation in place of a results
    card, used whenever analysis couldn't reach a full result. This only
    lays messages out -- it never invents new ones beyond what
    _diagnose_*_failure() already produced.
    """
    card = np.full((height, width, 3), config.COLOR_BACKGROUND_CREAM, dtype=np.uint8)
    cv2.putText(card, "ANALYSIS DID NOT COMPLETE", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.COLOR_TEXT, 1, cv2.LINE_AA)

    words = message.split(" ")
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) > 44:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)

    y = 58
    for line in lines:
        cv2.putText(card, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_RESULT_TEXT, 1, cv2.LINE_AA)
        y += 24

    return card


# ---------------------------------------------------------------------------
# Multi-stage panel (debug/demo visualization of the whole pipeline)
# ---------------------------------------------------------------------------

_STAGE_ORDER = [
    "original", "detection_overlay", "eye_region", "iris_region",
    "mask", "normalized", "enhanced", "detected", "final",
]

_STAGE_LABELS = {
    "original": "1. ORIGINAL",
    "detection_overlay": "FACE + EYE DETECTION",
    "eye_region": "2. EYE REGION",
    "iris_region": "3. IRIS REGION",
    "mask": "4. MASK",
    "normalized": "5. NORMALIZED",
    "enhanced": "6. ENHANCED",
    "detected": "7. DETECTED STRUCTURES",
    "final": "8. FINAL RESULT",
}

_STRIP_STAGES = {"normalized", "enhanced"}


_LABEL_FONT_SCALE = 0.34


def _add_label(image_bgr, label):
    """
    Adds a label bar under an image. If the image is narrower than the
    label text needs (common for tall portrait crops shrunk to a small
    thumbnail height), widens the whole stage -- image and bar alike,
    padding the image with cream on both sides -- rather than shrinking
    the font to the point of being unreadable.
    """
    (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _LABEL_FONT_SCALE, 1)
    required_w = text_w + 12
    h, w = image_bgr.shape[:2]

    if w < required_w:
        pad_total = required_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        image_bgr = cv2.copyMakeBorder(
            image_bgr, 0, 0, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=config.COLOR_BACKGROUND_CREAM,
        )
        w = required_w

    bar = np.full((config.RESULT_LABEL_BAR_HEIGHT, w, 3), config.COLOR_BACKGROUND_CREAM, dtype=np.uint8)
    cv2.putText(bar, label, (6, config.RESULT_LABEL_BAR_HEIGHT - 7),
                cv2.FONT_HERSHEY_SIMPLEX, _LABEL_FONT_SCALE, config.COLOR_TEXT, 1, cv2.LINE_AA)
    return np.vstack([image_bgr, bar])


def _labeled_stage(image_bgr, label, target_height):
    h, w = image_bgr.shape[:2]
    scale = target_height / h
    resized = cv2.resize(image_bgr, (max(1, int(w * scale)), target_height))
    return _add_label(resized, label)


def _stage_at_native_size(image_bgr, label):
    return _add_label(image_bgr, label)


def get_stage_entries(eye_result, top_level_stages=None):
    """
    Module 10 -- returns every stage image that actually exists for this
    eye, individually, in the project's documented order:
    [{"key": "eye_region", "label": "2. EYE REGION", "image": <BGR array>}, ...]

    This is the same real data build_stage_panel() above stitches into one
    flattened JPEG -- reused here, not recomputed -- so the frontend can
    build an interactive stage-by-stage viewer (tabs, zoom, compare) using
    the actual pipeline output instead of one static composite image.
    Stages that don't exist because the pipeline stopped early are
    skipped, never faked -- same honesty rule as build_stage_panel.
    """
    all_stages = dict(top_level_stages or {})
    all_stages.update(eye_result.get("stages", {}))

    entries = []
    for key in _STAGE_ORDER:
        image = all_stages.get(key)
        if image is None:
            continue
        entries.append({"key": key, "label": _STAGE_LABELS[key], "image": image})
    return entries


def build_stage_panel(eye_result, top_level_stages=None):
    """
    Assembles every stage image that was actually produced -- the shared
    whole-image stages plus one eye's own stages -- into a single labeled
    vertical panel, in the project's documented stage order. Stages that
    don't exist (because the pipeline failed before producing them) are
    skipped, never faked.
    """
    all_stages = dict(top_level_stages or {})
    all_stages.update(eye_result.get("stages", {}))

    panels = []
    for key in _STAGE_ORDER:
        image = all_stages.get(key)
        if image is None:
            continue
        if key == "final":
            panels.append(_stage_at_native_size(image, _STAGE_LABELS[key]))
        elif key in _STRIP_STAGES:
            panels.append(_labeled_stage(image, _STAGE_LABELS[key], config.RESULT_STRIP_STAGE_HEIGHT))
        else:
            panels.append(_labeled_stage(image, _STAGE_LABELS[key], config.RESULT_CROP_STAGE_HEIGHT))

    if not panels:
        return None

    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.full((p.shape[0], max_w - p.shape[1], 3), config.COLOR_BACKGROUND_CREAM, dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)

    return np.vstack(padded)
