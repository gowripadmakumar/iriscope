"""
Module 6 -- Web application + image input.

This is a thin Flask layer around the existing Modules 1-5 pipeline. It
contains no computer-vision logic of its own -- every submitted image is
handed straight to analyzer.analyze_iris(), the exact function
tests/test_module5.py already exercises against the project's test images.

Upload and camera capture both arrive here the same way: one image file
in a POST body. There is exactly one code path from "image bytes" to
"analysis result" regardless of which input method produced those bytes.

Images are decoded and processed entirely in memory (utils.decode_image_bytes)
and are never written to disk.
"""

import base64
import os

import cv2
from flask import Flask, jsonify, request, send_from_directory

import config
from analyzer import analyze_iris, build_stage_panel, get_stage_entries
from compatibility import match_irises
from utils import decode_image_bytes

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE_BYTES


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ---------------------------------------------------------------------------
# Analysis API
# ---------------------------------------------------------------------------

def _encode_image_base64(image_bgr):
    """BGR NumPy array -> base64 JPEG data URL, for embedding in JSON."""
    if image_bgr is None:
        return None
    ok, buffer = cv2.imencode(
        ".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, config.API_RESULT_JPEG_QUALITY]
    )
    if not ok:
        return None
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _encode_stage_entries(entries):
    """
    Module 10 -- encodes get_stage_entries()'s real intermediate images
    individually (key + human-readable label + base64 JPEG) so the
    frontend can build an interactive stage-by-stage viewer instead of
    only showing one flattened composite image.
    """
    return [
        {"key": entry["key"], "label": entry["label"], "image": _encode_image_base64(entry["image"])}
        for entry in entries
    ]


def _eye_to_json(eye_result, top_level_stages):
    """
    Turns one analyze_eye() result into a JSON-safe dict: the metrics and
    error message it already contains, its individual pipeline-stage
    images (Module 10, for the interactive viewer), and -- kept for
    backward compatibility -- the same flattened multi-stage panel Module
    5 already built (build_stage_panel, not reimplemented here).
    """
    panel = build_stage_panel(eye_result, top_level_stages=top_level_stages)
    stages = get_stage_entries(eye_result, top_level_stages=top_level_stages)
    return {
        "position": eye_result["position"],
        "success": eye_result["success"],
        "error": eye_result["error"],
        "metrics": eye_result["metrics"],
        "panel_image": _encode_image_base64(panel),
        "stages": _encode_stage_entries(stages),
    }


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return jsonify(success=False, error="No image was received.", eyes=[]), 400

    ext = os.path.splitext(upload.filename)[1].lower()
    if ext and ext not in config.ALLOWED_UPLOAD_EXTENSIONS:
        return jsonify(
            success=False, error="Please provide a JPG or PNG image.", eyes=[]
        ), 400

    data = upload.read()
    if not data:
        return jsonify(success=False, error="The uploaded image was empty.", eyes=[]), 400

    try:
        image = decode_image_bytes(data)
    except ValueError as exc:
        return jsonify(success=False, error=str(exc), eyes=[]), 400

    try:
        result = analyze_iris(image)
    except Exception as exc:  # last-resort safety net -- a bad photo should never crash the app
        app.logger.error(f"[IRISCOPE] Unexpected analysis failure: {exc}")
        return jsonify(
            success=False,
            error="Something went wrong analyzing this photo. Try a different one.",
            eyes=[],
        ), 500

    response = {
        "success": result["success"],
        "error": result["error"],
        "eyes": [_eye_to_json(eye, result["stages"]) for eye in result["eyes"]],
    }

    if not result["eyes"]:
        # Top-level failure before any eye was even reached (no face / no
        # eye detected at all). Still return a panel/stages so the browser
        # can show what detection actually saw.
        panel = build_stage_panel({"stages": {}}, top_level_stages=result["stages"])
        response["panel_image"] = _encode_image_base64(panel)
        response["stages"] = _encode_stage_entries(
            get_stage_entries({"stages": {}}, top_level_stages=result["stages"])
        )

    return jsonify(response)



# ---------------------------------------------------------------------------
# IRIS MATCH API (Module 7)
# ---------------------------------------------------------------------------

def _load_match_image(field_name, label):
    """
    Validates and decodes one of IRIS MATCH's two uploaded photos. Mirrors
    /api/analyze's own validation (same allowed extensions, same in-memory
    decode via decode_image_bytes) -- no new upload-handling logic, just
    applied twice with a labeled error message so the browser can say
    which of the two photos had the problem.

    Returns (image, error_message) -- exactly one of the two is None.
    """
    upload = request.files.get(field_name)
    if upload is None or upload.filename == "":
        return None, f"No {label} photo was received."

    ext = os.path.splitext(upload.filename)[1].lower()
    if ext and ext not in config.ALLOWED_UPLOAD_EXTENSIONS:
        return None, f"Please provide a JPG or PNG image for the {label} photo."

    data = upload.read()
    if not data:
        return None, f"The {label} photo was empty."

    try:
        return decode_image_bytes(data), None
    except ValueError as exc:
        return None, f"{label.capitalize()} photo: {exc}"


def _eye_summary_to_json(eye_result):
    """A slimmer per-eye payload for IRIS MATCH: the headline metrics plus
    Module 5's own 'final' result card (already built by analyze_eye) --
    not the full 8-stage panel, which would be too much detail for a
    side-by-side match card."""
    return {
        "position": eye_result["position"],
        "metrics": eye_result["metrics"],
        "card_image": _encode_image_base64(eye_result["stages"].get("final")),
    }


@app.route("/api/match", methods=["POST"])
def api_match():
    image_a, error_a = _load_match_image("image_a", "first")
    image_b, error_b = _load_match_image("image_b", "second")

    if error_a or error_b:
        return jsonify(success=False, error=error_a or error_b), 400

    try:
        result = match_irises(image_a, image_b)
    except Exception as exc:  # last-resort safety net, same policy as /api/analyze
        app.logger.error(f"[IRISCOPE] Unexpected match failure: {exc}")
        return jsonify(
            success=False, error="Something went wrong comparing these photos. Try different ones."
        ), 500

    response = {"success": result["success"], "error": result["error"]}
    if result["success"]:
        response["scores"] = result["scores"]
        response["verdict"] = result["verdict"]
        response["eye_a"] = _eye_summary_to_json(result["eye_a"])
        response["eye_b"] = _eye_summary_to_json(result["eye_b"])

    return jsonify(response)


@app.errorhandler(413)
def too_large(_exc):
    return jsonify(success=False, error="That image is too large. Try a smaller photo.", eyes=[]), 413


@app.errorhandler(500)
def server_error(_exc):
    return jsonify(success=False, error="Something went wrong on the server.", eyes=[]), 500


if __name__ == "__main__":
    # Prefer running via run.py at the project root -- this block exists
    # only so `python backend/app.py` also works. Same debug-mode policy
    # as run.py: off by default, since debug=True on a network-bound host
    # exposes Werkzeug's interactive debugger.
    debug = os.environ.get("IRISCOPE_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)
