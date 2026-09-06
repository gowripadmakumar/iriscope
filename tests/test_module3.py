"""
Manual test runner for Module 3 (Normalization + Texture Analysis).

Runs the full pipeline -- Module 1 detection, Module 2 segmentation, Module 3
normalization -- on every image in data/test_images. For every eye that
Module 2 successfully segments, saves a labeled panel showing each Module 3
stage (segmented crop / raw unwrapped / enhanced / edge map) so the
normalization can actually be inspected visually.

Run from the project root:
    python tests/test_module3.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import cv2
import numpy as np

import config
from utils import load_image, resize_if_large, save_image
from face_eye_detection import detect_eyes, get_eye_crop
from iris_segmentation import segment_iris, draw_circles_overlay
from iris_normalization import normalize_iris

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "test_images")
OUTPUT_DIR = os.path.join(TEST_IMAGES_DIR, "output_module3")

# Debug panels stack the (small, tall) polar strips above each other, so a
# taller display height than Module 2's panels is used to keep them readable.
STRIP_DISPLAY_HEIGHT = 120
CROP_PANEL_HEIGHT = 160
LABEL_BAR_HEIGHT = 18


def _labeled_panel(image_bgr, label, target_height):
    h, w = image_bgr.shape[:2]
    scale = target_height / h
    resized = cv2.resize(image_bgr, (int(w * scale), target_height))

    bar = np.full((LABEL_BAR_HEIGHT, resized.shape[1], 3), 244, dtype=np.uint8)  # cream background
    cv2.putText(bar, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (61, 52, 47), 1)

    return np.vstack([resized, bar])


def _to_bgr(gray_or_none, shape_if_none):
    """Converts a single-channel image to BGR for display, or produces a
    blank placeholder panel if the stage didn't run (e.g. failed early)."""
    if gray_or_none is None:
        return np.zeros((*shape_if_none, 3), dtype=np.uint8)
    return cv2.cvtColor(gray_or_none, cv2.COLOR_GRAY2BGR)


def _build_debug_panel(eye_crop, seg_result, norm_result):
    panels = []

    circles = draw_circles_overlay(eye_crop, seg_result)
    panels.append(_labeled_panel(circles, "segmented", CROP_PANEL_HEIGHT))

    strip_shape = (config.NORM_RADIAL_SAMPLES, config.NORM_ANGLE_SAMPLES)

    raw_bgr = _to_bgr(norm_result["normalized_gray"], strip_shape)
    panels.append(_labeled_panel(raw_bgr, "unwrapped (raw)", STRIP_DISPLAY_HEIGHT))

    mask_bgr = _to_bgr(norm_result["normalized_mask"], strip_shape)
    panels.append(_labeled_panel(mask_bgr, "unwrapped mask", STRIP_DISPLAY_HEIGHT))

    enhanced_bgr = _to_bgr(norm_result["normalized_enhanced"], strip_shape)
    label = f"enhanced ({norm_result['valid_ratio']:.0%} valid)" if norm_result["valid_ratio"] is not None else "enhanced"
    panels.append(_labeled_panel(enhanced_bgr, label, STRIP_DISPLAY_HEIGHT))

    edges_bgr = _to_bgr(norm_result["normalized_edges"], strip_shape)
    panels.append(_labeled_panel(edges_bgr, "edge map", STRIP_DISPLAY_HEIGHT))

    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.full((p.shape[0], max_w - p.shape[1], 3), 244, dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)

    return np.vstack(padded)


def run_on_image(filename):
    path = os.path.join(TEST_IMAGES_DIR, filename)
    name = os.path.splitext(filename)[0]

    image = load_image(path)
    image, _ = resize_if_large(image)

    detection = detect_eyes(image)
    print(f"\n{filename}")

    if not detection["success"]:
        print(f"  Module 1 did not find usable eyes ({detection['error']}) -- skipping.")
        return []

    results = []
    for eye in detection["eyes"]:
        eye_crop = get_eye_crop(image, eye)
        if eye_crop.size == 0:
            continue

        bx, by, bw, bh = eye["box"]
        cx, cy, _cw, _ch = eye["crop_box"]
        eye_box_in_crop = (bx - cx, by - cy, bw, bh)

        seg_result = segment_iris(eye_crop, eye_box_in_crop=eye_box_in_crop)
        if not seg_result["success"]:
            print(f"  {eye['position']:5s} eye: Module 2 segmentation failed ({seg_result['error']}) -- skipping normalization.")
            continue

        norm_result = normalize_iris(eye_crop, seg_result)
        results.append((eye["position"], norm_result))

        status = "OK" if norm_result["success"] else f"FAILED ({norm_result['error']})"
        valid = f", {norm_result['valid_ratio']:.0%} valid" if norm_result["valid_ratio"] is not None else ""
        print(f"  {eye['position']:5s} eye: {status}{valid}")

        panel = _build_debug_panel(eye_crop, seg_result, norm_result)
        save_image(os.path.join(OUTPUT_DIR, f"{name}_{eye['position']}_normalization.jpg"), panel)

    return results


def main():
    image_files = sorted(
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    print(f"Running Module 3 normalization on {len(image_files)} test image(s)...")

    total = 0
    successful = 0
    for filename in image_files:
        for _position, norm_result in run_on_image(filename):
            total += 1
            if norm_result["success"]:
                successful += 1

    print(f"\n{'=' * 40}")
    print(f"Summary: {successful}/{total} Module-2-successful eyes normalized successfully")
    print(f"Debug panels saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
