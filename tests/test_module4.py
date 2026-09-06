"""
Manual test runner for Module 4 (Radial Structure Detection + Counting).

Runs the full pipeline -- Module 1 detection, Module 2 segmentation,
Module 3 normalization, Module 4 counting -- on every image in
data/test_images. For every eye that Module 3 successfully normalizes,
saves a labeled panel showing the segmented crop, the edge map, the
per-column accept/reject profile, and the final structures drawn back onto
the eye, so the count can actually be inspected visually rather than
trusted from the number alone.

Run from the project root:
    python tests/test_module4.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import cv2
import numpy as np

import config
from utils import load_image, resize_if_large, save_image
from face_eye_detection import detect_eyes, get_eye_crop
from iris_segmentation import segment_iris
from iris_normalization import normalize_iris
from line_detection import count_radial_structures, draw_column_profile, draw_structures_overlay

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "test_images")
OUTPUT_DIR = os.path.join(TEST_IMAGES_DIR, "output_module4")

CROP_PANEL_HEIGHT = 160
STRIP_DISPLAY_HEIGHT = 100
PROFILE_HEIGHT = 60
LABEL_BAR_HEIGHT = 18


def _labeled_panel(image_bgr, label, target_height):
    h, w = image_bgr.shape[:2]
    scale = target_height / h
    resized = cv2.resize(image_bgr, (max(1, int(w * scale)), target_height))

    bar = np.full((LABEL_BAR_HEIGHT, resized.shape[1], 3), 244, dtype=np.uint8)  # cream background
    cv2.putText(bar, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (61, 52, 47), 1)

    return np.vstack([resized, bar])


def _to_bgr(gray_or_none, shape_if_none):
    if gray_or_none is None:
        return np.zeros((*shape_if_none, 3), dtype=np.uint8)
    return cv2.cvtColor(gray_or_none, cv2.COLOR_GRAY2BGR)


def _build_debug_panel(eye_crop, seg_result, norm_result, line_result):
    panels = []

    overlay = draw_structures_overlay(eye_crop, seg_result, norm_result, line_result)
    panels.append(_labeled_panel(overlay, "final structures", CROP_PANEL_HEIGHT))

    strip_shape = (config.NORM_RADIAL_SAMPLES, config.NORM_ANGLE_SAMPLES)
    edges_bgr = _to_bgr(norm_result["normalized_edges"], strip_shape)
    panels.append(_labeled_panel(edges_bgr, "edge map (Module 3 input)", STRIP_DISPLAY_HEIGHT))

    profile = draw_column_profile(line_result, bar_height=PROFILE_HEIGHT)
    count = line_result["structure_count"] if line_result["success"] else "n/a"
    panels.append(_labeled_panel(profile, f"column profile -> count: {count}", PROFILE_HEIGHT))

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
            print(f"  {eye['position']:5s} eye: Module 2 segmentation failed ({seg_result['error']}) -- skipping.")
            continue

        norm_result = normalize_iris(eye_crop, seg_result)
        if not norm_result["success"]:
            print(f"  {eye['position']:5s} eye: Module 3 normalization failed ({norm_result['error']}) -- skipping.")
            continue

        line_result = count_radial_structures(norm_result)
        results.append((eye["position"], line_result))

        if line_result["success"]:
            print(
                f"  {eye['position']:5s} eye: {line_result['structure_count']} detectable radial structures "
                f"(raw candidates: {line_result['raw_candidate_count']}, "
                f"detection quality: {line_result['detection_quality']:.0%})"
            )
        else:
            print(f"  {eye['position']:5s} eye: FAILED ({line_result['error']})")

        panel = _build_debug_panel(eye_crop, seg_result, norm_result, line_result)
        save_image(os.path.join(OUTPUT_DIR, f"{name}_{eye['position']}_structures.jpg"), panel)

    return results


def main():
    image_files = sorted(
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    print(f"Running Module 4 radial structure counting on {len(image_files)} test image(s)...")

    total = 0
    successful = 0
    counts = []
    for filename in image_files:
        for _position, line_result in run_on_image(filename):
            total += 1
            if line_result["success"]:
                successful += 1
                counts.append(line_result["structure_count"])

    print(f"\n{'=' * 40}")
    print(f"Summary: {successful}/{total} eyes produced a structure count")
    if counts:
        print(f"Counts observed: {counts}")
    print(f"Debug panels saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
