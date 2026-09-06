"""
Manual test runner for Module 2 (Iris Segmentation).

Runs Module 1's face/eye detection, then Module 2's iris segmentation, on
every image in data/test_images. Saves a labeled panel per eye showing each
stage (crop / pupil+iris circles / eyelid mask / reflection mask / final
usable region) so the segmentation can actually be inspected visually,
rather than just checking that the code ran without an exception.

Run from the project root:
    python tests/test_module2.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import cv2
import numpy as np

import config
from utils import load_image, resize_if_large, save_image
from face_eye_detection import detect_eyes, get_eye_crop
from iris_segmentation import segment_iris, draw_circles_overlay, draw_mask_overlay

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "test_images")
OUTPUT_DIR = os.path.join(TEST_IMAGES_DIR, "output_module2")

PANEL_HEIGHT = 160  # every panel image is resized to this height before being labeled and joined
LABEL_BAR_HEIGHT = 18


def _labeled_panel(image, label):
    """Resizes an image to a fixed height and stamps a small text label
    underneath it, so panels can be joined into one readable strip."""
    h, w = image.shape[:2]
    scale = PANEL_HEIGHT / h
    resized = cv2.resize(image, (int(w * scale), PANEL_HEIGHT))

    bar = np.full((LABEL_BAR_HEIGHT, resized.shape[1], 3), 244, dtype=np.uint8)  # cream background
    cv2.putText(bar, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (61, 52, 47), 1)

    return np.vstack([resized, bar])


def _build_debug_panel(eye_crop, result):
    """Builds a horizontal strip of every segmentation stage for one eye."""
    panels = [_labeled_panel(eye_crop, "eye crop")]

    if result["pupil"] is not None:
        circles = draw_circles_overlay(eye_crop, result)
        panels.append(_labeled_panel(circles, "pupil + iris"))

    if result["masks"] is not None:
        eyelid_view = draw_mask_overlay(eye_crop, result["masks"]["eyelid_mask"],
                                         config.COLOR_EYELID_MASK)
        panels.append(_labeled_panel(eyelid_view, "eyelid/lash mask"))

        reflection_view = draw_mask_overlay(eye_crop, result["masks"]["reflection_mask"],
                                             config.COLOR_REFLECTION_MASK)
        panels.append(_labeled_panel(reflection_view, "reflection mask"))

        usable_view = draw_mask_overlay(eye_crop, result["masks"]["usable_mask"],
                                         config.COLOR_USABLE_REGION)
        panels.append(_labeled_panel(usable_view, f"usable ({result['usable_area_ratio']:.0%})"))

    # pad all panels to the same width so hstack doesn't fail on odd aspect ratios
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.full((p.shape[0], max_w - p.shape[1], 3), 244, dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)

    return np.hstack(padded)


def run_on_image(filename):
    path = os.path.join(TEST_IMAGES_DIR, filename)
    name = os.path.splitext(filename)[0]

    image = load_image(path)
    image, _ = resize_if_large(image)

    detection = detect_eyes(image)
    print(f"\n{filename}")

    if not detection["success"]:
        print(f"  Module 1 did not find usable eyes ({detection['error']}) -- skipping segmentation.")
        return []

    eye_results = []
    for eye in detection["eyes"]:
        eye_crop = get_eye_crop(image, eye)
        if eye_crop.size == 0:
            continue

        # Translate Module 1's raw (unpadded) eye box from full-image
        # coordinates into this crop's local coordinates, so segment_iris
        # knows where the actual eye opening is within the padded crop.
        bx, by, bw, bh = eye["box"]
        cx, cy, _cw, _ch = eye["crop_box"]
        eye_box_in_crop = (bx - cx, by - cy, bw, bh)

        result = segment_iris(eye_crop, eye_box_in_crop=eye_box_in_crop)
        eye_results.append((eye["position"], result))

        status = "OK" if result["success"] else f"FAILED ({result['error']})"
        usable = f", usable area {result['usable_area_ratio']:.0%}" if result["usable_area_ratio"] is not None else ""
        print(f"  {eye['position']:5s} eye: {status}{usable}")

        panel = _build_debug_panel(eye_crop, result)
        save_image(os.path.join(OUTPUT_DIR, f"{name}_{eye['position']}_segmentation.jpg"), panel)

    return eye_results


def main():
    image_files = sorted(
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    print(f"Running Module 2 segmentation on {len(image_files)} test image(s)...")

    total_eyes = 0
    successful_eyes = 0
    for filename in image_files:
        for _position, result in run_on_image(filename):
            total_eyes += 1
            if result["success"]:
                successful_eyes += 1

    print(f"\n{'=' * 40}")
    print(f"Summary: {successful_eyes}/{total_eyes} detected eyes segmented successfully")
    print(f"Debug panels saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
