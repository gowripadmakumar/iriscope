"""
Manual test runner for Module 1 (Computer Vision Foundation).

Not a unit-test-framework suite — this is a visual inspection tool.
It runs the detector against every image in data/test_images, prints a
summary, and saves annotated debug images so we can actually look at the
results instead of trusting that "no exception was thrown" means it works.

Run from the project root:
    python tests/test_module1.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from utils import load_image, resize_if_large, save_image
from face_eye_detection import detect_eyes, get_eye_crop, draw_debug_overlay

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "test_images")
OUTPUT_DIR = os.path.join(TEST_IMAGES_DIR, "output")


def run_on_image(filename):
    path = os.path.join(TEST_IMAGES_DIR, filename)
    name = os.path.splitext(filename)[0]

    image = load_image(path)
    image, scale = resize_if_large(image)

    start = time.time()
    result = detect_eyes(image)
    elapsed_ms = (time.time() - start) * 1000

    print(f"\n{filename}")
    print(f"  processed size : {image.shape[1]}x{image.shape[0]} (scale {scale:.2f})")
    print(f"  time           : {elapsed_ms:.1f} ms")
    print(f"  face found     : {result['face_box'] is not None}")
    print(f"  eyes found     : {len(result['eyes'])}")
    if not result["success"]:
        print(f"  error          : {result['error']}")

    overlay = draw_debug_overlay(image, result)
    save_image(os.path.join(OUTPUT_DIR, f"{name}_debug.jpg"), overlay)

    for eye in result["eyes"]:
        crop = get_eye_crop(image, eye)
        save_image(os.path.join(OUTPUT_DIR, f"{name}_eye_{eye['position']}.jpg"), crop)

    return result


def main():
    image_files = sorted(
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        print(f"No test images found in {TEST_IMAGES_DIR}")
        return

    print(f"Running Module 1 detection on {len(image_files)} test image(s)...")

    results = {}
    for filename in image_files:
        results[filename] = run_on_image(filename)

    successes = sum(1 for r in results.values() if r["success"])
    print(f"\n{'=' * 40}")
    print(f"Summary: {successes}/{len(results)} images fully succeeded (face + eyes found)")
    print(f"Debug images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
