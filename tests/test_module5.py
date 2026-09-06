"""
Manual test runner for Module 5 (Results + Visualization).

Runs the complete IRISCOPE pipeline (Modules 1-4, via analyzer.analyze_iris_file)
on every image in data/test_images, saves a labeled multi-stage panel per
eye (or per image, for a top-level failure), and prints a summary so the
final result can be inspected visually rather than trusted from the numbers
alone.

Run from the project root:
    python tests/test_module5.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from utils import save_image
from analyzer import analyze_iris_file, build_stage_panel

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "test_images")
OUTPUT_DIR = os.path.join(TEST_IMAGES_DIR, "output_module5")


def run_on_image(filename):
    path = os.path.join(TEST_IMAGES_DIR, filename)
    name = os.path.splitext(filename)[0]

    print(f"\n{filename}")
    result = analyze_iris_file(path)

    if not result["eyes"]:
        # Either a top-level failure (no face/eye/unreadable file), or a
        # genuinely eyeless detection -- either way, there's one panel to save.
        print(f"  {result['error']}")
        panel = build_stage_panel({"stages": {}}, top_level_stages=result["stages"])
        if panel is not None:
            save_image(os.path.join(OUTPUT_DIR, f"{name}_result.jpg"), panel)
        return [result["success"]]

    outcomes = []
    for eye_result in result["eyes"]:
        pos = eye_result["position"]
        if eye_result["success"]:
            m = eye_result["metrics"]
            print(
                f"  {pos:5s} eye: {m['structure_count']} detectable radial structures "
                f"(quality: {m['detection_quality']:.0%}, usable area: {m['usable_area_ratio']:.0%}, "
                f"density: {m['structure_density']:.1f}/100deg)"
            )
        else:
            print(f"  {pos:5s} eye: FAILED at '{eye_result['failed_stage']}' -- {eye_result['error']}")

        panel = build_stage_panel(eye_result, top_level_stages=result["stages"])
        save_image(os.path.join(OUTPUT_DIR, f"{name}_{pos}_result.jpg"), panel)
        outcomes.append(eye_result["success"])

    return outcomes


def main():
    image_files = sorted(
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    print(f"Running Module 5 full-pipeline analysis on {len(image_files)} test image(s)...")

    total = 0
    successful = 0
    for filename in image_files:
        for outcome in run_on_image(filename):
            total += 1
            if outcome:
                successful += 1

    print(f"\n{'=' * 40}")
    print(f"Summary: {successful}/{total} eyes/images produced a full analysis result")
    print(f"Result panels saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
