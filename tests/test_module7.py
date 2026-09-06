"""
Manual test runner for Module 7 (Design + IRIS MATCH).

Module 7 is mostly a frontend redesign (no automated test replaces looking
at it -- see DEVELOPMENT_STATE.md for what was visually verified with a
headless browser during development). What IS fully testable here is the
new backend piece: /api/match, via the same Flask test-client approach
test_module6.py uses for /api/analyze.

This does NOT re-test Modules 1-6 -- test_module1.py through
test_module6.py already do that, and still pass unmodified after Module 7
(see DEVELOPMENT_STATE.md). This file only exercises what Module 7 added.

Run from the project root:
    python tests/test_module7.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app import app  # noqa: E402

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "test_images")

OBAMA_SMALL = os.path.join(TEST_IMAGES_DIR, "obama_small.jpg")
OBAMA = os.path.join(TEST_IMAGES_DIR, "obama.jpg")
FRUITS = os.path.join(TEST_IMAGES_DIR, "fruits.jpg")  # no face -- known Module 1 failure case


def post_match(client, path_a, path_b, filename_a="a.jpg", filename_b="b.jpg"):
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        data_a, data_b = fa.read(), fb.read()
    return client.post(
        "/api/match",
        data={
            "image_a": (io.BytesIO(data_a), filename_a),
            "image_b": (io.BytesIO(data_b), filename_b),
        },
        content_type="multipart/form-data",
    )


def test_valid_match(client):
    print("\n[valid match] two real eye photos")
    response = post_match(client, OBAMA_SMALL, OBAMA)
    body = response.get_json()

    ok = bool(
        response.status_code == 200
        and body.get("success") is True
        and body.get("scores") is not None
        and body.get("verdict")
        and body.get("eye_a", {}).get("card_image")
        and body.get("eye_b", {}).get("card_image")
    )
    print(f"  status={response.status_code}  success={body.get('success')}")
    print(f"  scores={body.get('scores')}")
    print(f"  verdict={body.get('verdict')!r}")
    return ok


def test_deterministic(client):
    print("\n[determinism] same pair of photos run twice")
    r1 = post_match(client, OBAMA_SMALL, OBAMA).get_json()
    r2 = post_match(client, OBAMA_SMALL, OBAMA).get_json()
    same = r1.get("scores") == r2.get("scores") and r1.get("verdict") == r2.get("verdict")
    print(f"  run 1 scores: {r1.get('scores')}")
    print(f"  run 2 scores: {r2.get('scores')}")
    print(f"  identical: {same}")
    return same


def test_self_match(client):
    print("\n[self-match] identical photo on both sides should score 100 everywhere")
    body = post_match(client, OBAMA_SMALL, OBAMA_SMALL).get_json()
    scores = body.get("scores") or {}
    all_100 = body.get("success") and all(v == 100 for v in scores.values())
    print(f"  scores: {scores}")
    return all_100


def test_no_face_failure(client):
    print("\n[failure case] one photo has no detectable face")
    body = post_match(client, FRUITS, OBAMA_SMALL).get_json()
    ok = body.get("success") is False and bool(body.get("error"))
    print(f"  success={body.get('success')}  error={body.get('error')!r}")
    return ok


def test_invalid_inputs(client):
    print("\n[invalid inputs]")
    results = {}

    # Missing the second image entirely.
    with open(OBAMA_SMALL, "rb") as fa:
        response = client.post(
            "/api/match",
            data={"image_a": (fa, "a.jpg")},
            content_type="multipart/form-data",
        )
    results["missing image_b"] = response.status_code == 400
    print(f"  missing image_b       -> status {response.status_code}, error: {response.get_json()['error']}")

    # Wrong extension on one side.
    with open(OBAMA_SMALL, "rb") as fa, open(OBAMA, "rb") as fb:
        response = client.post(
            "/api/match",
            data={"image_a": (fa, "notes.txt"), "image_b": (fb, "b.jpg")},
            content_type="multipart/form-data",
        )
    results["wrong extension"] = response.status_code == 400
    print(f"  wrong extension (A)   -> status {response.status_code}, error: {response.get_json()['error']}")

    # Corrupt bytes with a valid-looking extension.
    with open(OBAMA_SMALL, "rb") as fb:
        response = client.post(
            "/api/match",
            data={"image_a": (io.BytesIO(b"not actually a jpeg"), "fake.jpg"), "image_b": (fb, "b.jpg")},
            content_type="multipart/form-data",
        )
    results["corrupt image"] = response.status_code == 400
    print(f"  corrupt .jpg bytes    -> status {response.status_code}, error: {response.get_json()['error']}")

    return all(results.values()), results


def test_existing_endpoints_unaffected(client):
    """
    Module 7 touched app.py (added a route) and the frontend files. This
    is a quick guard that /api/analyze and the frontend are still served
    exactly as test_module6.py already verifies in full -- not a
    replacement for running that file too.
    """
    print("\n[regression check] /api/analyze and frontend still served")
    frontend = client.get("/")
    analyze_response = client.post(
        "/api/analyze",
        data={"image": (open(OBAMA_SMALL, "rb"), "a.jpg")},
        content_type="multipart/form-data",
    )
    ok = (
        frontend.status_code == 200 and b"IRISCOPE" in frontend.data
        and analyze_response.status_code == 200
        and analyze_response.get_json().get("success") is True
    )
    print(f"  GET / -> {frontend.status_code}   POST /api/analyze -> {analyze_response.status_code}")
    return ok


def main():
    client = app.test_client()

    valid_ok = test_valid_match(client)
    deterministic_ok = test_deterministic(client)
    self_match_ok = test_self_match(client)
    no_face_ok = test_no_face_failure(client)
    invalid_ok, invalid_details = test_invalid_inputs(client)
    regression_ok = test_existing_endpoints_unaffected(client)

    print(f"\n{'=' * 40}")
    print("Summary:")
    print(f"  valid match returns full result : {valid_ok}")
    print(f"  scoring is deterministic         : {deterministic_ok}")
    print(f"  self-match scores 100 everywhere : {self_match_ok}")
    print(f"  no-face photo fails honestly     : {no_face_ok}")
    print(f"  invalid inputs handled correctly : {invalid_ok}  {invalid_details}")
    print(f"  /api/analyze + frontend unaffected: {regression_ok}")

    all_passed = valid_ok and deterministic_ok and self_match_ok and no_face_ok and invalid_ok and regression_ok
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    print("\nNote: this file does not re-verify Modules 1-6 -- run test_module1.py")
    print("through test_module6.py separately to confirm those (they still pass).")


if __name__ == "__main__":
    main()
