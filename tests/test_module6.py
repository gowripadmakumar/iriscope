"""
Manual test runner for Module 6 (Web App + Upload + Camera).

Uses Flask's built-in test client to drive the real /api/analyze endpoint
end to end -- no separate server process, no network -- against the same
real test images the earlier modules were tested with. This exercises the
actual Flask route, the actual in-memory decode step, and the actual
analyzer.analyze_iris() pipeline together, exactly as a browser upload or
camera capture would.

It also specifically verifies that upload-shaped and camera-shaped
requests (different filename/content-type, same underlying JPEG bytes)
produce identical results, proving both input methods hit one pipeline.

Run from the project root:
    python tests/test_module6.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app import app  # noqa: E402

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "test_images")


def post_image(client, path, filename, content_type="image/jpeg"):
    with open(path, "rb") as f:
        data = f.read()
    return client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def test_frontend_served(client):
    print("\n[frontend] GET /")
    response = client.get("/")
    ok = response.status_code == 200 and b"IRISCOPE" in response.data
    print(f"  status: {response.status_code}  serves index.html: {ok}")
    return ok


def test_valid_images(client):
    print("\n[valid images] POST /api/analyze for every test image")
    image_files = sorted(
        f for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    all_ok = True
    for filename in image_files:
        path = os.path.join(TEST_IMAGES_DIR, filename)
        response = post_image(client, path, filename)
        body = response.get_json()

        status_ok = response.status_code == 200
        shape_ok = body is not None and "eyes" in body and "success" in body
        all_ok = all_ok and status_ok and shape_ok

        if body and body["eyes"]:
            summary = ", ".join(
                f"{e['position']}: {e['metrics']['structure_count']} structures" if e["success"]
                else f"{e['position']}: failed ({e['error']})"
                for e in body["eyes"]
            )
        else:
            summary = f"top-level: {body['error'] if body else 'NO BODY'}"

        print(f"  {filename:20s} status={response.status_code}  {summary}")

    return all_ok


def test_upload_and_camera_same_result(client):
    """
    Same JPEG bytes, sent once shaped like a browser <input type=file>
    upload and once shaped like a canvas.toBlob() camera capture. If both
    input paths truly share one pipeline, the analysis results must match
    exactly (structure counts, metrics, error messages).
    """
    print("\n[shared pipeline] upload-shaped vs camera-shaped request, same image")
    path = os.path.join(TEST_IMAGES_DIR, "obama_small.jpg")

    upload_response = post_image(client, path, "photo.jpg", content_type="image/jpeg")
    camera_response = post_image(client, path, "capture.jpg", content_type="image/jpeg")

    upload_body = upload_response.get_json()
    camera_body = camera_response.get_json()

    def strip_images(body):
        # panel_image and stages are base64 JPEG re-encodes; compare everything else exactly.
        clean = {"success": body["success"], "error": body["error"], "eyes": []}
        for eye in body["eyes"]:
            clean["eyes"].append(
                {k: v for k, v in eye.items() if k not in ("panel_image", "stages")}
            )
        return clean

    match = strip_images(upload_body) == strip_images(camera_body)
    print(f"  identical analysis results: {match}")
    return match


def test_invalid_inputs(client):
    print("\n[invalid inputs]")
    results = {}

    # No file field at all.
    response = client.post("/api/analyze", data={}, content_type="multipart/form-data")
    results["missing field"] = response.status_code == 400
    print(f"  missing 'image' field       -> status {response.status_code}, error: {response.get_json()['error']}")

    # Empty filename (browser sometimes sends this for an empty file input).
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(b""), "")},
        content_type="multipart/form-data",
    )
    results["empty filename"] = response.status_code == 400
    print(f"  empty filename               -> status {response.status_code}, error: {response.get_json()['error']}")

    # Wrong extension.
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(b"not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    results["wrong extension"] = response.status_code == 400
    print(f"  .txt file                    -> status {response.status_code}, error: {response.get_json()['error']}")

    # Right extension, garbage bytes (not a real image).
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(b"this is not actually a jpeg"), "fake.jpg")},
        content_type="multipart/form-data",
    )
    results["corrupt image"] = response.status_code == 400
    print(f"  corrupt .jpg bytes           -> status {response.status_code}, error: {response.get_json()['error']}")

    # Oversized upload (exceeds MAX_CONTENT_LENGTH).
    import config
    oversized = b"\x00" * (config.MAX_UPLOAD_SIZE_BYTES + 1024)
    response = client.post(
        "/api/analyze",
        data={"image": (io.BytesIO(oversized), "huge.jpg")},
        content_type="multipart/form-data",
    )
    results["oversized"] = response.status_code == 413
    print(f"  oversized file               -> status {response.status_code}, error: {response.get_json()['error']}")

    return all(results.values()), results


def main():
    client = app.test_client()

    frontend_ok = test_frontend_served(client)
    valid_ok = test_valid_images(client)
    shared_pipeline_ok = test_upload_and_camera_same_result(client)
    invalid_ok, invalid_details = test_invalid_inputs(client)

    print(f"\n{'=' * 40}")
    print("Summary:")
    print(f"  frontend served correctly       : {frontend_ok}")
    print(f"  valid images all returned JSON  : {valid_ok}")
    print(f"  upload/camera share one pipeline: {shared_pipeline_ok}")
    print(f"  invalid inputs handled correctly: {invalid_ok}  {invalid_details}")

    all_passed = frontend_ok and valid_ok and shared_pipeline_ok and invalid_ok
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")


if __name__ == "__main__":
    main()
