"""
Module 1 — Computer Vision Foundation

Finds a face in an image, locates the eyes within it, and extracts padded
eye-region crops ready for the iris pipeline that later modules will add.

Deliberately does NOT touch the iris/pupil itself yet — that's Module 2.
"""

import cv2

import config


# Cascades are loaded once at import time and reused across calls — creating
# a CascadeClassifier per call would be wasteful.
_face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_PATH)
_eye_cascade = cv2.CascadeClassifier(config.EYE_CASCADE_PATH)


def _preprocess_for_detection(image):
    """Grayscale + histogram equalization, which helps the cascades work
    consistently across both dark and light iris/skin tones and uneven
    lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def _detect_largest_face(gray_image):
    """Runs the face cascade and returns the single largest detected face
    box (x, y, w, h), or None if nothing passed the minimum size filter."""
    height, width = gray_image.shape[:2]
    min_side = min(height, width)
    min_size = int(min_side * config.FACE_MIN_SIZE_RATIO)

    faces = _face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=config.FACE_SCALE_FACTOR,
        minNeighbors=config.FACE_MIN_NEIGHBORS,
        minSize=(min_size, min_size),
    )

    if len(faces) == 0:
        return None

    # If more than one face was detected, assume the largest is the subject.
    largest = max(faces, key=lambda box: box[2] * box[3])
    return tuple(int(v) for v in largest)


def _detect_eyes_in_face(gray_image, face_box):
    """Runs the eye cascade restricted to the upper portion of the face box.
    Returns a list of eye boxes in FULL-IMAGE coordinates."""
    fx, fy, fw, fh = face_box

    search_h = int(fh * config.EYE_SEARCH_REGION_HEIGHT_RATIO)
    face_region = gray_image[fy:fy + search_h, fx:fx + fw]

    min_size = int(fw * config.EYE_MIN_SIZE_RATIO)

    raw_eyes = _eye_cascade.detectMultiScale(
        face_region,
        scaleFactor=config.EYE_SCALE_FACTOR,
        minNeighbors=config.EYE_MIN_NEIGHBORS,
        minSize=(min_size, min_size),
    )

    # Shift coordinates from the cropped search region back to the full image.
    eyes_full_coords = [
        (int(ex + fx), int(ey + fy), int(ew), int(eh))
        for (ex, ey, ew, eh) in raw_eyes
    ]

    return _remove_duplicate_eyes(eyes_full_coords, fw)


def _remove_duplicate_eyes(eye_boxes, face_width):
    """The Haar cascade occasionally fires twice on the same eye with
    slightly different box sizes. Merge boxes whose centers are closer than
    EYE_DUPLICATE_DISTANCE_RATIO * face_width, keeping the larger box."""
    if len(eye_boxes) <= 1:
        return eye_boxes

    min_distance = face_width * config.EYE_DUPLICATE_DISTANCE_RATIO

    def center(box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    # Largest-first so we keep the bigger box when merging duplicates.
    boxes_sorted = sorted(eye_boxes, key=lambda b: b[2] * b[3], reverse=True)

    kept = []
    for box in boxes_sorted:
        cx, cy = center(box)
        is_duplicate = any(
            ((cx - center(k)[0]) ** 2 + (cy - center(k)[1]) ** 2) ** 0.5 < min_distance
            for k in kept
        )
        if not is_duplicate:
            kept.append(box)

    return kept


def _label_left_right(eye_boxes):
    """Labels eyes by their position AS SEEN IN THE IMAGE (not the subject's
    anatomical left/right). The eye further left on screen is 'left'."""
    if len(eye_boxes) < 2:
        return [{"box": box, "position": "unknown"} for box in eye_boxes]

    boxes_sorted = sorted(eye_boxes, key=lambda b: b[0])
    positions = ["left", "right"] + ["unknown"] * (len(boxes_sorted) - 2)
    return [
        {"box": box, "position": pos}
        for box, pos in zip(boxes_sorted, positions)
    ]


def _pad_box(box, image_shape, padding_ratio=config.EYE_CROP_PADDING_RATIO):
    """Expands a (x, y, w, h) box by padding_ratio on each side, clipped to
    stay inside the image bounds. Used to give downstream iris analysis some
    breathing room around the raw eye detection."""
    x, y, w, h = box
    height, width = image_shape[:2]

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    return (x1, y1, x2 - x1, y2 - y1)


def detect_eyes(image):
    """
    Main entry point for Module 1.

    Given a BGR image, returns a dict describing what was found:

    {
        "success": bool,
        "error": str or None,
        "face_box": (x, y, w, h) or None,
        "eyes": [
            {
                "position": "left" | "right" | "unknown",
                "box": (x, y, w, h),          # raw detection
                "crop_box": (x, y, w, h),     # padded, for iris analysis
            },
            ...
        ],
    }

    This function deliberately stops at eye-region extraction. Iris
    localization, masking, and normalization are handled by later modules.
    """
    gray = _preprocess_for_detection(image)

    face_box = _detect_largest_face(gray)
    if face_box is None:
        return {
            "success": False,
            "error": "I can't find a face in this image.",
            "face_box": None,
            "eyes": [],
        }

    raw_eyes = _detect_eyes_in_face(gray, face_box)
    if len(raw_eyes) == 0:
        return {
            "success": False,
            "error": "I can't find an eye in this image.",
            "face_box": face_box,
            "eyes": [],
        }

    labeled_eyes = _label_left_right(raw_eyes)
    eyes = [
        {
            "position": eye["position"],
            "box": eye["box"],
            "crop_box": _pad_box(eye["box"], image.shape),
        }
        for eye in labeled_eyes
    ]

    return {
        "success": True,
        "error": None,
        "face_box": face_box,
        "eyes": eyes,
    }


def get_eye_crop(image, eye):
    """Given one entry from detect_eyes()'s 'eyes' list, return the actual
    cropped pixel data for that eye using its padded crop_box."""
    x, y, w, h = eye["crop_box"]
    return image[y:y + h, x:x + w]


def draw_debug_overlay(image, result):
    """Draws the detected face box and eye boxes onto a copy of the image,
    using the IRISCOPE color palette. Purely for visual inspection during
    development — not part of the production pipeline output."""
    annotated = image.copy()

    if result["face_box"] is not None:
        x, y, w, h = result["face_box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), config.COLOR_FACE_BOX, 2)
        cv2.putText(annotated, "FACE", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_FACE_BOX, 2)

    for eye in result["eyes"]:
        x, y, w, h = eye["box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), config.COLOR_EYE_BOX, 2)
        cv2.putText(annotated, eye["position"].upper(), (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_EYE_BOX, 2)

        cx, cy, cw, ch = eye["crop_box"]
        cv2.rectangle(annotated, (cx, cy), (cx + cw, cy + ch),
                      config.COLOR_EYE_BOX, 1)

    return annotated
