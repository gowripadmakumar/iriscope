"""
Small, general-purpose helpers shared across the IRISCOPE backend.
Nothing iris-specific lives here — just image I/O plumbing.
"""

import os
import cv2
import numpy as np

from config import MAX_IMAGE_DIMENSION


def decode_image_bytes(data):
    """
    Decode raw image bytes (e.g. an uploaded file or a browser-captured
    frame, still in memory) into a BGR NumPy array -- the same shape
    load_image() returns from disk. Used by the web app so uploaded/
    captured images never need to touch disk before analysis.

    Raises ValueError with a clear message if the bytes can't be decoded
    as an image, mirroring load_image()'s error handling below.
    """
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "Could not read this as an image. It may be corrupted or an unsupported format."
        )
    return image


def load_image(path):
    """
    Load an image from disk as a BGR NumPy array.

    Raises FileNotFoundError / ValueError with a clear message instead of
    silently returning None, so callers don't need to guess why something
    failed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at: {path}")

    image = cv2.imread(path)
    if image is None:
        raise ValueError(
            f"Could not read '{path}' as an image. "
            "It may be corrupted or an unsupported format."
        )
    return image


def resize_if_large(image, max_dimension=MAX_IMAGE_DIMENSION):
    """
    Downscale an image proportionally if either side exceeds max_dimension.

    Returns (resized_image, scale_factor). scale_factor is 1.0 if no
    resizing was needed. Keeping track of the scale lets callers map
    coordinates back to the original image if ever needed.
    """
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dimension:
        return image, 1.0

    scale_factor = max_dimension / float(longest_side)
    new_size = (int(width * scale_factor), int(height * scale_factor))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale_factor


def ensure_dir(path):
    """Create a directory (including parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def save_image(path, image):
    """Save an image to disk, creating the destination folder if needed."""
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image)
