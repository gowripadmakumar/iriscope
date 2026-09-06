"""
Module 3 — Normalization + Texture Analysis

Takes Module 2's segmented iris (pupil/iris circles + usable mask) and
unwraps it into a fixed-size rectangular ("polar") representation, then
enhances that representation so texture is clearly visible for later
radial-structure detection.

Deliberately does NOT detect, count, or filter any structures yet --
that's Module 4. This module only produces a clean, consistent
representation for Module 4 to work on.
"""

import cv2
import numpy as np

import config


def _build_polar_maps(center, r_inner, r_outer, angle_samples, radial_samples):
    """
    Builds the (map_x, map_y) coordinate grids used to unwrap an annular
    region of the original eye crop into a rectangular image.

    In the output, columns correspond to angle (0 to 2*pi around the iris)
    and rows correspond to radius (r_inner at the top, r_outer at the
    bottom). This means a genuine radial structure in the eye -- a texture
    line running from the pupil outward at some fixed angle -- becomes a
    roughly VERTICAL line (constant column) in the unwrapped image. That's
    the geometric property Module 4 will rely on.
    """
    cx, cy = center
    thetas = np.linspace(0, 2 * np.pi, angle_samples, endpoint=False)
    radii = np.linspace(r_inner, r_outer, radial_samples)

    theta_grid, radius_grid = np.meshgrid(thetas, radii)  # both (radial_samples, angle_samples)

    map_x = (cx + radius_grid * np.cos(theta_grid)).astype(np.float32)
    map_y = (cy + radius_grid * np.sin(theta_grid)).astype(np.float32)
    return map_x, map_y


def _unwrap(image, map_x, map_y, interpolation):
    """Applies a precomputed polar map to any single-channel image (the
    grayscale crop or a binary mask). Areas of the map that fall outside
    the source image are filled with 0 (treated as invalid/black)."""
    return cv2.remap(
        image, map_x, map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _enhance_texture(normalized_gray):
    """
    Denoise-then-contrast-boost the raw unwrapped grayscale.

    Order matters: CLAHE amplifies local contrast, which would also amplify
    pixel noise if applied first. Bilateral filtering is used instead of a
    plain Gaussian blur because it smooths flat regions while leaving real
    edges (the texture we actually care about) intact.
    """
    denoised = cv2.bilateralFilter(
        normalized_gray,
        config.NORM_BILATERAL_DIAMETER,
        config.NORM_BILATERAL_SIGMA_COLOR,
        config.NORM_BILATERAL_SIGMA_SPACE,
    )

    clahe = cv2.createCLAHE(
        clipLimit=config.NORM_CLAHE_CLIP_LIMIT,
        tileGridSize=config.NORM_CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(denoised)


def _compute_edge_map(enhanced, normalized_mask):
    """
    Computes a vertical-edge-emphasis map from the enhanced texture.

    A radial iris structure appears as a roughly vertical line in the
    unwrapped image (see _build_polar_maps), so a horizontal gradient
    (Scharr in the x/column direction) responds strongly to genuine radial
    structures while mostly ignoring horizontal texture (eyelash shadows,
    horizontal banding), which run the other way.

    This is texture *preparation*, not detection: no thresholding, counting,
    or filtering of individual structures happens here -- that's Module 4.
    Masked-out pixels (eyelid/reflection, from Module 2) are zeroed so
    Module 4 doesn't have to re-check the mask itself.
    """
    scharr_x = cv2.Scharr(enhanced, cv2.CV_32F, dx=1, dy=0)
    magnitude = np.abs(scharr_x)

    peak = magnitude.max()
    if peak > 0:
        magnitude = (magnitude / peak) * 255.0
    edge_map = magnitude.astype(np.uint8)

    return cv2.bitwise_and(edge_map, edge_map, mask=normalized_mask)


def normalize_iris(eye_crop_bgr, segmentation_result):
    """
    Main entry point for Module 3.

    Given the original BGR eye crop and Module 2's segment_iris() output,
    returns a dict describing the normalized, texture-enhanced iris.

    {
        "success": bool,
        "error": str or None,
        "normalized_gray": raw unwrapped grayscale, shape (RADIAL, ANGLE),
        "normalized_mask": unwrapped usable-region mask, same shape (0/255),
        "normalized_enhanced": denoised + contrast-enhanced version,
        "normalized_edges": vertical-edge-emphasis map, masked, same shape,
        "valid_ratio": float,  # fraction of the normalized rep that's usable
    }

    Requires segmentation_result["success"] to be True -- this module has
    nothing reliable to normalize otherwise.
    """
    if not segmentation_result.get("success"):
        return {
            "success": False,
            "error": "No usable iris segmentation to normalize.",
            "normalized_gray": None,
            "normalized_mask": None,
            "normalized_enhanced": None,
            "normalized_edges": None,
            "valid_ratio": None,
        }

    cx, cy = segmentation_result["iris_center"]
    pupil_r = segmentation_result["pupil"][2]
    iris_r = segmentation_result["iris_radius"]
    usable_mask = segmentation_result["masks"]["usable_mask"]

    span = iris_r - pupil_r
    r_inner = pupil_r + span * config.NORM_INNER_MARGIN_RATIO
    r_outer = pupil_r + span * config.NORM_OUTER_MARGIN_RATIO

    if r_outer <= r_inner:
        return {
            "success": False,
            "error": "Iris ring too narrow to normalize reliably.",
            "normalized_gray": None,
            "normalized_mask": None,
            "normalized_enhanced": None,
            "normalized_edges": None,
            "valid_ratio": None,
        }

    gray = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2GRAY)

    map_x, map_y = _build_polar_maps(
        (cx, cy), r_inner, r_outer,
        config.NORM_ANGLE_SAMPLES, config.NORM_RADIAL_SAMPLES,
    )

    normalized_gray = _unwrap(gray, map_x, map_y, cv2.INTER_LINEAR)

    normalized_mask_raw = _unwrap(usable_mask, map_x, map_y, cv2.INTER_NEAREST)
    _, normalized_mask = cv2.threshold(
        normalized_mask_raw, config.NORM_MASK_VALID_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    valid_ratio = float(np.count_nonzero(normalized_mask)) / normalized_mask.size

    if valid_ratio < config.NORM_MIN_VALID_RATIO:
        return {
            "success": False,
            "error": "Not enough usable iris texture survived normalization.",
            "normalized_gray": normalized_gray,
            "normalized_mask": normalized_mask,
            "normalized_enhanced": None,
            "normalized_edges": None,
            "valid_ratio": round(valid_ratio, 3),
        }

    normalized_enhanced = _enhance_texture(normalized_gray)
    normalized_edges = _compute_edge_map(normalized_enhanced, normalized_mask)

    return {
        "success": True,
        "error": None,
        "normalized_gray": normalized_gray,
        "normalized_mask": normalized_mask,
        "normalized_enhanced": normalized_enhanced,
        "normalized_edges": normalized_edges,
        "valid_ratio": round(valid_ratio, 3),
    }
