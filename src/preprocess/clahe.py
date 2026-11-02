"""CLAHE on the lightness channel. Shared by the dataset preprocessing and the inference API, so both apply
exactly the same transformation. Only depends on OpenCV."""

import cv2


def apply_clahe(img_bgr, clip_limit, tile_grid):
    """Apply CLAHE to the L channel of a BGR image and return a BGR image."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
