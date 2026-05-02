"""
OpenCV-based void segmenter for Checkpoint 2 (Die Attach).

Pipeline:
  1. Grayscale + Otsu threshold inside the die border
  2. Morphological cleanup (open + close)
  3. Connected-component labeling → individual voids
  4. Void area / die area = void_fraction
  5. Spatial clustering test (DBSCAN-style on centroids) → is_clustered

Limits:
  - Assumes the synthetic generator's die border at DIE_BORDER_PX = 24
  - Real X-rays would need a die-edge detection preprocess

The U-Net version in `void_segmenter_unet.py` (torch-based) is more robust for
real images. This file is the fast, no-ML default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DIE_BORDER_PX = 24  # mirror of synthetic.voids


@dataclass
class VoidMeasurement:
    void_fraction: float
    is_clustered: bool
    n_voids: int
    largest_void_px: int
    confidence: float
    void_centroids_px: list[tuple[int, int]]


class VoidSegmenterCV:
    """OpenCV threshold + connected components void segmenter."""

    def __init__(self, *, min_void_px: int = 9) -> None:
        self.min_void_px = min_void_px

    def segment(self, image: np.ndarray) -> VoidMeasurement:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape
        # Crop to die area (assumes border known)
        die = gray[DIE_BORDER_PX : h - DIE_BORDER_PX, DIE_BORDER_PX : w - DIE_BORDER_PX]
        die_area_px = die.size

        # Otsu — voids are darker than background
        _, binary = cv2.threshold(die, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological cleanup — remove noise, fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        void_area = 0
        kept_centroids: list[tuple[int, int]] = []
        largest_void_px = 0

        # Skip background (label 0)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_void_px:
                continue
            void_area += area
            cx, cy = centroids[i]
            kept_centroids.append((int(cx), int(cy)))
            if area > largest_void_px:
                largest_void_px = int(area)

        void_fraction = void_area / die_area_px
        is_clustered = self._test_clustering(kept_centroids, die.shape)

        # Confidence: higher when clearly above/below typical thresholds, lower in mid-band
        if void_fraction < 0.05 or void_fraction > 0.30:
            confidence = 0.95
        else:
            confidence = max(0.70, 0.95 - 5 * abs(void_fraction - 0.15))

        return VoidMeasurement(
            void_fraction=float(void_fraction),
            is_clustered=is_clustered,
            n_voids=len(kept_centroids),
            largest_void_px=largest_void_px,
            confidence=float(confidence),
            void_centroids_px=kept_centroids,
        )

    @staticmethod
    def _test_clustering(centroids: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
        """Heuristic: are the void centroids concentrated in < ~25% of die area?"""
        if len(centroids) < 3:
            return False
        xs = np.array([c[0] for c in centroids])
        ys = np.array([c[1] for c in centroids])
        std_x = float(np.std(xs))
        std_y = float(np.std(ys))
        h, w = shape
        # If both stds are < ~20% of die size, voids are clustered
        return std_x < 0.20 * w and std_y < 0.20 * h

    def segment_path(self, path: str | Path) -> VoidMeasurement:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        return self.segment(img)


def measure_void_fraction(image: np.ndarray | str | Path) -> VoidMeasurement:
    """Convenience wrapper — accepts ndarray or path."""
    seg = VoidSegmenterCV()
    if isinstance(image, (str, Path)):
        return seg.segment_path(image)
    return seg.segment(image)


__all__ = ["VoidSegmenterCV", "VoidMeasurement", "measure_void_fraction"]
