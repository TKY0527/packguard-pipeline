"""
OpenCV-based die crack detector for Checkpoint 1.

Pipeline:
  1. Grayscale + light denoising (bilateral filter preserves edges)
  2. Canny edge detection
  3. Probabilistic Hough line transform → straight cracks
  4. Filter lines by length, angle, and edge proximity (cracks usually start at edges)
  5. Sum line lengths in mm space → "worst crack length"

Why no ML: Edge-based crack detection is the textbook approach for this problem
and works extremely well on synthetic data with controlled lighting. The brief
explicitly allows "Edge detection + classifier" for cracks. Saves ML training
time and avoids torch dependency for the demo path.

Calibration: assumes the synthetic generator's `PX_PER_MM = 18.0`. Real data
calibrates from a fiducial mark or wafer-known-good standard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PX_PER_MM = 18.0  # matches synthetic.dicing


@dataclass
class CrackDetection:
    crack_length_mm: float
    crack_count: int
    longest_crack_px: int
    confidence: float
    visualization_png_bytes: bytes | None = None


class CrackDetector:
    """Stateless crack detector. Cheap to construct, safe to share."""

    def __init__(
        self,
        *,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 30,
        min_line_length_px: int = 18,    # ≈ 1mm at 18 px/mm
        max_line_gap_px: int = 4,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length_px = min_line_length_px
        self.max_line_gap_px = max_line_gap_px

    def detect(self, image: np.ndarray, *, return_visualization: bool = False) -> CrackDetection:
        """Detect cracks in a die image."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Denoise without smearing edges
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)

        # Edges
        edges = cv2.Canny(denoised, self.canny_low, self.canny_high)

        # Probabilistic Hough lines — finds disconnected segments
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=math.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length_px,
            maxLineGap=self.max_line_gap_px,
        )

        crack_segments: list[tuple[int, int, int, int, float]] = []
        if lines is not None:
            h, w = gray.shape
            edge_band = max(20, min(w, h) // 8)  # "near die edge" tolerance
            for L in lines:
                x1, y1, x2, y2 = L[0]
                length_px = math.hypot(x2 - x1, y2 - y1)
                # Filter: cracks are typically near an edge
                near_edge = (
                    min(x1, x2) < edge_band
                    or max(x1, x2) > w - edge_band
                    or min(y1, y2) < edge_band
                    or max(y1, y2) > h - edge_band
                )
                if near_edge and length_px >= self.min_line_length_px:
                    crack_segments.append((x1, y1, x2, y2, length_px))

        # Aggregate: longest segment is "the worst crack"
        longest_px = int(max((s[4] for s in crack_segments), default=0))
        longest_mm = longest_px / PX_PER_MM
        n = len(crack_segments)

        # Confidence: high when we have a clear single dominant crack,
        # lower when many noisy segments are detected.
        if n == 0:
            confidence = 0.95   # high confidence in the negative
        elif n <= 3:
            confidence = 0.92
        else:
            confidence = max(0.55, 0.92 - 0.04 * n)

        viz_bytes: bytes | None = None
        if return_visualization and lines is not None:
            viz = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for x1, y1, x2, y2, _ in crack_segments:
                cv2.line(viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ok, buf = cv2.imencode(".png", viz)
            if ok:
                viz_bytes = bytes(buf.tobytes())

        return CrackDetection(
            crack_length_mm=longest_mm,
            crack_count=n,
            longest_crack_px=longest_px,
            confidence=confidence,
            visualization_png_bytes=viz_bytes,
        )

    def detect_path(self, path: str | Path, *, return_visualization: bool = False) -> CrackDetection:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        return self.detect(img, return_visualization=return_visualization)


def detect_cracks_in_dir(dir_path: str | Path) -> dict[str, CrackDetection]:
    """Run detection on every PNG in `dir_path`. Useful for batching a lot."""
    d = Path(dir_path)
    detector = CrackDetector()
    results: dict[str, CrackDetection] = {}
    for img_path in sorted(d.glob("*.png")):
        results[img_path.name] = detector.detect_path(img_path)
    return results


__all__ = ["CrackDetector", "CrackDetection", "detect_cracks_in_dir"]
