"""Computer-vision modules. OpenCV-only by default; torch-based heavier models opt-in."""

from .crack_detector import CrackDetector, detect_cracks_in_dir
from .void_segmenter_cv import VoidSegmenterCV, measure_void_fraction

__all__ = [
    "CrackDetector",
    "detect_cracks_in_dir",
    "VoidSegmenterCV",
    "measure_void_fraction",
]
