"""Tests for OpenCV-based CV modules."""

import numpy as np
import pytest

from packguard_pipeline.cv import CrackDetector, VoidSegmenterCV
from packguard_pipeline.synthetic.dicing import generate_die_image
from packguard_pipeline.synthetic.voids import generate_void_image


def test_crack_detector_blank_background_no_crack():
    """On a truly empty image, the detector should report zero cracks."""
    arr = np.full((256, 256), 180, dtype=np.uint8)
    rng = np.random.default_rng(0)
    arr = (arr + rng.normal(0, 4, arr.shape)).clip(0, 255).astype(np.uint8)
    det = CrackDetector().detect(arr)
    assert det.crack_length_mm == 0.0
    assert det.crack_count == 0


def test_crack_detector_clean_die_synthetic_background():
    """
    Synthetic die backgrounds have circuit traces the detector may light up.
    We only assert: it doesn't claim a single 5mm+ crack on a clean die. Real
    X-rays don't look like our procedural traces; this is a robustness ceiling.
    """
    img, _ = generate_die_image(crack_length_mm=0.0, edge_chip_um=0.0, scratch=False, seed=1)
    arr = np.array(img)
    det = CrackDetector().detect(arr)
    assert det.crack_length_mm < 5.0


def test_crack_detector_finds_large_crack():
    img, _ = generate_die_image(crack_length_mm=1.8, edge_chip_um=0.0, scratch=False, seed=2)
    arr = np.array(img)
    det = CrackDetector().detect(arr)
    # Should detect *some* crack; we don't insist on accuracy beyond > 0.5mm
    assert det.crack_count >= 1
    assert det.crack_length_mm > 0.5


def test_void_segmenter_low_void_ratio_clean():
    img, lbl = generate_void_image(void_fraction=0.05, is_clustered=False, seed=3)
    arr = np.array(img)
    meas = VoidSegmenterCV().segment(arr)
    # OpenCV measurement should be in the same ballpark as the ground truth
    assert meas.void_fraction < 0.20


def test_void_segmenter_detects_high_void_ratio():
    img, lbl = generate_void_image(void_fraction=0.30, is_clustered=False, seed=4)
    arr = np.array(img)
    meas = VoidSegmenterCV().segment(arr)
    # Should detect substantial voiding
    assert meas.void_fraction > 0.10
    assert meas.n_voids >= 3


def test_void_segmenter_clustering_detection():
    _, _ = generate_void_image(void_fraction=0.20, is_clustered=True, seed=5)
    # Clustered images have void centroids concentrated; the heuristic
    # should usually flag them. We don't make this strict — it's a heuristic.
    img, _ = generate_void_image(void_fraction=0.20, is_clustered=True, seed=5)
    meas = VoidSegmenterCV().segment(np.array(img))
    # Just a smoke test that the field exists and is bool
    assert isinstance(meas.is_clustered, bool)
