"""
Tests for the Claude Vision wrapper. We DO NOT make real API calls — we only
verify the cost-control / no-key fallback behaviour that runs during normal
test execution and during demos without `ANTHROPIC_API_KEY` set.
"""

import os

from packguard_pipeline.claude_vision import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    VisionDeferred,
    get_second_opinion,
)


def test_above_threshold_does_not_call_api(tmp_path):
    """High CV confidence → no API call (saves money)."""
    img = tmp_path / "any.png"
    img.write_bytes(b"fake")
    out = get_second_opinion(
        image_path=img,
        cv_confidence=0.95,  # well above threshold
        question="dust or chip?",
    )
    assert isinstance(out, VisionDeferred)
    assert out.reason == "above_threshold"


def test_no_api_key_returns_deferred(tmp_path, monkeypatch):
    """Below threshold but no API key → graceful deferral, not crash."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    img = tmp_path / "any.png"
    img.write_bytes(b"fake")
    out = get_second_opinion(
        image_path=img,
        cv_confidence=0.4,
        question="dust or chip?",
    )
    assert isinstance(out, VisionDeferred)
    assert out.reason == "no_api_key"


def test_threshold_constant_is_sensible():
    assert 0.0 < DEFAULT_CONFIDENCE_THRESHOLD < 1.0
