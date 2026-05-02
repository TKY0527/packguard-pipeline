"""
Claude Vision integration for low-confidence CV escalation.

Per brief Section 4 (Checkpoint 1) and Person 2's spec:
> When the deterministic CV model has confidence < 80%, package the image plus
> context and send it to Claude Vision API for a second opinion.

This module is the "vision-AI integration glue". It is **opt-in**:
- If `ANTHROPIC_API_KEY` is not set in the environment → returns a graceful
  fallback `VisionDeferred` that the checkpoint can record without crashing.
- If the SDK is missing → same fallback.

Cost-control:
- Only call when CV confidence < threshold (default 0.80).
- Cap image size at 1024×1024 to keep tokens bounded.
- Cache by image hash so re-runs on the same lot don't double-bill.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image


CLAUDE_MODEL = "claude-opus-4-7"
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
MAX_IMG_SIDE_PX = 1024


@dataclass
class VisionVerdict:
    """Result of a successful Claude Vision second-opinion call."""
    detected_class: str
    confidence: float
    rationale: str  # Claude's free-form explanation
    raw_response: str
    cached: bool = False


@dataclass
class VisionDeferred:
    """Returned when Claude was not invoked (no key, no SDK, or above threshold)."""
    reason: str  # "no_api_key" | "above_threshold" | "sdk_missing"


def _hash_image_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _resize_for_api(image_path: Path) -> bytes:
    """Resize image to ≤ MAX_IMG_SIDE_PX and return PNG bytes."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    scale = min(1.0, MAX_IMG_SIDE_PX / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".vision_cache"


def _cache_get(key: str) -> Optional[VisionVerdict]:
    if not _CACHE_DIR.exists():
        return None
    p = _CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["cached"] = True
    return VisionVerdict(**d)


def _cache_put(key: str, v: VisionVerdict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _CACHE_DIR / f"{key}.json"
    payload = {**v.__dict__}
    payload.pop("cached", None)
    p.write_text(json.dumps(payload))


def get_second_opinion(
    *,
    image_path: str | Path,
    cv_confidence: float,
    question: str,
    context: dict | None = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> VisionVerdict | VisionDeferred:
    """
    Ask Claude Vision for a second opinion on a low-confidence CV result.

    Args:
        image_path: Path to the image file.
        cv_confidence: Confidence reported by the deterministic CV model.
                       If >= threshold, we skip the call (saves money).
        question: Concrete question for Claude (e.g., "Is this a crack or dust?").
        context: Optional supporting numbers (e.g., {"crack_length_mm": 1.8}).
        threshold: Below this CV confidence, escalate.
    """
    if cv_confidence >= threshold:
        return VisionDeferred(reason="above_threshold")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return VisionDeferred(reason="no_api_key")

    try:
        import anthropic  # noqa: F401
    except ImportError:
        return VisionDeferred(reason="sdk_missing")

    image_path = Path(image_path)
    if not image_path.exists():
        return VisionDeferred(reason=f"image_not_found:{image_path}")

    img_bytes = _resize_for_api(image_path)
    cache_key = _hash_image_bytes(img_bytes) + "_" + hashlib.sha256(question.encode()).hexdigest()[:8]
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Lazy import so module loads even if anthropic absent
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    b64 = base64.standard_b64encode(img_bytes).decode("ascii")

    system_prompt = (
        "You are a senior packaging-reliability engineer reviewing a low-confidence "
        "computer-vision detection on a semiconductor package image. Answer concisely. "
        "Reply ONLY with a JSON object of the form: "
        '{"detected_class": "<one word>", "confidence": <0..1>, "rationale": "<1-2 sentences>"}.'
    )
    ctx_str = json.dumps(context) if context else "{}"
    user_text = (
        f"CV model confidence: {cv_confidence:.2f}\n"
        f"Numerical context: {ctx_str}\n"
        f"Question: {question}\n"
    )

    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )

    raw = msg.content[0].text if msg.content else ""

    # Best-effort JSON parse — Claude usually complies given the system prompt
    try:
        # Strip code fences if present
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s[s.find("{"): s.rfind("}") + 1]
        data = json.loads(s)
        verdict = VisionVerdict(
            detected_class=str(data.get("detected_class", "unknown")),
            confidence=float(data.get("confidence", 0.5)),
            rationale=str(data.get("rationale", "")),
            raw_response=raw,
        )
    except (json.JSONDecodeError, ValueError):
        verdict = VisionVerdict(
            detected_class="unknown",
            confidence=0.5,
            rationale=raw[:300],
            raw_response=raw,
        )

    _cache_put(cache_key, verdict)
    return verdict


__all__ = [
    "VisionVerdict",
    "VisionDeferred",
    "get_second_opinion",
    "DEFAULT_CONFIDENCE_THRESHOLD",
]
