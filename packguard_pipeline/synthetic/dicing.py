"""
Generate synthetic AOI top-down die images for Checkpoint 1 (Dicing).

Why synthetic: we don't have access to real Micron data. The brief explicitly
allows synthetic data so long as it looks convincing — judges will see these
in the demo. Ground truth (crack length, edge chip width) is known by
construction, which lets Person 2 validate CV models against it.

Output:
  data/synthetic/dicing/<lot_id>/die_<idx>.png
  data/synthetic/dicing/<lot_id>/labels.json     # ground truth

Run:
    python -m packguard_pipeline.synthetic.dicing --lot LOT-2026-002 --count 24
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

DIE_SIZE_PX = 256
EDGE_BORDER_PX = 18  # 1mm-equivalent for our scale (1px ≈ 0.055mm)
PX_PER_MM = 18.0


@dataclass
class DieLabel:
    die_idx: int
    crack_length_mm: float
    crack_angle_deg: Optional[float]
    edge_chip_um: float  # max edge chip in micrometers
    has_scratch: bool


def _random_circuit_pattern(rng: random.Random) -> Image.Image:
    """A grayscale die with a faint circuit pattern. Looks convincing enough for demo."""
    img = Image.new("L", (DIE_SIZE_PX, DIE_SIZE_PX), color=180)
    draw = ImageDraw.Draw(img)

    # Concentric squares — vague suggestion of bond-pad ring
    for i in range(0, EDGE_BORDER_PX, 4):
        draw.rectangle(
            [i, i, DIE_SIZE_PX - i, DIE_SIZE_PX - i],
            outline=160 + rng.randint(-10, 10),
            width=1,
        )

    # Random horizontal/vertical "trace" lines
    for _ in range(rng.randint(40, 70)):
        x0 = rng.randint(EDGE_BORDER_PX, DIE_SIZE_PX - EDGE_BORDER_PX)
        y0 = rng.randint(EDGE_BORDER_PX, DIE_SIZE_PX - EDGE_BORDER_PX)
        length = rng.randint(8, 40)
        if rng.random() < 0.5:
            draw.line([(x0, y0), (x0 + length, y0)], fill=130, width=1)
        else:
            draw.line([(x0, y0), (x0, y0 + length)], fill=130, width=1)

    # Fine speckle noise so the image doesn't look painted
    arr = np.array(img, dtype=np.int16)
    noise = np.random.normal(0, 6, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _draw_crack(img: Image.Image, length_mm: float, rng: random.Random) -> float:
    """Draw a crack as a bright zigzag line. Returns angle in degrees."""
    draw = ImageDraw.Draw(img)
    angle_deg = rng.uniform(-90, 90)
    angle = math.radians(angle_deg)
    length_px = int(length_mm * PX_PER_MM)

    # Start near an edge so cracks look like they propagated inward
    edge = rng.choice(["top", "bottom", "left", "right"])
    if edge == "top":
        x0, y0 = rng.randint(40, DIE_SIZE_PX - 40), rng.randint(2, 10)
    elif edge == "bottom":
        x0, y0 = rng.randint(40, DIE_SIZE_PX - 40), rng.randint(DIE_SIZE_PX - 10, DIE_SIZE_PX - 2)
    elif edge == "left":
        x0, y0 = rng.randint(2, 10), rng.randint(40, DIE_SIZE_PX - 40)
    else:
        x0, y0 = rng.randint(DIE_SIZE_PX - 10, DIE_SIZE_PX - 2), rng.randint(40, DIE_SIZE_PX - 40)

    # Multi-segment zigzag for realism
    seg_count = max(2, length_px // 20)
    seg_len = length_px / seg_count
    px, py = float(x0), float(y0)
    for i in range(seg_count):
        jitter = math.radians(rng.uniform(-12, 12))
        nx = px + seg_len * math.cos(angle + jitter)
        ny = py + seg_len * math.sin(angle + jitter)
        draw.line([(px, py), (nx, ny)], fill=240, width=1)
        px, py = nx, ny

    return angle_deg


def _draw_edge_chip(img: Image.Image, width_um: float, rng: random.Random) -> None:
    """Draw a chip notch on a random edge."""
    draw = ImageDraw.Draw(img)
    width_px = max(2, int(width_um / 1000.0 * PX_PER_MM))

    edge = rng.choice(["top", "bottom", "left", "right"])
    base = rng.randint(40, DIE_SIZE_PX - 40)
    depth = max(2, int(width_px * 0.6))

    if edge == "top":
        bbox = [base, 0, base + width_px, depth]
    elif edge == "bottom":
        bbox = [base, DIE_SIZE_PX - depth, base + width_px, DIE_SIZE_PX]
    elif edge == "left":
        bbox = [0, base, depth, base + width_px]
    else:
        bbox = [DIE_SIZE_PX - depth, base, DIE_SIZE_PX, base + width_px]

    draw.rectangle(bbox, fill=70)  # dark chip


def _draw_scratch(img: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0 = rng.randint(20, DIE_SIZE_PX - 20), rng.randint(20, DIE_SIZE_PX - 20)
    angle = math.radians(rng.uniform(0, 180))
    length = rng.randint(40, 90)
    x1 = x0 + length * math.cos(angle)
    y1 = y0 + length * math.sin(angle)
    draw.line([(x0, y0), (x1, y1)], fill=210, width=1)


def generate_die_image(
    *,
    crack_length_mm: float = 0.0,
    edge_chip_um: float = 0.0,
    scratch: bool = False,
    seed: int = 0,
) -> tuple[Image.Image, DieLabel]:
    """Generate one die image with controllable defects. Returns image + ground truth."""
    rng = random.Random(seed)
    np.random.seed(seed)
    img = _random_circuit_pattern(rng)

    angle: Optional[float] = None
    if crack_length_mm > 0.0:
        angle = _draw_crack(img, crack_length_mm, rng)
    if edge_chip_um > 0.0:
        _draw_edge_chip(img, edge_chip_um, rng)
    if scratch:
        _draw_scratch(img, rng)

    label = DieLabel(
        die_idx=seed,
        crack_length_mm=crack_length_mm,
        crack_angle_deg=angle,
        edge_chip_um=edge_chip_um,
        has_scratch=scratch,
    )
    return img, label


def generate_lot(
    *,
    lot_id: str,
    out_root: Path,
    count: int = 24,
    scenario: str = "clean",
    seed: int = 42,
) -> Path:
    """
    Generate a folder of die images representing one lot.

    scenarios:
      clean      — all clean dies
      early_kill — at least one die has a 1.8mm crack (the killer demo)
      debate     — borderline 0.6mm cracks scattered, no kills
    """
    rng = random.Random(seed)
    out_dir = out_root / lot_id
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: list[DieLabel] = []
    for i in range(count):
        if scenario == "early_kill" and i == 0:
            crack, chip, scratch = 1.8, 0.0, False
        elif scenario == "early_kill" and i in (3, 7):
            crack, chip, scratch = 0.6, 0.0, False
        elif scenario == "debate" and i % 5 == 0:
            crack, chip, scratch = 0.6, 25.0, False
        else:
            crack = 0.0 if rng.random() > 0.15 else rng.uniform(0.05, 0.3)
            chip = 0.0 if rng.random() > 0.20 else rng.uniform(5.0, 30.0)
            scratch = rng.random() < 0.05

        img, label = generate_die_image(
            crack_length_mm=crack,
            edge_chip_um=chip,
            scratch=scratch,
            seed=seed * 1000 + i,
        )
        img.save(out_dir / f"die_{i:03d}.png")
        labels.append(label)

    (out_dir / "labels.json").write_text(
        json.dumps([asdict(lbl) for lbl in labels], indent=2)
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lot", default="LOT-2026-002")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--scenario", default="early_kill", choices=["clean", "early_kill", "debate"])
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "data" / "synthetic" / "dicing"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = generate_lot(
        lot_id=args.lot,
        out_root=Path(args.out),
        count=args.count,
        scenario=args.scenario,
        seed=args.seed,
    )
    print(f"Generated {args.count} die images for {args.lot} ({args.scenario}) -> {out_dir}")


if __name__ == "__main__":
    main()
