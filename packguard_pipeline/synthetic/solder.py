"""
Synthetic solder joint X-ray images for Checkpoint 5 (Reflow).

Image: a grid of solder balls (BGA-style) with controllable defects:
  - voids (small bright spots inside balls)
  - head-in-pillow (HIP — half-circle defect)
  - missing balls
  - bridging (two adjacent balls connected)

Run:
    python -m packguard_pipeline.synthetic.solder --lot LOT-2026-001
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

XRAY_SIZE_PX = 480
GRID_N = 8        # 8x8 ball grid
BALL_R_PX = 14
BALL_PITCH_PX = (XRAY_SIZE_PX - 60) // GRID_N


@dataclass
class SolderLabel:
    image_idx: int
    n_balls: int
    n_voids: int
    n_hip: int
    n_missing: int
    n_bridges: int
    overall_defect_rate: float


def _solder_background() -> Image.Image:
    arr = np.full((XRAY_SIZE_PX, XRAY_SIZE_PX), 215, dtype=np.uint8)
    noise = np.random.normal(0, 6, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 130, 240).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _grid_centers(offset: int = 30) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for r in range(GRID_N):
        for c in range(GRID_N):
            x = offset + c * BALL_PITCH_PX + BALL_PITCH_PX // 2
            y = offset + r * BALL_PITCH_PX + BALL_PITCH_PX // 2
            centers.append((x, y))
    return centers


def _draw_ball(draw: ImageDraw.ImageDraw, x: int, y: int, fill: int = 70) -> None:
    """Solder ball — dark in X-ray (denser than substrate)."""
    draw.ellipse([x - BALL_R_PX, y - BALL_R_PX, x + BALL_R_PX, y + BALL_R_PX], fill=fill)


def _draw_void_in_ball(draw: ImageDraw.ImageDraw, x: int, y: int, rng: random.Random) -> None:
    r = rng.randint(2, 5)
    dx = rng.randint(-BALL_R_PX // 2, BALL_R_PX // 2)
    dy = rng.randint(-BALL_R_PX // 2, BALL_R_PX // 2)
    draw.ellipse([x + dx - r, y + dy - r, x + dx + r, y + dy + r], fill=200)


def _draw_hip(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    """Head-in-pillow — top half is detached from bottom half."""
    # bright thin line through middle = unjoined interface
    draw.rectangle([x - BALL_R_PX, y - 1, x + BALL_R_PX, y + 1], fill=190)


def _draw_bridge(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int) -> None:
    """Solder bridge — dark line connecting two balls."""
    draw.line([(x1, y1), (x2, y2)], fill=70, width=6)


def generate_solder_image(
    *,
    void_count: int = 0,
    hip_count: int = 0,
    missing_count: int = 0,
    bridge_count: int = 0,
    seed: int = 0,
) -> tuple[Image.Image, SolderLabel]:
    rng = random.Random(seed)
    np.random.seed(seed)

    img = _solder_background()
    draw = ImageDraw.Draw(img)
    centers = _grid_centers()

    # Decide which balls are missing
    missing_indices = set(rng.sample(range(len(centers)), k=min(missing_count, len(centers))))

    # Draw all balls (skipping missing)
    for i, (x, y) in enumerate(centers):
        if i in missing_indices:
            continue
        _draw_ball(draw, x, y)

    # Voids
    void_indices = rng.sample(
        [i for i in range(len(centers)) if i not in missing_indices],
        k=min(void_count, len(centers) - len(missing_indices)),
    )
    for i in void_indices:
        x, y = centers[i]
        _draw_void_in_ball(draw, x, y, rng)

    # HIP
    hip_indices = rng.sample(
        [i for i in range(len(centers)) if i not in missing_indices],
        k=min(hip_count, len(centers) - len(missing_indices)),
    )
    for i in hip_indices:
        x, y = centers[i]
        _draw_hip(draw, x, y)

    # Bridges (between adjacent balls in same row)
    bridges_drawn = 0
    while bridges_drawn < bridge_count:
        i = rng.randint(0, len(centers) - 2)
        if i in missing_indices or (i + 1) in missing_indices:
            continue
        x1, y1 = centers[i]
        x2, y2 = centers[i + 1]
        # Same row only
        if y1 != y2:
            continue
        _draw_bridge(draw, x1, y1, x2, y2)
        bridges_drawn += 1

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    n_balls = len(centers) - len(missing_indices)
    n_defective = len(void_indices) + len(hip_indices) + len(missing_indices) + bridges_drawn
    overall_rate = n_defective / max(1, len(centers))

    label = SolderLabel(
        image_idx=seed,
        n_balls=n_balls,
        n_voids=len(void_indices),
        n_hip=len(hip_indices),
        n_missing=len(missing_indices),
        n_bridges=bridges_drawn,
        overall_defect_rate=overall_rate,
    )
    return img, label


def generate_lot_solder(
    *,
    lot_id: str,
    out_root: Path,
    count: int = 4,
    scenario: str = "clean",
    seed: int = 42,
) -> Path:
    rng = random.Random(seed)
    out_dir = out_root / lot_id
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: list[SolderLabel] = []
    for i in range(count):
        if scenario == "clean":
            v, h, m, b = rng.randint(0, 1), 0, 0, 0
        elif scenario == "debate":
            v, h, m, b = rng.randint(2, 5), rng.randint(0, 1), 0, 0
        elif scenario == "bad":
            v, h, m, b = rng.randint(8, 14), rng.randint(2, 4), rng.randint(1, 2), rng.randint(0, 1)
        else:  # early_kill / default
            v, h, m, b = 0, 0, 0, 0

        img, lbl = generate_solder_image(
            void_count=v,
            hip_count=h,
            missing_count=m,
            bridge_count=b,
            seed=seed * 1000 + i,
        )
        img.save(out_dir / f"solder_{i:03d}.png")
        labels.append(lbl)

    (out_dir / "labels.json").write_text(json.dumps([asdict(l) for l in labels], indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lot", default="LOT-2026-001")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--scenario", default="clean", choices=["clean", "debate", "bad", "early_kill"])
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "data" / "synthetic" / "solder"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = generate_lot_solder(
        lot_id=args.lot,
        out_root=Path(args.out),
        count=args.count,
        scenario=args.scenario,
        seed=args.seed,
    )
    print(f"Generated {args.count} solder X-rays for {args.lot} ({args.scenario}) -> {out}")


if __name__ == "__main__":
    main()
