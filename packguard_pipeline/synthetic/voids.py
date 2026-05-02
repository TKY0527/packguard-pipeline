"""
Synthetic die-attach X-ray images for Checkpoint 2 (Die Attach).

Each image:
  - White-ish background (substrate visible through die)
  - Dark blob "voids" with controllable area ratio
  - Optional clustering pattern

Ground truth (`labels.json`):
  void_fraction (float)  area of voids / area of die
  is_clustered (bool)    voids spatially concentrated
  void_centroids ([[x,y]])  pixel centers of each void

Run:
    python -m packguard_pipeline.synthetic.voids --lot LOT-2026-002 --count 12 \\
        --void-fraction 0.30 --clustered
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

XRAY_SIZE_PX = 320
DIE_BORDER_PX = 24


@dataclass
class VoidLabel:
    image_idx: int
    void_fraction: float
    is_clustered: bool
    void_centroids_px: list[tuple[int, int]]
    n_voids: int


def _xray_background(rng: random.Random) -> Image.Image:
    """Light gray background simulating X-ray penetration through substrate."""
    arr = np.full((XRAY_SIZE_PX, XRAY_SIZE_PX), 200, dtype=np.uint8)
    noise = np.random.normal(0, 8, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 100, 250).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")

    # Draw die border (slightly darker rectangle showing die outline)
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [DIE_BORDER_PX, DIE_BORDER_PX, XRAY_SIZE_PX - DIE_BORDER_PX, XRAY_SIZE_PX - DIE_BORDER_PX],
        outline=160,
        width=2,
    )
    return img


def _draw_voids(
    img: Image.Image,
    void_fraction: float,
    is_clustered: bool,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Draw dark voids onto the image, return centroid pixel coords."""
    draw = ImageDraw.Draw(img)
    die_area_px = (XRAY_SIZE_PX - 2 * DIE_BORDER_PX) ** 2
    target_void_area = void_fraction * die_area_px

    # Pick a cluster center if clustering is requested
    if is_clustered:
        cluster_x = rng.randint(DIE_BORDER_PX + 30, XRAY_SIZE_PX - DIE_BORDER_PX - 30)
        cluster_y = rng.randint(DIE_BORDER_PX + 30, XRAY_SIZE_PX - DIE_BORDER_PX - 30)
        cluster_radius = 60
    else:
        cluster_x = cluster_y = cluster_radius = None  # type: ignore

    centroids: list[tuple[int, int]] = []
    drawn_area = 0.0
    attempts = 0

    while drawn_area < target_void_area and attempts < 200:
        attempts += 1
        # Void radius — Gaussian-ish
        r_px = max(3, int(rng.gauss(8, 4)))

        if is_clustered and cluster_x is not None:
            x = int(rng.gauss(cluster_x, cluster_radius / 2))
            y = int(rng.gauss(cluster_y, cluster_radius / 2))
        else:
            x = rng.randint(DIE_BORDER_PX + r_px, XRAY_SIZE_PX - DIE_BORDER_PX - r_px)
            y = rng.randint(DIE_BORDER_PX + r_px, XRAY_SIZE_PX - DIE_BORDER_PX - r_px)

        # Bounds check
        if not (DIE_BORDER_PX + r_px <= x <= XRAY_SIZE_PX - DIE_BORDER_PX - r_px and
                DIE_BORDER_PX + r_px <= y <= XRAY_SIZE_PX - DIE_BORDER_PX - r_px):
            continue

        # Dark gray fill
        gray = rng.randint(40, 80)
        draw.ellipse([x - r_px, y - r_px, x + r_px, y + r_px], fill=gray)
        centroids.append((x, y))
        drawn_area += math.pi * r_px ** 2

    return centroids


def generate_void_image(
    *,
    void_fraction: float = 0.10,
    is_clustered: bool = False,
    seed: int = 0,
) -> tuple[Image.Image, VoidLabel]:
    """Generate one void X-ray image. Returns image + ground truth."""
    rng = random.Random(seed)
    np.random.seed(seed)
    img = _xray_background(rng)
    centroids = _draw_voids(img, void_fraction=void_fraction, is_clustered=is_clustered, rng=rng)

    # Slight blur to simulate X-ray detector PSF
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    label = VoidLabel(
        image_idx=seed,
        void_fraction=void_fraction,
        is_clustered=is_clustered,
        void_centroids_px=centroids,
        n_voids=len(centroids),
    )
    return img, label


def generate_lot_voids(
    *,
    lot_id: str,
    out_root: Path,
    count: int = 12,
    scenario: str = "clean",
    seed: int = 42,
) -> Path:
    """
    Generate void X-ray images for a lot.

    Scenarios:
      clean       — void fractions 0.05-0.10, dispersed
      early_kill  — N/A (KILL at C1 doesn't reach C2)
      debate      — void fractions 0.10-0.18, occasional clustered
      bad         — void fractions 0.25-0.40 (would KILL at C2)
    """
    rng = random.Random(seed)
    out_dir = out_root / lot_id
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: list[VoidLabel] = []
    for i in range(count):
        if scenario == "clean":
            vf = rng.uniform(0.05, 0.10)
            clustered = False
        elif scenario == "debate":
            vf = rng.uniform(0.10, 0.18)
            clustered = (i % 4 == 0)
        elif scenario == "bad":
            vf = rng.uniform(0.25, 0.40)
            clustered = True
        else:  # early_kill / fallback
            vf = rng.uniform(0.03, 0.08)
            clustered = False

        img, lbl = generate_void_image(
            void_fraction=vf,
            is_clustered=clustered,
            seed=seed * 1000 + i,
        )
        img.save(out_dir / f"xray_{i:03d}.png")
        labels.append(lbl)

    (out_dir / "labels.json").write_text(json.dumps([asdict(l) for l in labels], indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lot", default="LOT-2026-001")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--scenario", default="clean", choices=["clean", "debate", "bad", "early_kill"])
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "data" / "synthetic" / "voids"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = generate_lot_voids(
        lot_id=args.lot,
        out_root=Path(args.out),
        count=args.count,
        scenario=args.scenario,
        seed=args.seed,
    )
    print(f"Generated {args.count} void X-rays for {args.lot} ({args.scenario}) -> {out}")


if __name__ == "__main__":
    main()
