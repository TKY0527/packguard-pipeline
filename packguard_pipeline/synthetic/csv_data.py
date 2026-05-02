"""
Synthetic CSV/log files used at multiple checkpoints.

- reflow_profile.csv  — temperature/time trace from oven thermocouple (Checkpoint 5)
- bond_force.csv      — pull/shear force per sampled bond (Checkpoint 3)
- burn_in.csv         — chip-level burn-in time-to-failure (Checkpoint 6)

Profiles loosely follow JEDEC J-STD-020 published reflow curves.

Run:
    python -m packguard_pipeline.synthetic.csv_data --lot LOT-2026-001
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


# ---------- Reflow profile ----------

def reflow_profile(
    *,
    duration_s: float = 300.0,
    sample_hz: float = 5.0,
    peak_temp_C: float = 245.0,
    soak_temp_C: float = 150.0,
    ambient_temp_C: float = 25.0,
    max_dT_dt_C_per_s: float = 3.0,  # spec ramp limit
    overshoot: bool = False,         # True for a "bad" lot
) -> list[tuple[float, float]]:
    """Return [(time_s, temperature_C), ...] for a typical Pb-free SAC305 reflow."""
    n = int(duration_s * sample_hz)
    points: list[tuple[float, float]] = []
    rng = random.Random(0)
    for i in range(n):
        t = i / sample_hz
        # Piecewise: ambient → soak → peak → cool
        if t < 60:                         # preheat ramp
            T = ambient_temp_C + (soak_temp_C - ambient_temp_C) * (t / 60)
        elif t < 130:                      # soak
            T = soak_temp_C + 5 * math.sin(t / 5)
        elif t < 180:                      # ramp to peak
            T = soak_temp_C + (peak_temp_C - soak_temp_C) * ((t - 130) / 50)
            if overshoot:
                T += 8  # ~253°C peak — too high
        elif t < 200:                      # peak (time above liquidus)
            T = peak_temp_C + 2 * math.cos(t / 3)
            if overshoot:
                T += 6
        else:                              # cool (limited by max_dT_dt)
            cool_rate = max_dT_dt_C_per_s
            if overshoot:
                cool_rate = max_dT_dt_C_per_s * 1.5  # too fast → thermal shock
            T = peak_temp_C - cool_rate * (t - 200)
            T = max(T, ambient_temp_C)
        T += rng.gauss(0, 0.4)             # thermocouple noise
        points.append((round(t, 2), round(T, 2)))
    return points


def write_reflow_csv(path: Path, *, scenario: str = "clean") -> None:
    overshoot = scenario in ("bad",)
    pts = reflow_profile(overshoot=overshoot)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "temperature_C"])
        w.writerows(pts)


# ---------- Bond force log ----------

def bond_force_samples(
    *,
    n_samples: int = 80,
    target_grams: float = 12.0,
    sigma_grams: float = 0.6,
    drift: float = 0.0,
) -> list[tuple[int, str, float]]:
    """Return [(sample_idx, type, force_grams)]. type ∈ {pull, shear}."""
    rng = random.Random(42)
    rows: list[tuple[int, str, float]] = []
    for i in range(n_samples):
        kind = "pull" if i % 2 == 0 else "shear"
        # Drift simulates a process going out of control — used in 'debate'.
        # Linear drift across the run to produce a Cpk-failing distribution.
        mean = target_grams + drift * (i / n_samples)
        f = rng.gauss(mean, sigma_grams)
        if kind == "shear":
            f *= 1.4  # shear stronger than pull
        rows.append((i, kind, round(f, 2)))
    return rows


def write_bond_force_csv(path: Path, *, scenario: str = "clean") -> None:
    if scenario == "debate":
        drift = -1.5    # downward drift → Cpk falls
    elif scenario == "bad":
        drift = -3.0
    else:
        drift = 0.0
    rows = bond_force_samples(drift=drift)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "test_type", "force_grams"])
        w.writerows(rows)


# ---------- Burn-in time-to-failure ----------

def burn_in_ttf(
    *,
    n_chips: int = 200,
    median_ttf_h: float = 7000.0,
    beta: float = 1.5,
    seed: int = 7,
) -> list[tuple[int, float, bool]]:
    """
    Simulate Weibull burn-in TTFs.
    Returns [(chip_idx, ttf_hours, censored)]. Censored=True if chip didn't fail.
    """
    rng = random.Random(seed)
    eta = median_ttf_h / (math.log(2) ** (1.0 / beta))
    ttf: list[tuple[int, float, bool]] = []
    test_duration = 8000.0  # hours
    for i in range(n_chips):
        # Weibull sample: u ~ U(0,1); t = eta * (-ln(1-u))^(1/beta)
        u = rng.random()
        t = eta * ((-math.log(1.0 - u)) ** (1.0 / beta))
        if t > test_duration:
            ttf.append((i, test_duration, True))   # censored
        else:
            ttf.append((i, round(t, 1), False))
    return ttf


def write_burn_in_csv(path: Path, *, scenario: str = "clean") -> None:
    if scenario == "debate":
        beta = 0.85   # infant mortality
    elif scenario == "bad":
        beta = 0.6
    else:
        beta = 1.5    # healthy wear-out
    rows = burn_in_ttf(beta=beta)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chip_idx", "ttf_hours", "censored"])
        w.writerows(rows)


# ---------- Bulk generator ----------

def generate_lot_csvs(
    *,
    lot_id: str,
    out_root: Path,
    scenario: str = "clean",
) -> Path:
    out_dir = out_root / lot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_reflow_csv(out_dir / "reflow.csv", scenario=scenario)
    write_bond_force_csv(out_dir / "bond_force.csv", scenario=scenario)
    write_burn_in_csv(out_dir / "burn_in.csv", scenario=scenario)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lot", default="LOT-2026-001")
    parser.add_argument("--scenario", default="clean", choices=["clean", "debate", "bad", "early_kill"])
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "data" / "synthetic" / "csvs"),
    )
    args = parser.parse_args()
    out = generate_lot_csvs(lot_id=args.lot, out_root=Path(args.out), scenario=args.scenario)
    print(f"Generated CSVs for {args.lot} ({args.scenario}) -> {out}")


if __name__ == "__main__":
    main()
