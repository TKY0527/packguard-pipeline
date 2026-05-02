"""
Generate the full multi-modal synthetic dataset for all 3 demo scenarios.

Per-lot output:
  data/synthetic/dicing/<lot_id>/   — die top-down AOI images + labels.json
  data/synthetic/voids/<lot_id>/    — die-attach X-ray images + labels.json
  data/synthetic/solder/<lot_id>/   — solder-joint X-ray images + labels.json
  data/synthetic/csvs/<lot_id>/     — reflow.csv, bond_force.csv, burn_in.csv

Run:
    python scripts/gen_demo_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packguard_pipeline.synthetic.dicing import generate_lot as gen_dicing  # noqa: E402
from packguard_pipeline.synthetic.voids import generate_lot_voids  # noqa: E402
from packguard_pipeline.synthetic.solder import generate_lot_solder  # noqa: E402
from packguard_pipeline.synthetic.csv_data import generate_lot_csvs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data" / "synthetic"

LOTS = [
    ("LOT-2026-001", "clean"),
    ("LOT-2026-002", "early_kill"),
    ("LOT-2026-003", "debate"),
]


def main() -> None:
    for lot_id, scenario in LOTS:
        d1 = gen_dicing(lot_id=lot_id, out_root=ROOT / "dicing", count=24, scenario=scenario, seed=42)
        d2 = generate_lot_voids(lot_id=lot_id, out_root=ROOT / "voids", count=12, scenario=scenario, seed=42)
        d3 = generate_lot_solder(lot_id=lot_id, out_root=ROOT / "solder", count=4, scenario=scenario, seed=42)
        d4 = generate_lot_csvs(lot_id=lot_id, out_root=ROOT / "csvs", scenario=scenario)
        print(f"  {lot_id} ({scenario:11s}) -> dicing+voids+solder+csvs OK")
    print(f"\nDataset root: {ROOT}")


if __name__ == "__main__":
    main()
