"""
Generate the synthetic data folder for all 3 demo scenarios.

Run:
    python scripts/gen_demo_data.py
"""

import sys
from pathlib import Path

# Make project root importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packguard_pipeline.synthetic.dicing import generate_lot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "dicing"


def main() -> None:
    for lot_id, scenario in [
        ("LOT-2026-001", "clean"),
        ("LOT-2026-002", "early_kill"),
        ("LOT-2026-003", "debate"),
    ]:
        out = generate_lot(
            lot_id=lot_id,
            out_root=ROOT,
            count=24,
            scenario=scenario,
            seed=42,
        )
        print(f"  {lot_id} ({scenario:11s}) -> {out}")


if __name__ == "__main__":
    main()
