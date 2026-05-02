"""
Export the LotState JSON Schema to docs/lot_state_schema.json so Person 4 can
generate TypeScript types from it (e.g., via `json-schema-to-typescript`).

Run:
    python -m packguard_pipeline.export_schema
"""

import json
from pathlib import Path

from .models import LotState

OUT = Path(__file__).resolve().parents[1] / "docs" / "lot_state_schema.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    schema = LotState.model_json_schema()
    OUT.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}  ({len(schema.get('$defs', {}))} sub-schemas, "
          f"{OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
