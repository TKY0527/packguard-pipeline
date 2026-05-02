# Notes for the team API contract

These are Person 2's deltas / clarifications to the contract Person 4 is
maintaining. Send these to the group on Day 1.

## §1 Lot State JSON

* Fully defined in [`packguard_pipeline/models.py`](../packguard_pipeline/models.py) (`LotState`).
* Run `python -m packguard_pipeline.export_schema` → writes
  [`docs/lot_state_schema.json`](./lot_state_schema.json).
* Person 4: pipe that JSON Schema through `json-schema-to-typescript` to get
  `lib/types.ts` deterministically:
  ```bash
  npx json-schema-to-typescript docs/lot_state_schema.json -o web/lib/types.ts
  ```

### Lot ID convention

`LOT-YYYY-NNN` — for the 3 demo scenarios:
| lot_id | scenario |
|---|---|
| `LOT-2026-001` | Clean (all PASS, SHIPs) |
| `LOT-2026-002` | Early kill at Checkpoint 1 |
| `LOT-2026-003` | Debate trigger at Checkpoint 3 |

The pipeline's `analyze()` dispatches on `lot_id` for Day 1 fixtures.

## §2 Physics Function Output

**Update contract.** Person 1's actual `ReliabilityResult` has TWO extra fields:

```python
{
  "probability_of_failure": float,    # 0.0 .. 1.0
  "confidence_interval":   [low, high],
  "predicted_lifetime":    float,
  "units":                 str,
  "model_used":            str,
  "assumptions":           [str, ...],
  "inputs":                {...},     # NEW — audit trail
  "citations":             [str, ...] # NEW — JEDEC / textbook refs
}
```

`citations` in particular is a judge-trust feature — surface them in the
final report.

## §3 Pipeline Service

* **Base URL: `http://localhost:8001`**
* `POST /analyze` (multipart) → `{ lot_id, decision_state, current_step, message }`
* `GET /lot/{lot_id}` → full `LotState`
* `GET /demo/{scenario}` → instant-fire demo (`clean` | `early_kill` | `debate`)
* `GET /schema/lot_state` → live JSON Schema dump
* `GET /lots` → debug, list stored lot_ids
* `GET /healthz` → liveness

## §5 CORS

Will be added Day 5 once Person 4's frontend is calling us. Tracking todo —
no premature port opening.

## §6 Change log entry

```
2026-05-02 — Person 2: Locked LotState schema, pipeline runtime, and 3 demo
fixtures. Verified Person 1 physics output. Requested §2 update to add
`inputs` and `citations` fields.
```
