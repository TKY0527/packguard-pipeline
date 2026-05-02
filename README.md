# PackGuard Pipeline (Person 2)

**Inline checkpoint pipeline + computer-vision defect detection** for the
PackGuard v2.0 hackathon entry — Micron Case Study Competition 2026.

This service is the demo runtime: it ingests a lot's data, runs it through 7
inline checkpoints, calls Person 1's physics + the local CV models, and
produces a structured `LotState` that Person 3's orchestrator and Person 4's
frontend consume.

```
                   ┌──────────────────────────┐
   POST /analyze ─►│   CheckpointPipeline     │─► LotState ─► Person 3
                   │   ├── 1. Dicing          │              (FastAPI :8002)
                   │   ├── 2. Die Attach      │
                   │   ├── 3. Wire Bond       │─► LotState ─► Person 4
                   │   ├── 4. Molding         │              (Next.js :3000)
                   │   ├── 5. Reflow          │
                   │   ├── 6. Test            │
                   │   └── 7. Final Gate      │
                   └──────────────────────────┘
```

## Run it

```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash
# or:  source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

uvicorn packguard_pipeline.main:app --reload --port 8001
```

Then open <http://localhost:8001/docs> for the interactive Swagger UI.

## Day 1 demo endpoints

| URL | What you get |
|---|---|
| `GET /demo/clean` | All 7 checkpoints PASS, lot SHIPs |
| `GET /demo/early_kill` | KILL at Checkpoint 1 — forward sim narrative + $1,847 cost-avoided |
| `GET /demo/debate` | Vision OK + SPC drift at Wire Bond → Debate Rule 2 fires |
| `GET /lot/{lot_id}` | Full lot state for any previously analyzed lot |
| `GET /schema/lot_state` | JSON Schema (Person 4: pipe to `json-schema-to-typescript`) |

## Architecture

* `packguard_pipeline/models.py` — Pydantic schemas. **The shared data contract.**
* `packguard_pipeline/pipeline.py` — `Checkpoint` ABC + `CheckpointPipeline` runner.
* `packguard_pipeline/checkpoints/` — One module per production step (1-7).
* `packguard_pipeline/mock_data.py` — Day 1 fixtures for the 3 demo scenarios.
* `packguard_pipeline/main.py` — FastAPI app.
* `packguard_pipeline/synthetic/` — Procedural defect-image generators.
* `packguard_pipeline/storage.py` — In-memory `LotStore`.

## Test it

```bash
pytest tests/ -v
```

## Generate synthetic demo data

```bash
python scripts/gen_demo_data.py
```

Produces `data/synthetic/dicing/LOT-2026-{001,002,003}/die_*.png` plus
`labels.json` ground truth.

## Export the JSON schema (for Person 4)

```bash
python -m packguard_pipeline.export_schema
# writes docs/lot_state_schema.json
```

## Day 2-7 work

* **Day 2-4** — Replace `mock_data.py` per-checkpoint analyses with real
  CV models (`ultralytics` YOLOv8 fine-tuned on synthetic data, U-Net for void
  segmentation) and real calls into `packguard_physics`.
* **Day 5** — Add CORS middleware (per API contract §5).
* **Day 5** — Wire to Person 3's `/report` endpoint after final gate.
* **Day 6** — Demo polish, edge-case handling.

## API contract

This service implements §3 of the team API contract owned by Person 4. Port
**8001** is locked. If contract §1 changes, regenerate the JSON schema via
`python -m packguard_pipeline.export_schema` and notify the team.

## Person 1 physics interop

Verified against `packguard_physics.ReliabilityResult` shape. The actual repo
struct has two extra fields beyond contract §2 — both surfaced in
`models.PhysicsOutput`:

```python
inputs:    dict[str, Any]   # what was passed in (audit)
citations: list[str]        # JEDEC / textbook references
```

**Update API contract §2** to include these.
