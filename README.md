# PackGuard Pipeline (Person 2)

**Inline 7-checkpoint pipeline + computer-vision defect detection** for the
PackGuard v2.0 hackathon entry — Micron Case Study Competition 2026.

The demo runtime: ingest a lot's data → run 7 inline checkpoints → call
Person 1's [`packguard_physics`](https://github.com/JohnDoeSKipper/packguard-physics)
+ local CV models → produce a structured `LotState` that Person 3's orchestrator
and Person 4's frontend consume.

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

## Day 1 → Day 7 status

| Subsystem | Status | Notes |
|---|---|---|
| Lot State JSON Schema | ✅ locked | `docs/lot_state_schema.json` (14 sub-schemas) |
| FastAPI service `:8001` | ✅ | `/analyze`, `/lot/{id}`, `/demo/{scenario}`, `/schema/lot_state` |
| 7 checkpoints | ✅ | All seven implemented, decision rules + traceability |
| 3 demo scenarios | ✅ | clean / early_kill / debate, all back-to-back deterministic |
| Real Person 1 physics | ✅ | `packguard_physics` imported and called from each checkpoint |
| Forward-Sim Engine | ✅ | Person 1's `simulate_defect` adapted to our `ForwardSimPrediction` |
| OpenCV crack detector | ✅ | Edge → Hough → length filter |
| OpenCV void segmenter | ✅ | Otsu + connected components + clustering test |
| U-Net void seg (torch) | ✅ scaffold | Training + inference; included for completeness |
| YOLOv8 solder detector | ✅ scaffold | Training + inference; uses ultralytics |
| Synthetic data generators | ✅ | dicing, voids, solder, reflow CSV, bond CSV, burn-in CSV |
| Claude Vision wrapper | ✅ | Cost-controlled, env-key gated, deferral fallback |
| File upload (real bytes) | ✅ | `data/uploads/<lot_id>/...` with categorization |
| CORS for Next.js | ✅ | `localhost:3000` allowed |
| Tests | ✅ 31/31 | pipeline, physics_adapter, CV, file_storage, claude_vision |

## Run it

```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash
# or:  source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
pip install -e ../packguard-physics/Physics_Model    # Person 1's package

uvicorn packguard_pipeline.main:app --reload --port 8001
```

Then open <http://localhost:8001/docs> for the interactive Swagger UI.

## Demo endpoints

| URL | What you get |
|---|---|
| `GET /demo/clean` | All 7 checkpoints PASS, lot SHIPs (consumer threshold) |
| `GET /demo/early_kill` | KILL at Checkpoint 1 — forward sim narrative + $1,847 cost-avoided |
| `GET /demo/debate` | Vision OK + SPC drift @ wire bond → Rule 2 fires → HOLD |
| `POST /analyze` | multipart upload — files saved to `data/uploads/<lot_id>/`, real CV runs |
| `GET /lot/{lot_id}` | Full lot state for any previously analyzed lot |
| `GET /schema/lot_state` | Live JSON Schema dump (Person 4 → `lib/types.ts`) |

## Architecture

```
packguard_pipeline/
├── models.py             — Pydantic schemas (THE shared data contract)
├── pipeline.py           — Checkpoint ABC + CheckpointPipeline runner
├── checkpoints/          — One module per production step (1-7)
│   ├── c1_dicing.py
│   ├── c2_die_attach.py
│   ├── ...
│   └── c7_final_gate.py
├── physics_adapter.py    — Bridges to Person 1's packguard_physics
├── mock_data.py          — Scenario fixtures (with real physics under the hood)
├── cv/                   — Computer vision
│   ├── crack_detector.py        — OpenCV Canny + Hough
│   ├── void_segmenter_cv.py     — OpenCV Otsu + ConnectedComponents
│   ├── void_segmenter_unet.py   — torch + smp U-Net (training + inference)
│   └── solder_yolo.py           — ultralytics YOLOv8 (training + inference)
├── synthetic/            — Procedural defect generators
│   ├── dicing.py         — die top-down AOI (cracks, edge chips, scratches)
│   ├── voids.py          — die-attach X-ray (void blobs, clustering)
│   ├── solder.py         — solder-joint X-ray (HIP, voids, missing, bridges)
│   └── csv_data.py       — reflow profile, bond force, burn-in TTF
├── claude_vision.py      — Anthropic API wrapper (env-key gated)
├── file_storage.py       — Per-lot upload directory
├── storage.py            — In-memory LotStore
└── main.py               — FastAPI app
```

## Test it

```bash
pytest tests/ -v
```

31 tests — pipeline integrity, physics outputs (real Person 1 calls), CV
inference, file storage, Claude Vision deferral.

## Generate the demo dataset

```bash
python scripts/gen_demo_data.py
```

Produces:
- `data/synthetic/dicing/LOT-2026-{001,002,003}/die_*.png + labels.json`
- `data/synthetic/voids/LOT-2026-{001,002,003}/xray_*.png + labels.json`
- `data/synthetic/solder/LOT-2026-{001,002,003}/solder_*.png + labels.json`
- `data/synthetic/csvs/LOT-2026-{001,002,003}/{reflow,bond_force,burn_in}.csv`

## Train the CV models (optional)

```bash
# U-Net for void segmentation
python -m packguard_pipeline.cv.void_segmenter_unet train \
  --data data/synthetic/voids/LOT-2026-001 \
  --out models/void_unet.pt --epochs 5

# YOLOv8 for solder defects
python -m packguard_pipeline.cv.solder_yolo train \
  --data data/synthetic/solder/LOT-2026-001 \
  --out models/solder_yolo --epochs 3
```

For the demo, the OpenCV-only path works without training and is the default
inside the checkpoints.

## Claude Vision (optional)

To enable second-opinion CV escalation when the deterministic detector reports
confidence below 80%:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Without the key, the pipeline gracefully defers (`VisionDeferred`) and the
checkpoint records that fact in the audit log.

## Export the JSON schema (for Person 4)

```bash
python -m packguard_pipeline.export_schema    # writes docs/lot_state_schema.json
```

Person 4 generates TypeScript types via:
```bash
npx json-schema-to-typescript docs/lot_state_schema.json -o web/lib/types.ts
```

## API contract status

This service implements §3 of the team API contract. Port **8001** is locked.

**§2 update note**: Person 1's `ReliabilityResult` includes two fields beyond
the original contract — please reflect them in the contract doc:

```python
inputs:    dict[str, Any]   # echoed inputs, audit trail
citations: list[str]        # JEDEC / textbook references
```
