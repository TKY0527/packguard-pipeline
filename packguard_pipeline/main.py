"""
PackGuard Pipeline FastAPI service (Person 2).

Endpoints:
  GET  /                    health probe
  GET  /healthz             liveness
  POST /analyze             multipart upload → kicks off pipeline → returns lot_id
  GET  /lot/{lot_id}        full LotState for the frontend / orchestrator
  GET  /lots                list known lot_ids (debug)
  GET  /demo/{scenario}     instant-fire demo (clean | early_kill | debate)
  GET  /schema/lot_state    dump JSON Schema (export for Person 4)

Day 1: synchronous pipeline run. If we need long-running async, switch to
BackgroundTasks or a job queue on Day 4 — but the demo runs in <500ms so
synchronous is fine for now.

Run locally:
  uvicorn packguard_pipeline.main:app --reload --port 8001
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .checkpoints import ALL_CHECKPOINTS
from .models import (
    AnalyzeResponse,
    Application,
    DecisionState,
    InputFiles,
    LotState,
)
from .pipeline import CheckpointPipeline
from .storage import LotStore, get_store

app = FastAPI(
    title="PackGuard Pipeline Service",
    version="0.1.0",
    description="Person 2 — Inline Checkpoint Pipeline + CV. Day 1 mock pipeline.",
)

# CORS — allow Person 4's Next.js dev server (localhost:3000) to call us.
# Without this, browsers block cross-origin XHR/fetch from the frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_pipeline = CheckpointPipeline(ALL_CHECKPOINTS)


def _new_lot_id(scenario: str | None = None) -> str:
    seq = uuid4().hex[:6].upper()
    suffix = {"clean": "001", "early_kill": "002", "debate": "003"}.get(
        scenario or "", seq[:3]
    )
    return f"LOT-2026-{suffix}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------- Health ----------

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "packguard-pipeline",
        "version": "0.1.0",
        "owner": "Person 2",
        "endpoints": "/analyze, /lot/{lot_id}, /demo/{scenario}, /schema/lot_state",
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "ts": _now().isoformat()}


# ---------- Analyze (production path) ----------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    package_type: Annotated[str, Form()],
    application: Annotated[Application, Form()],
    files: Annotated[list[UploadFile], File()] = None,  # type: ignore[assignment]
    lot_id_hint: Annotated[Optional[str], Form()] = None,
    store: LotStore = Depends(get_store),
) -> AnalyzeResponse:
    """
    Day 1: file content not actually inspected — checkpoints dispatch by lot_id
    pattern. Day 2-4: hook real CV / physics here.
    """
    lot_id = lot_id_hint or _new_lot_id()

    # Persist file names (we are NOT writing bytes to disk yet — Day 2 task).
    upload_names: list[str] = [f.filename for f in (files or []) if f.filename]
    input_files = InputFiles(
        xray_images=[n for n in upload_names if "xray" in n.lower()],
        aoi_images=[n for n in upload_names if "aoi" in n.lower() or "die" in n.lower()],
        reflow_csv=next((n for n in upload_names if n.lower().endswith(".csv") and "reflow" in n.lower()), None),
        bond_force_log=next((n for n in upload_names if "bond" in n.lower()), None),
        test_data_csv=next((n for n in upload_names if "test" in n.lower() and n.lower().endswith(".csv")), None),
        material_spec_json=next((n for n in upload_names if n.lower().endswith(".json")), None),
    )

    lot = LotState(
        lot_id=lot_id,
        package_type=package_type,
        application=application,
        input_files=input_files,
        created_at=_now(),
        updated_at=_now(),
    )

    lot = _pipeline.run(lot)
    store.put(lot)

    return AnalyzeResponse(
        lot_id=lot.lot_id,
        decision_state=lot.decision_state,
        current_step=lot.current_step,
        message=f"Pipeline complete: {lot.decision_state.value} at step {lot.current_step}",
    )


# ---------- Get lot state ----------

@app.get("/lot/{lot_id}", response_model=LotState)
def get_lot(lot_id: str, store: LotStore = Depends(get_store)) -> LotState:
    lot = store.get(lot_id)
    if lot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown lot {lot_id}")
    return lot


@app.get("/lots")
def list_lots(store: LotStore = Depends(get_store)) -> dict[str, list[str]]:
    return {"lot_ids": store.all_ids()}


# ---------- Demo shortcuts ----------

@app.get("/demo/{scenario}", response_model=LotState)
def fire_demo(
    scenario: str,
    store: LotStore = Depends(get_store),
) -> LotState:
    """One-click demo. Useful for Person 4's UI and rehearsals."""
    if scenario not in {"clean", "early_kill", "debate"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scenario must be one of: clean, early_kill, debate",
        )

    application = Application.AUTOMOTIVE if scenario != "debate" else Application.SERVER
    lot = LotState(
        lot_id=_new_lot_id(scenario),
        package_type="BGA-256",
        application=application,
        input_files=InputFiles(),
        created_at=_now(),
        updated_at=_now(),
    )
    lot = _pipeline.run(lot)
    store.put(lot)
    return lot


# ---------- Schema export (for Person 4) ----------

@app.get("/schema/lot_state")
def schema_lot_state() -> JSONResponse:
    """Returns the JSON Schema of LotState. Person 4 uses this to generate TS types."""
    return JSONResponse(content=LotState.model_json_schema())
