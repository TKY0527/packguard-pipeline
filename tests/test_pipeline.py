"""
Smoke tests for the pipeline runtime. These guarantee that every demo scenario
end-to-end produces the *kind* of result we expect — they are NOT thorough unit
tests. Person 1's physics modules will get their own unit tests in their repo.
"""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from packguard_pipeline.checkpoints import ALL_CHECKPOINTS
from packguard_pipeline.main import app
from packguard_pipeline.models import (
    Action,
    Application,
    DecisionState,
    InputFiles,
    LotState,
    StepName,
)
from packguard_pipeline.pipeline import CheckpointPipeline
from packguard_pipeline.storage import get_store


def _make_lot(lot_id: str, application: Application = Application.AUTOMOTIVE) -> LotState:
    now = datetime.now(timezone.utc)
    return LotState(
        lot_id=lot_id,
        package_type="BGA-256",
        application=application,
        input_files=InputFiles(),
        created_at=now,
        updated_at=now,
    )


def _pipeline() -> CheckpointPipeline:
    return CheckpointPipeline(ALL_CHECKPOINTS)


# ---------- Direct pipeline tests ----------

def test_clean_lot_passes_all_seven():
    # Clean lot is consumer-grade — application threshold is loose (1e-3).
    # Same physical package against automotive's tight 1e-5 would HOLD/REJECT.
    lot = _make_lot("LOT-2026-001", application=Application.CONSUMER)
    lot = _pipeline().run(lot)
    # Every checkpoint visited.
    assert len(lot.checkpoints) == 7
    # Final decision exists and verdict is SHIP for clean lot.
    assert lot.final_decision is not None
    assert lot.final_decision.verdict.value == "SHIP"
    assert lot.decision_state in (DecisionState.PASS_, DecisionState.FLAG)


def test_early_kill_stops_at_checkpoint_1():
    lot = _make_lot("LOT-2026-002")
    lot = _pipeline().run(lot)
    # Pipeline short-circuited at C1.
    assert len(lot.checkpoints) == 1
    assert lot.checkpoints[0].action == Action.KILL
    assert lot.checkpoints[0].step_name == StepName.DICING
    # Forward sim narrative present — real physics; could fail at any
    # downstream step depending on stress profile. We only assert that it
    # predicts SOME failure and the cost-avoided number is the demo's.
    fs = lot.checkpoints[0].forward_sim_prediction
    assert fs is not None
    assert fs.fails_at_step is not None
    assert fs.cost_avoided_usd > 0
    # Lot state KILLed, no final decision because we never got to C7.
    assert lot.decision_state == DecisionState.KILL
    assert lot.final_decision is None


def test_debate_trigger_at_wire_bond():
    lot = _make_lot("LOT-2026-003", application=Application.SERVER)
    lot = _pipeline().run(lot)

    # Pipeline ran all 7.
    assert len(lot.checkpoints) == 7
    c3 = next(cp for cp in lot.checkpoints if cp.step_name == StepName.WIRE_BOND)
    assert c3.action == Action.FLAG
    assert any("DEBATE TRIGGER" in r for r in c3.reasons)
    assert "Process beats specification" in (c3.rule_fired or "")
    # Final verdict should be HOLD (yellow): borderline aggregate, not a REJECT.
    assert lot.final_decision is not None
    assert lot.final_decision.verdict.value == "HOLD"


def test_kill_short_circuits_pipeline():
    """KILL at C1 must prevent C2-C7 from running. Demo cost-saved counter depends on this."""
    lot = _make_lot("LOT-2026-002")
    lot = _pipeline().run(lot)
    step_names = [cp.step_name for cp in lot.checkpoints]
    assert step_names == [StepName.DICING]


def test_application_threshold_used():
    lot = _make_lot("LOT-2026-001", application=Application.CONSUMER)
    lot = _pipeline().run(lot)
    assert lot.final_decision is not None
    # Consumer threshold should be looser than automotive
    assert lot.final_decision.threshold_used == 0.001


# ---------- HTTP layer ----------

@pytest.fixture
def client() -> TestClient:
    get_store().clear()
    return TestClient(app)


def test_root_endpoint(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "packguard-pipeline"


def test_demo_clean(client: TestClient):
    r = client.get("/demo/clean")
    assert r.status_code == 200
    body = r.json()
    assert body["decision_state"] in ("PASS", "FLAG")
    assert len(body["checkpoints"]) == 7


def test_demo_early_kill(client: TestClient):
    r = client.get("/demo/early_kill")
    assert r.status_code == 200
    body = r.json()
    assert body["decision_state"] == "KILL"
    assert len(body["checkpoints"]) == 1
    assert body["checkpoints"][0]["action"] == "kill"
    assert body["checkpoints"][0]["forward_sim_prediction"]["cost_avoided_usd"] > 0


def test_demo_debate(client: TestClient):
    r = client.get("/demo/debate")
    assert r.status_code == 200
    body = r.json()
    assert len(body["checkpoints"]) == 7
    c3 = next(cp for cp in body["checkpoints"] if cp["step_name"] == "WIRE_BOND")
    assert c3["action"] == "flag"


def test_lot_roundtrip(client: TestClient):
    fired = client.get("/demo/clean").json()
    lot_id = fired["lot_id"]
    fetched = client.get(f"/lot/{lot_id}").json()
    assert fetched["lot_id"] == lot_id


def test_unknown_lot_404(client: TestClient):
    r = client.get("/lot/LOT-9999-999")
    assert r.status_code == 404


def test_schema_export_endpoint(client: TestClient):
    r = client.get("/schema/lot_state")
    assert r.status_code == 200
    schema = r.json()
    # JSON Schema sanity check.
    assert "$defs" in schema or "properties" in schema
