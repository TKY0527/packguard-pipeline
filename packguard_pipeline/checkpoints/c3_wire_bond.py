"""
Checkpoint 3 — After Wire Bond.

Tools:
- DET: Bond Pull/Shear SPC + Cpk (Western Electric rules)
- DET: Arrhenius IMC Growth Model
- DET: Bond geometry stats
- AI: Vision (wire sweep pattern recognition, fish-eye defects)

Decision rule:
- Cpk < 1.33 → FLAG
- Predicted IMC > 5µm → KILL
- sweep deflection > 10% pitch → KILL

This checkpoint is also where the **Debate Trigger** demo scenario fires:
Vision says OK, but SPC shows drift > 2σ → Rule 2 (Process beats specification)
hands the conflict to Person 3's orchestrator.
"""

from datetime import datetime, timezone
from typing import Any

from ..models import (
    Action,
    CheckpointResult,
    LotState,
    StepName,
    ToolCall,
)
from ..pipeline import Checkpoint
from ..mock_data import demo_wire_bond_analysis

CPK_FLAG = 1.33     # process capability threshold
IMC_KILL_UM = 5.0   # brittle joint above this thickness
SWEEP_KILL_PCT = 0.10  # 10% of pitch


class WireBondCheckpoint(Checkpoint):
    checkpoint_id = 3
    step_name = StepName.WIRE_BOND

    def analyze(self, lot: LotState) -> dict[str, Any]:
        return demo_wire_bond_analysis(lot)

    def decide(self, lot: LotState, analysis: dict[str, Any]) -> CheckpointResult:
        cpk: float = analysis["cpk"]
        imc_um: float = analysis["predicted_imc_um"]
        sweep_pct: float = analysis["sweep_deflection_pct"]
        vision_says_ok: bool = analysis.get("vision_says_ok", True)
        sigma_drift: float = analysis.get("sigma_drift", 0.0)
        tools: list[ToolCall] = analysis["tool_calls"]

        reasons: list[str] = []
        rule_fired: str | None = None
        action: Action
        cost_avoided = 0.0

        if imc_um > IMC_KILL_UM:
            action = Action.KILL
            rule_fired = f"predicted_IMC > {IMC_KILL_UM}µm"
            reasons.append(
                f"Arrhenius model predicts {imc_um:.2f}µm IMC over service life — brittle joint"
            )
            cost_avoided = analysis.get("cost_avoided_usd", 3100.0)

        elif sweep_pct > SWEEP_KILL_PCT:
            action = Action.KILL
            rule_fired = f"wire_sweep > {SWEEP_KILL_PCT * 100:.0f}% pitch"
            reasons.append(f"Sweep deflection {sweep_pct * 100:.1f}% — short risk")
            cost_avoided = analysis.get("cost_avoided_usd", 3100.0)

        elif cpk < CPK_FLAG:
            action = Action.FLAG
            rule_fired = f"Cpk < {CPK_FLAG}"
            reasons.append(f"Cpk {cpk:.2f} below {CPK_FLAG} — process unstable")

            # Debate trigger: Vision says OK but SPC shows drift > 2σ.
            # Per brief §6.1, this is one of the formal trigger conditions.
            if vision_says_ok and sigma_drift > 2.0:
                reasons.append(
                    f"DEBATE TRIGGER: Vision OK but SPC drift {sigma_drift:.1f}σ — "
                    "Rule 2 (Process beats specification). Escalating to orchestrator."
                )
                rule_fired += " + Rule 2 (Process beats specification)"

        else:
            action = Action.PASS_
            reasons.append(f"Cpk {cpk:.2f}, predicted IMC {imc_um:.2f}µm — within spec")

        now = datetime.now(timezone.utc)
        return CheckpointResult(
            checkpoint_id=self.checkpoint_id,
            step_name=self.step_name,
            tools_called=tools,
            action=action,
            reasons=reasons,
            rule_fired=rule_fired,
            cost_avoided_usd=cost_avoided,
            started_at=now,
            finished_at=now,
        )
