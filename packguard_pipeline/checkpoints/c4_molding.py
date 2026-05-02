"""
Checkpoint 4 — After Molding.

Tools:
- DET: Wire Sweep Calculator (fluid dynamics)
- DET: Cure Shrinkage Stress Model
- DET: EMC Void Detection
- AI: Vision (cosmetic vs functional defect)

Decision rule:
- wire_deflection > 10% pitch → KILL
- interface_stress > adhesion_strength → KILL
- borderline → FLAG
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
from ..mock_data import demo_molding_analysis

DEFLECTION_KILL_PCT = 0.10
STRESS_RATIO_KILL = 1.0  # interface_stress / adhesion_strength


class MoldingCheckpoint(Checkpoint):
    checkpoint_id = 4
    step_name = StepName.MOLDING

    def analyze(self, lot: LotState) -> dict[str, Any]:
        return demo_molding_analysis(lot)

    def decide(self, lot: LotState, analysis: dict[str, Any]) -> CheckpointResult:
        deflection: float = analysis["wire_deflection_pct"]
        stress_ratio: float = analysis["interface_stress_ratio"]
        tools: list[ToolCall] = analysis["tool_calls"]

        reasons: list[str] = []
        rule_fired: str | None = None
        action: Action
        cost_avoided = 0.0

        if deflection > DEFLECTION_KILL_PCT:
            action = Action.KILL
            rule_fired = f"wire_deflection > {DEFLECTION_KILL_PCT * 100:.0f}% pitch"
            reasons.append(f"Wire deflection {deflection * 100:.1f}% — short risk")
            cost_avoided = analysis.get("cost_avoided_usd", 4200.0)

        elif stress_ratio > STRESS_RATIO_KILL:
            action = Action.KILL
            rule_fired = "interface_stress > adhesion_strength"
            reasons.append(f"Cure shrinkage stress ratio {stress_ratio:.2f} > 1.0 — delamination risk")
            cost_avoided = analysis.get("cost_avoided_usd", 4200.0)

        elif deflection > 0.07 or stress_ratio > 0.85:
            action = Action.FLAG
            rule_fired = "borderline mold pattern"
            reasons.append("Borderline deflection or stress — extra inspection downstream")

        else:
            action = Action.PASS_
            reasons.append("Mold cure within fluid-dynamics + stress envelope")

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
