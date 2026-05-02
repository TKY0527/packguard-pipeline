"""
Checkpoint 6 — After Test (Electrical + Burn-in).

Tools:
- DET: Western Electric SPC rules
- DET: Weibull beta/eta fit
- DET: Anomaly detection
- AI: Knowledge match (RAG over historical fails)

Decision rule:
- β < 1 (infant mortality) → FLAG batch
- critical SPC violation → KILL
- otherwise PASS
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
from ..mock_data import demo_test_analysis

WEIBULL_BETA_INFANT = 1.0


class TestCheckpoint(Checkpoint):
    checkpoint_id = 6
    step_name = StepName.TEST

    def analyze(self, lot: LotState) -> dict[str, Any]:
        return demo_test_analysis(lot)

    def decide(self, lot: LotState, analysis: dict[str, Any]) -> CheckpointResult:
        beta: float = analysis["weibull_beta"]
        spc_violation: bool = analysis.get("critical_spc_violation", False)
        tools: list[ToolCall] = analysis["tool_calls"]

        reasons: list[str] = []
        rule_fired: str | None = None
        action: Action

        if spc_violation:
            action = Action.KILL
            rule_fired = "critical SPC violation"
            reasons.append("Western Electric run rules — process out of statistical control")

        elif beta < WEIBULL_BETA_INFANT:
            action = Action.FLAG
            rule_fired = f"Weibull β < {WEIBULL_BETA_INFANT}"
            reasons.append(
                f"β = {beta:.2f} indicates infant-mortality population — burn-in screening recommended"
            )

        else:
            action = Action.PASS_
            reasons.append(f"Weibull β = {beta:.2f} indicates healthy random/wear-out distribution")

        now = datetime.now(timezone.utc)
        return CheckpointResult(
            checkpoint_id=self.checkpoint_id,
            step_name=self.step_name,
            tools_called=tools,
            action=action,
            reasons=reasons,
            rule_fired=rule_fired,
            started_at=now,
            finished_at=now,
        )
