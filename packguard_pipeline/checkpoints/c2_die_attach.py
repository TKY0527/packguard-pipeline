"""
Checkpoint 2 — After Die Attach.

Tools:
- DET: Void Ratio Calculator (CNN U-Net segmentation + area ratio)
- DET: Thermal Resistance Estimator
- DET: Post-reflow Void Predictor (Ideal Gas Law)
- AI: Vision (clustered vs dispersed void distribution)

Decision rule:
- void_ratio > 25% (JEDEC) OR junction temp predicted > limit → KILL
- Clustered void pattern → FLAG
- PASS otherwise
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
from ..mock_data import demo_die_attach_analysis

VOID_KILL_RATIO = 0.25   # JEDEC threshold
VOID_FLAG_RATIO = 0.15


class DieAttachCheckpoint(Checkpoint):
    checkpoint_id = 2
    step_name = StepName.DIE_ATTACH

    def analyze(self, lot: LotState) -> dict[str, Any]:
        return demo_die_attach_analysis(lot)

    def decide(self, lot: LotState, analysis: dict[str, Any]) -> CheckpointResult:
        void_ratio: float = analysis["void_ratio"]
        is_clustered: bool = analysis.get("is_clustered", False)
        rupture_predicted: bool = analysis.get("post_reflow_rupture", False)
        tj_excess: bool = analysis.get("junction_temp_exceeds_limit", False)
        tools: list[ToolCall] = analysis["tool_calls"]

        reasons: list[str] = []
        rule_fired: str | None = None
        action: Action
        cost_avoided = 0.0

        if void_ratio > VOID_KILL_RATIO:
            action = Action.KILL
            rule_fired = f"void_ratio > {VOID_KILL_RATIO * 100:.0f}% (JEDEC)"
            reasons.append(f"Void ratio {void_ratio * 100:.1f}% exceeds JEDEC kill threshold")
            cost_avoided = analysis.get("cost_avoided_usd", 2400.0)

        elif rupture_predicted:
            action = Action.KILL
            rule_fired = "post_reflow_void_predictor: rupture predicted"
            reasons.append("Ideal gas law predicts void rupture into delamination at reflow")
            cost_avoided = analysis.get("cost_avoided_usd", 2400.0)

        elif tj_excess:
            action = Action.KILL
            rule_fired = "thermal_resistance: junction temp will exceed safe limit"
            reasons.append("Estimated junction temperature exceeds operating envelope")
            cost_avoided = analysis.get("cost_avoided_usd", 2400.0)

        elif is_clustered or void_ratio > VOID_FLAG_RATIO:
            action = Action.FLAG
            rule_fired = "clustered_voids OR borderline_void_ratio"
            if is_clustered:
                reasons.append("Voids clustered — potential hot spot risk")
            if void_ratio > VOID_FLAG_RATIO:
                reasons.append(f"Void ratio {void_ratio * 100:.1f}% in flag band")

        else:
            action = Action.PASS_
            reasons.append(f"Void ratio {void_ratio * 100:.1f}% within JEDEC spec, dispersed pattern")

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
