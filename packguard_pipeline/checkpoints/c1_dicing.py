"""
Checkpoint 1 — After Dicing.

Tools:
- DET: Edge Chip Classifier (JEDEC JESD22-B116)
- DET: Griffith Crack Propagation
- DET: Survival Simulator (forward sim across remaining 6 steps)
- AI: Vision (only for ambiguous chip-vs-dust)

Decision rule:
- Crack > 1.5mm OR survival sim predicts failure → KILL
- Borderline → FLAG
- Clean → PASS
"""

from datetime import datetime, timezone
from typing import Any

from ..models import (
    Action,
    CheckpointResult,
    ForwardSimPrediction,
    LotState,
    StepName,
    ToolCall,
    ToolType,
)
from ..pipeline import Checkpoint
from ..mock_data import demo_dicing_analysis

CRACK_KILL_MM = 1.5  # JEDEC JESD22-B116 derived
CRACK_FLAG_MM = 1.0


class DicingCheckpoint(Checkpoint):
    checkpoint_id = 1
    step_name = StepName.DICING

    def analyze(self, lot: LotState) -> dict[str, Any]:
        # Day 1: dispatch by lot_id to a fixture. Day 2+: replace with real CV
        # over `lot.input_files.aoi_images` and a real call into
        # packguard_physics.griffith_fracture / survival_simulator.
        return demo_dicing_analysis(lot)

    def decide(self, lot: LotState, analysis: dict[str, Any]) -> CheckpointResult:
        worst_crack_mm: float = analysis["worst_crack_mm"]
        survival: ForwardSimPrediction | None = analysis.get("survival_sim")
        tools: list[ToolCall] = analysis["tool_calls"]

        reasons: list[str] = []
        rule_fired: str | None = None
        action: Action
        cost_avoided = 0.0

        # Rule 1: explicit crack-length kill
        if worst_crack_mm > CRACK_KILL_MM:
            action = Action.KILL
            rule_fired = f"crack_length > {CRACK_KILL_MM}mm (JEDEC JESD22-B116)"
            reasons.append(
                f"Worst crack {worst_crack_mm:.2f}mm exceeds kill threshold {CRACK_KILL_MM}mm"
            )
            cost_avoided = analysis.get("cost_avoided_usd", 1847.0)

        # Rule 2: forward sim says it'll die later
        elif survival and survival.fails_at_step is not None:
            action = Action.KILL
            rule_fired = "survival_simulator: predicted catastrophic failure"
            reasons.append(
                f"Forward sim: {survival.failure_reason} at {survival.fails_at_step.value}"
            )
            cost_avoided = survival.cost_avoided_usd

        elif worst_crack_mm > CRACK_FLAG_MM:
            action = Action.FLAG
            rule_fired = f"crack_length > {CRACK_FLAG_MM}mm"
            reasons.append(f"Borderline crack {worst_crack_mm:.2f}mm — tighten downstream inspection")

        else:
            action = Action.PASS_
            reasons.append("All dies within JEDEC JESD22-B116 limits")

        now = datetime.now(timezone.utc)
        return CheckpointResult(
            checkpoint_id=self.checkpoint_id,
            step_name=self.step_name,
            tools_called=tools,
            action=action,
            reasons=reasons,
            rule_fired=rule_fired,
            forward_sim_prediction=survival,
            cost_avoided_usd=cost_avoided,
            started_at=now,
            finished_at=now,
        )
