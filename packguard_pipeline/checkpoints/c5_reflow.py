"""
Checkpoint 5 — After Reflow.

The most physics-rich checkpoint. Five separate models run in parallel:

- DET: Coffin-Manson (solder fatigue Nf)
- DET: Black's equation (electromigration MTTF)
- DET: Peck's model (humidity TTF)
- DET: dT/dt rate-of-change + FFT thermal analysis
- DET: Warpage calculator
- AI: Vision (X-ray solder joints — head-in-pillow, voids)

Decision rule:
- predicted Nf < customer service-life cycles → KILL
- warpage exceeds spec → KILL
- popcorn risk high → KILL
- borderline lifetime → FLAG
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
from ..mock_data import demo_reflow_analysis


class ReflowCheckpoint(Checkpoint):
    checkpoint_id = 5
    step_name = StepName.REFLOW

    def analyze(self, lot: LotState) -> dict[str, Any]:
        return demo_reflow_analysis(lot)

    def decide(self, lot: LotState, analysis: dict[str, Any]) -> CheckpointResult:
        nf_predicted: int = analysis["coffin_manson_nf"]
        nf_required: int = analysis["service_life_cycles"]
        warpage_um: float = analysis["warpage_um"]
        warpage_spec_um: float = analysis["warpage_spec_um"]
        popcorn_risk: float = analysis["popcorn_risk"]
        tools: list[ToolCall] = analysis["tool_calls"]

        reasons: list[str] = []
        rule_fired: str | None = None
        action: Action
        cost_avoided = 0.0

        if nf_predicted < nf_required:
            action = Action.KILL
            rule_fired = "Coffin-Manson Nf < service-life cycles"
            reasons.append(
                f"Predicted {nf_predicted:,} cycles < required {nf_required:,} — solder fatigue"
            )
            cost_avoided = analysis.get("cost_avoided_usd", 5800.0)

        elif warpage_um > warpage_spec_um:
            action = Action.KILL
            rule_fired = "warpage > spec"
            reasons.append(f"Warpage {warpage_um:.1f}µm exceeds spec {warpage_spec_um:.1f}µm")
            cost_avoided = analysis.get("cost_avoided_usd", 5800.0)

        elif popcorn_risk > 0.5:
            action = Action.KILL
            rule_fired = "popcorn_risk > 50%"
            reasons.append(f"Popcorn cracking risk {popcorn_risk:.0%} — moisture-driven fail at reflow")
            cost_avoided = analysis.get("cost_avoided_usd", 5800.0)

        elif nf_predicted < int(nf_required * 1.2):
            action = Action.FLAG
            rule_fired = "Nf within 20% of margin"
            reasons.append(f"Borderline lifetime: Nf {nf_predicted:,} near required {nf_required:,}")

        else:
            action = Action.PASS_
            reasons.append(
                f"Nf {nf_predicted:,} >> required {nf_required:,}; warpage and humidity within spec"
            )

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
