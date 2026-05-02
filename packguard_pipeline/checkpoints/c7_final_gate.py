"""
Checkpoint 7 — Final Gate.

This is the only checkpoint where Person 3's orchestrator LLM is meant to write
the human-readable narrative. From Person 2's side, the role is:

1. Aggregate per-failure-mode P(fail) from previous checkpoints' physics outputs
2. Apply application-specific threshold engine
3. Apply Debate Protocol Rules Engine when tools disagreed earlier
4. Hand the lot off to Person 3's /report endpoint

For Day 1: this stub computes the aggregate P(fail) and writes a placeholder
FinalDecision so the frontend can show *something*. Person 3 will overwrite
final_decision.narrative once the orchestrator service is wired in.
"""

from datetime import datetime, timezone
from typing import Any

from ..models import (
    Action,
    Application,
    CheckpointResult,
    FailureModeProbability,
    FinalDecision,
    FinalVerdict,
    LotState,
    StepName,
    ToolCall,
    ToolType,
)
from ..pipeline import Checkpoint

# §5.3 cost-of-quality thresholds (DPPM expressed as P(fail) for hackathon clarity)
APP_THRESHOLDS: dict[Application, float] = {
    Application.AUTOMOTIVE: 0.00001,   # < 10 DPPM
    Application.SERVER: 0.0001,        # < 100 DPPM
    Application.CONSUMER: 0.001,       # < 1000 DPPM
    Application.INDUSTRIAL: 0.00005,   # < 50 DPPM
}


# Only LIFETIME / operational physics models contribute to lot-level P(fail).
# One-shot manufacturing-event models (warpage, wire_sweep, griffith_fracture)
# either already KILLed at their checkpoint (so we wouldn't be here) or are
# orthogonal to lifetime risk. Calibration models (weibull_fit) report a
# distribution shape, not a probability.
_LIFETIME_MODELS = {
    "coffin_manson",
    "blacks_equation",
    "pecks_model",
    "arrhenius_imc",
    "void_thermal_resistance",
}


def _aggregate_p_fail(lot: LotState) -> tuple[float, list[FailureModeProbability]]:
    """
    Implement P(any failure) = 1 - prod(1 - P_i) over LIFETIME physics modes.

    Brief §5: per-failure-mode probabilities for solder fatigue, electromigration,
    humidity-driven corrosion, IMC growth, etc. — the modes that quantify a
    *lifetime* risk over service. Reflow-event and calibration models are
    intentionally excluded from this aggregate.
    """
    modes: list[FailureModeProbability] = []
    survivors = 1.0

    for cp in lot.checkpoints:
        for tc in cp.tools_called:
            if tc.tool_type != ToolType.DETERMINISTIC:
                continue
            out = tc.output
            if "probability_of_failure" not in out:
                continue
            model = out.get("model_used", tc.tool_name)
            if model not in _LIFETIME_MODELS:
                continue
            p_fail = float(out["probability_of_failure"])
            modes.append(
                FailureModeProbability(
                    failure_mode=model,
                    physics_model=model,
                    p_fail=p_fail,
                    confidence_interval=tuple(out.get("confidence_interval", (max(0.0, p_fail - 0.01), min(1.0, p_fail + 0.01)))),
                    predicted_lifetime=out.get("predicted_lifetime"),
                    units=out.get("units"),
                )
            )
            survivors *= (1.0 - p_fail)

    overall = 1.0 - survivors
    return overall, modes


class FinalGateCheckpoint(Checkpoint):
    checkpoint_id = 7
    step_name = StepName.FINAL_GATE

    def analyze(self, lot: LotState) -> dict[str, Any]:
        # No external tools at the gate — just aggregation.
        overall_p, modes = _aggregate_p_fail(lot)
        threshold = APP_THRESHOLDS[lot.application]
        return {
            "overall_p_fail": overall_p,
            "failure_modes": modes,
            "threshold": threshold,
            "tool_calls": [
                ToolCall(
                    tool_name="per_failure_mode_aggregator",
                    tool_type=ToolType.DETERMINISTIC,
                    output={"overall_p_fail": overall_p, "n_modes": len(modes)},
                    confidence=1.0,
                    runtime_ms=1,
                ),
                ToolCall(
                    tool_name="application_threshold_engine",
                    tool_type=ToolType.DETERMINISTIC,
                    output={"application": lot.application.value, "threshold": threshold},
                    confidence=1.0,
                    runtime_ms=1,
                ),
            ],
        }

    def decide(self, lot: LotState, analysis: dict[str, Any]) -> CheckpointResult:
        overall_p: float = analysis["overall_p_fail"]
        modes: list[FailureModeProbability] = analysis["failure_modes"]
        threshold: float = analysis["threshold"]

        # If any earlier checkpoint already KILLed, we wouldn't be here. So this
        # is just SHIP vs HOLD vs (rare) REJECT based on aggregate.
        if overall_p < threshold:
            verdict = FinalVerdict.SHIP
            action = Action.PASS_
            reasons = [
                f"Aggregate P(fail) {overall_p:.5f} < threshold {threshold:.5f} for {lot.application.value}"
            ]
        elif overall_p < threshold * 10:
            verdict = FinalVerdict.HOLD
            action = Action.FLAG
            reasons = [
                f"Aggregate P(fail) {overall_p:.5f} > threshold — recommend selective inspection"
            ]
        else:
            verdict = FinalVerdict.REJECT
            action = Action.KILL
            reasons = [f"Aggregate P(fail) {overall_p:.5f} >> threshold {threshold:.5f}"]

        # Compute lot-wide cost saved by KILLs upstream.
        total_saved = sum(cp.cost_avoided_usd for cp in lot.checkpoints)

        # Day 1 placeholder narrative — Person 3 will overwrite via /report.
        narrative = (
            f"Lot {lot.lot_id} ({lot.package_type}, {lot.application.value}) "
            f"completed pipeline. Overall P(fail)={overall_p:.5f} vs threshold={threshold:.5f}. "
            f"Verdict: {verdict.value}. (Awaiting orchestrator narrative.)"
        )

        # Write placeholder final_decision so frontend can render. Person 3 mutates.
        lot.final_decision = FinalDecision(
            verdict=verdict,
            overall_p_fail=overall_p,
            threshold_used=threshold,
            failure_modes=modes,
            debate_log=[],  # Person 3 fills
            narrative=narrative,
            recommended_actions=[],
            total_cost_avoided_usd=total_saved,
        )

        now = datetime.now(timezone.utc)
        return CheckpointResult(
            checkpoint_id=self.checkpoint_id,
            step_name=self.step_name,
            tools_called=analysis["tool_calls"],
            action=action,
            reasons=reasons,
            rule_fired=f"verdict_engine: {verdict.value}",
            cost_avoided_usd=0.0,  # accumulated cost is on FinalDecision, not here
            started_at=now,
            finished_at=now,
        )
