"""
CheckpointPipeline framework.

Each of the 7 production steps is a Checkpoint subclass that implements
analyze() and decide(). The pipeline runs them in order, mutating the LotState
as it goes. Early KILL short-circuits the rest of the pipeline — that's the
inline-QC value proposition.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from .models import (
    Action,
    CheckpointResult,
    DecisionState,
    LotState,
    StepName,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Checkpoint(ABC):
    """
    Abstract base. One subclass per production step.

    analyze(): call physics + CV tools, return raw findings.
    decide(): apply pass/flag/kill rules to those findings, build a CheckpointResult.

    Splitting analyze/decide is intentional: it keeps decision rules visible
    (auditable) and separable from tool plumbing (testable).
    """

    checkpoint_id: int
    step_name: StepName

    @abstractmethod
    def analyze(self, lot: LotState) -> dict[str, Any]:
        """Run physics + CV tools, return raw analysis dict."""

    @abstractmethod
    def decide(self, lot: LotState, analysis: dict[str, Any]) -> CheckpointResult:
        """Apply deterministic decision rules to analysis. Build CheckpointResult."""

    def run(self, lot: LotState) -> CheckpointResult:
        started = _now()
        analysis = self.analyze(lot)
        result = self.decide(lot, analysis)
        # Force timestamps to be set even if subclass forgot.
        if not hasattr(result, "started_at") or result.started_at is None:
            result.started_at = started
        if not hasattr(result, "finished_at") or result.finished_at is None:
            result.finished_at = _now()
        return result


class CheckpointPipeline:
    """Runs an ordered list of checkpoints over a LotState. Stops on KILL."""

    def __init__(self, checkpoints: list[Checkpoint]) -> None:
        self.checkpoints = checkpoints

    def run(self, lot: LotState) -> LotState:
        lot.decision_state = DecisionState.IN_PROGRESS
        for cp in self.checkpoints:
            result = cp.run(lot)
            lot.checkpoints.append(result)
            lot.current_step = cp.checkpoint_id
            lot.updated_at = _now()

            if result.action == Action.KILL:
                lot.decision_state = DecisionState.KILL
                # KILL short-circuits the rest of the pipeline — saves $$$.
                return lot

            if result.action == Action.FLAG:
                # FLAG continues but marks the lot.
                lot.decision_state = DecisionState.FLAG

        # If we got here without KILL, lot ran to completion.
        # Final state may be PASS or FLAG depending on what fired along the way.
        if lot.decision_state == DecisionState.IN_PROGRESS:
            lot.decision_state = DecisionState.PASS_
        return lot
