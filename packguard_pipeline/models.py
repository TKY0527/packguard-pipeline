"""
Pydantic schemas for PackGuard v2.0.

This file is the central data contract. Every other team member depends on it:
- Person 1's physics functions return PhysicsOutput
- Person 3's orchestrator consumes LotState and writes FinalDecision
- Person 4's frontend mirrors these as TypeScript types in lib/types.ts

Run `python -m packguard_pipeline.export_schema` to regenerate the JSON Schema.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------- Enums ----------

class Application(str, Enum):
    AUTOMOTIVE = "automotive"
    SERVER = "server"
    CONSUMER = "consumer"
    INDUSTRIAL = "industrial"


class Action(str, Enum):
    PASS_ = "pass"
    FLAG = "flag"
    KILL = "kill"


class DecisionState(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    PASS_ = "PASS"
    FLAG = "FLAG"
    KILL = "KILL"


class FinalVerdict(str, Enum):
    SHIP = "SHIP"
    HOLD = "HOLD"
    REJECT = "REJECT"


class StepName(str, Enum):
    DICING = "DICING"
    DIE_ATTACH = "DIE_ATTACH"
    WIRE_BOND = "WIRE_BOND"
    MOLDING = "MOLDING"
    REFLOW = "REFLOW"
    TEST = "TEST"
    FINAL_GATE = "FINAL_GATE"


class ToolType(str, Enum):
    DETERMINISTIC = "deterministic"
    AI = "ai"


# ---------- Person 1 contract: physics function output ----------

class PhysicsOutput(BaseModel):
    """
    Standard output of every Person 1 physics function.

    Verified against Person 1's repo (packguard_physics.ReliabilityResult) on Day 1.
    NOTE: Person 1's actual struct has two extra fields beyond API contract §2:
      - inputs (dict): what was passed in, for audit
      - citations (list[str]): JEDEC / textbook references
    Person 2 (this repo) MUST surface these to Person 3's orchestrator —
    citations especially are how we earn judge trust.

    Update API contract §2 to include `inputs` and `citations`.
    """
    probability_of_failure: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    predicted_lifetime: float
    units: str
    model_used: str
    assumptions: list[str]
    inputs: dict[str, Any] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)


# ---------- Vision / AI tool output ----------

class VisionOutput(BaseModel):
    """Output of a CV model (CNN / U-Net / YOLO) or Claude Vision call."""
    detected_class: str  # e.g., "void", "crack", "wire_sweep"
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_boxes: list[list[float]] = Field(default_factory=list)
    measurements: dict[str, float] = Field(default_factory=dict)  # e.g., {"void_ratio": 0.18, "crack_length_mm": 1.8}
    model_used: str
    notes: Optional[str] = None


# ---------- Tool call record ----------

class ToolCall(BaseModel):
    """Records exactly which tool was invoked and what it returned. Required for traceability."""
    tool_name: str
    tool_type: ToolType
    output: dict[str, Any]  # PhysicsOutput.model_dump() or VisionOutput.model_dump()
    confidence: float = Field(ge=0.0, le=1.0)
    runtime_ms: int


# ---------- Forward simulation (killer demo feature) ----------

class ForwardSimStep(BaseModel):
    """One simulated future-step prediction."""
    step_name: StepName
    predicted_state: dict[str, Any]  # e.g., {"crack_length_mm": 2.3}
    will_fail: bool
    failure_mode: Optional[str] = None


class ForwardSimPrediction(BaseModel):
    """Output of the Survival Simulator at a given checkpoint."""
    starting_state: dict[str, Any]  # current measured defect state
    steps: list[ForwardSimStep]
    fails_at_step: Optional[StepName] = None
    failure_reason: Optional[str] = None
    cost_avoided_usd: float = 0.0
    narrative: str  # human-readable, e.g., "Crack will grow to 2.3mm at wire bond, fracture at reflow"


# ---------- Per-checkpoint result ----------

class CheckpointResult(BaseModel):
    """Output of one checkpoint analysis."""
    checkpoint_id: int = Field(ge=1, le=7)
    step_name: StepName
    tools_called: list[ToolCall]
    action: Action
    reasons: list[str]  # human-readable trigger explanations
    rule_fired: Optional[str] = None  # e.g., "Cpk < 1.33", "void_ratio > 25%"
    forward_sim_prediction: Optional[ForwardSimPrediction] = None
    cost_avoided_usd: float = 0.0
    started_at: datetime
    finished_at: datetime


# ---------- Final decision (Person 3 fills this) ----------

class FailureModeProbability(BaseModel):
    failure_mode: str  # e.g., "solder_joint_thermal_fatigue"
    physics_model: str  # e.g., "Coffin-Manson"
    p_fail: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    predicted_lifetime: Optional[float] = None
    units: Optional[str] = None


class DebateLogEntry(BaseModel):
    """One entry in the Debate Protocol audit log."""
    trigger: str  # e.g., "Vision-Process disagreement"
    rule_applied: str  # e.g., "Rule 2: Process beats specification"
    tools_in_conflict: list[str]
    resolution: str
    timestamp: datetime


class FinalDecision(BaseModel):
    """Output of Person 3's orchestrator at Checkpoint 7 (Final Gate)."""
    verdict: FinalVerdict  # SHIP / HOLD / REJECT
    overall_p_fail: float = Field(ge=0.0, le=1.0)
    threshold_used: float
    failure_modes: list[FailureModeProbability]
    debate_log: list[DebateLogEntry] = Field(default_factory=list)
    narrative: str  # LLM-written human-readable report
    recommended_actions: list[str] = Field(default_factory=list)
    pdf_url: Optional[str] = None
    total_cost_avoided_usd: float = 0.0


# ---------- Top-level: LotState ----------

class InputFiles(BaseModel):
    """Files uploaded for analysis. Stored as paths (filesystem) or URLs."""
    xray_images: list[str] = Field(default_factory=list)
    aoi_images: list[str] = Field(default_factory=list)
    reflow_csv: Optional[str] = None
    bond_force_log: Optional[str] = None
    test_data_csv: Optional[str] = None
    material_spec_json: Optional[str] = None


class LotState(BaseModel):
    """
    Top-level lot object. Created by POST /analyze, mutated as it flows through
    the 7-checkpoint pipeline, finalized when Person 3's orchestrator writes
    `final_decision`.
    """
    # Identity
    lot_id: str = Field(pattern=r"^LOT-\d{4}-\d{3,}$")  # LOT-2026-001
    package_type: str  # "BGA-256", "QFN-48", "FCBGA-1234"
    application: Application
    lot_size: int = 4000  # number of chips, default mid of 3000-5000

    # Pipeline state
    current_step: int = Field(ge=0, le=7, default=0)  # 0=intake, 1-7=after checkpoint N
    decision_state: DecisionState = DecisionState.IN_PROGRESS

    # Inputs
    input_files: InputFiles

    # Per-checkpoint findings
    checkpoints: list[CheckpointResult] = Field(default_factory=list)

    # Final decision (filled by Person 3 at Checkpoint 7)
    final_decision: Optional[FinalDecision] = None

    # Audit
    created_at: datetime
    updated_at: datetime


# ---------- API request/response wrappers ----------

class AnalyzeResponse(BaseModel):
    """Response from POST /analyze."""
    lot_id: str
    decision_state: DecisionState
    current_step: int
    message: str
