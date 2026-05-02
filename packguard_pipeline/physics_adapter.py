"""
Adapter between Person 2's pipeline and Person 1's `packguard_physics` package.

Two responsibilities:

1. **Per-checkpoint typed wrappers** — small functions that call individual
   physics modules with named, scenario-controlled parameters and return a
   `ToolCall` ready to drop into a `CheckpointResult.tools_called` list.

2. **`ReliabilityResult` → `dict` conversion** — Person 1's result is a Pydantic
   v2 BaseModel (frozen=True). `model_dump(mode="json")` gives us a plain dict
   shaped exactly like our `PhysicsOutput`.

Why this layer exists: the brief insists deterministic physics owns decisions.
Our `mock_data.py` needs to call REAL physics so judges can audit the numbers,
but the checkpoint logic should not import packguard_physics directly — that
would couple the pipeline to one specific physics version.
"""

from __future__ import annotations

import time
from typing import Any

from packguard_physics import ReliabilityResult
from packguard_physics.api import run_all_models, run_survival_simulation
from packguard_physics.griffith_fracture import assess_crack_growth
from packguard_physics.coffin_manson import predict_solder_fatigue
from packguard_physics.blacks_equation import predict_electromigration
from packguard_physics.pecks_model import predict_humidity_failure
from packguard_physics.arrhenius_imc import predict_imc_thickness
from packguard_physics.void_thermal_resistance import assess_void_impact
from packguard_physics.wire_sweep import predict_wire_sweep
from packguard_physics.warpage import predict_warpage
from packguard_physics.weibull_fit import fit_weibull
from packguard_physics.survival_simulator import simulate_defect, SimulationTrace

from .models import (
    ForwardSimPrediction,
    ForwardSimStep,
    StepName,
    ToolCall,
    ToolType,
)


# ---------- ReliabilityResult → dict ----------

def reliability_to_dict(r: ReliabilityResult) -> dict[str, Any]:
    """Convert Pydantic ReliabilityResult to a plain JSON-safe dict."""
    return r.model_dump(mode="json")


# ---------- Timed call helpers ----------

def _call(fn, **kwargs) -> tuple[ReliabilityResult, int]:
    """Run a physics fn, return result + elapsed milliseconds."""
    t0 = time.perf_counter()
    r = fn(**kwargs)
    elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))
    return r, elapsed_ms


def physics_tool_call(
    *,
    tool_name: str,
    fn,
    **fn_kwargs,
) -> ToolCall:
    """Call a physics function and wrap result in a ToolCall."""
    r, ms = _call(fn, **fn_kwargs)
    return ToolCall(
        tool_name=tool_name,
        tool_type=ToolType.DETERMINISTIC,
        output=reliability_to_dict(r),
        confidence=1.0 - abs(r.confidence_interval[1] - r.confidence_interval[0]) / 2.0,
        runtime_ms=ms,
    )


# ---------- Per-checkpoint convenience wrappers ----------

def griffith(crack_length_mm: float, applied_stress_MPa: float = 80) -> ToolCall:
    return physics_tool_call(
        tool_name="griffith_fracture",
        fn=assess_crack_growth,
        crack_length_mm=crack_length_mm,
        applied_stress_MPa=applied_stress_MPa,
    )


def coffin_manson(
    delta_t_celsius: float,
    cycles_per_year: float,
    service_life_years: float,
    solder_alloy: str = "SAC305",
) -> ToolCall:
    return physics_tool_call(
        tool_name="coffin_manson",
        fn=predict_solder_fatigue,
        delta_t_celsius=delta_t_celsius,
        cycles_per_year=cycles_per_year,
        service_life_years=service_life_years,
        solder_alloy=solder_alloy,
    )


def blacks(
    current_density_A_per_cm2: float,
    temperature_celsius: float = 85.0,
    conductor_material: str = "Cu",
    service_life_years: float = 10.0,
) -> ToolCall:
    return physics_tool_call(
        tool_name="blacks_equation",
        fn=predict_electromigration,
        current_density_A_per_cm2=current_density_A_per_cm2,
        temperature_celsius=temperature_celsius,
        conductor_material=conductor_material,
        service_life_years=service_life_years,
    )


def pecks(
    relative_humidity_pct: float,
    temperature_celsius: float = 85.0,
    msl_rating: int = 3,
    service_life_years: float = 10.0,
) -> ToolCall:
    return physics_tool_call(
        tool_name="pecks_model",
        fn=predict_humidity_failure,
        relative_humidity_pct=relative_humidity_pct,
        temperature_celsius=temperature_celsius,
        msl_rating=msl_rating,
        service_life_years=service_life_years,
    )


def arrhenius_imc(
    wire_bond_temps_C: list[float],
    wire_bond_times_h: list[float],
    wire_pad_metallurgy: str = "Cu-Al",
) -> ToolCall:
    return physics_tool_call(
        tool_name="arrhenius_imc",
        fn=predict_imc_thickness,
        temperature_history_celsius=wire_bond_temps_C,
        time_at_temperature_hours=wire_bond_times_h,
        wire_pad_metallurgy=wire_pad_metallurgy,
    )


def void_impact(
    void_fraction: float,
    void_distribution: str = "dispersed",
    nominal_thermal_resistance_K_per_W: float = 1.0,
    max_junction_temp_C: float = 125.0,
    ambient_temp_C: float = 70.0,
    power_dissipation_W: float = 5.0,
) -> ToolCall:
    return physics_tool_call(
        tool_name="void_thermal_resistance",
        fn=assess_void_impact,
        void_fraction=void_fraction,
        void_distribution=void_distribution,
        nominal_thermal_resistance_K_per_W=nominal_thermal_resistance_K_per_W,
        max_junction_temp_C=max_junction_temp_C,
        ambient_temp_C=ambient_temp_C,
        power_dissipation_W=power_dissipation_W,
    )


def wire_sweep(
    wire_length_mm: float = 2.0,
    wire_diameter_um: float = 25.0,
    resin_viscosity_Pa_s: float = 5.0,
    fill_velocity_m_per_s: float = 0.05,
    wire_pitch_mm: float = 0.1,
) -> ToolCall:
    return physics_tool_call(
        tool_name="wire_sweep",
        fn=predict_wire_sweep,
        wire_length_mm=wire_length_mm,
        wire_diameter_um=wire_diameter_um,
        resin_viscosity_Pa_s=resin_viscosity_Pa_s,
        fill_velocity_m_per_s=fill_velocity_m_per_s,
        wire_pitch_mm=wire_pitch_mm,
    )


def warpage(
    die_cte_ppm_per_C: float = 2.6,
    substrate_cte_ppm_per_C: float = 18.0,
    package_size_mm: float = 15.0,
    peak_reflow_temp_C: float = 245.0,
    package_thickness_mm: float = 1.0,
) -> ToolCall:
    return physics_tool_call(
        tool_name="warpage",
        fn=predict_warpage,
        die_cte_ppm_per_C=die_cte_ppm_per_C,
        substrate_cte_ppm_per_C=substrate_cte_ppm_per_C,
        package_size_mm=package_size_mm,
        peak_reflow_temp_C=peak_reflow_temp_C,
        package_thickness_mm=package_thickness_mm,
    )


def weibull(time_to_failure_hours: list[float], censored: list[bool] | None = None) -> ToolCall:
    return physics_tool_call(
        tool_name="weibull_fit",
        fn=fit_weibull,
        time_to_failure_hours=time_to_failure_hours,
        censored=censored,
    )


# ---------- Forward simulation ----------

_SIM_STEP_TO_OUR_STEP: dict[str, StepName] = {
    "die_attach":    StepName.DIE_ATTACH,
    "wire_bond":     StepName.WIRE_BOND,
    "molding":       StepName.MOLDING,
    "reflow":        StepName.REFLOW,
    "burn_in":       StepName.TEST,
    "field_service": StepName.FINAL_GATE,  # closest match
}


def survival_sim(
    initial_crack_mm: float,
    profile: str = "automotive",
    solder_alloy: str = "SAC305",
    cost_per_lot_usd: float = 10000.0,
) -> tuple[ForwardSimPrediction, ToolCall]:
    """
    Run Person 1's survival simulator and adapt it to our ForwardSimPrediction.

    Returns (forward_sim_prediction, tool_call_record).
    """
    t0 = time.perf_counter()
    trace: SimulationTrace = simulate_defect(
        initial_crack_mm=initial_crack_mm,
        profile=profile,
        solder_alloy=solder_alloy,
    )
    elapsed_ms = max(1, int((time.perf_counter() - t0) * 1000))

    # Build our ForwardSimStep list — only manufacturing steps (skip burn_in / field).
    steps: list[ForwardSimStep] = []
    fail_step_name: StepName | None = None
    failure_reason: str | None = None

    for s in trace.steps:
        our_step = _SIM_STEP_TO_OUR_STEP.get(s.step_name)
        if our_step is None:
            continue
        steps.append(
            ForwardSimStep(
                step_name=our_step,
                predicted_state={
                    s.metric_name: s.metric_value,
                    "p_fail": s.p_fail,
                },
                will_fail=s.defect_killed,
                failure_mode=s.notes if s.defect_killed else None,
            )
        )
        if s.defect_killed and fail_step_name is None:
            fail_step_name = our_step
            failure_reason = (
                f"Crack reaches critical state at {s.step_name} "
                f"(P(fail)={s.p_fail:.3f}, {s.metric_name}={s.metric_value:.3f} {s.metric_units})"
            )

    cost_avoided = cost_per_lot_usd if (not trace.survived) else 0.0

    if not trace.survived:
        narrative = (
            f"{initial_crack_mm}mm crack on {profile} profile: forward simulation "
            f"predicts catastrophic failure at {trace.kill_step}. "
            f"Killing now saves ${cost_avoided:,.0f} per lot (full downstream processing cost)."
        )
    else:
        narrative = (
            f"{initial_crack_mm}mm crack on {profile} profile: forward simulation "
            f"shows the defect survives all manufacturing steps. No early kill needed."
        )

    prediction = ForwardSimPrediction(
        starting_state={"crack_length_mm": initial_crack_mm},
        steps=steps,
        fails_at_step=fail_step_name,
        failure_reason=failure_reason,
        cost_avoided_usd=cost_avoided,
        narrative=narrative,
    )

    tool_call = ToolCall(
        tool_name="survival_simulator",
        tool_type=ToolType.DETERMINISTIC,
        output={
            "survived": trace.survived,
            "kill_step": trace.kill_step,
            "n_steps": len(trace.steps),
            "profile": profile,
        },
        confidence=0.95,
        runtime_ms=elapsed_ms,
    )

    return prediction, tool_call


# ---------- Bulk run via Person 1's api.run_all_models ----------

def run_all(lot_state: dict[str, Any]) -> dict[str, ToolCall]:
    """
    Call Person 1's `api.run_all_models` with a flat dict and convert each
    ReliabilityResult into a ToolCall. Useful when a checkpoint wants all
    applicable models at once.
    """
    raw = run_all_models(lot_state)
    out: dict[str, ToolCall] = {}
    for name, r in raw.items():
        out[name] = ToolCall(
            tool_name=name,
            tool_type=ToolType.DETERMINISTIC,
            output=reliability_to_dict(r),
            confidence=1.0 - abs(r.confidence_interval[1] - r.confidence_interval[0]) / 2.0,
            runtime_ms=1,
        )
    return out


__all__ = [
    "griffith",
    "coffin_manson",
    "blacks",
    "pecks",
    "arrhenius_imc",
    "void_impact",
    "wire_sweep",
    "warpage",
    "weibull",
    "survival_sim",
    "run_all",
    "reliability_to_dict",
]
