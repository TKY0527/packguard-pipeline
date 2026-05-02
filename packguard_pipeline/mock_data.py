"""
Demo fixtures for the 3 scripted scenarios.

Brief calls these out explicitly:
  1. Clean lot — sails through all 7 checkpoints
  2. Early kill — bad dicing → kill at Checkpoint 1 (THE killer demo)
  3. Debate trigger — Vision says OK, SPC shows drift, Rule 2 fires at Checkpoint 3

Day 1: every checkpoint's analyze() dispatches by lot_id to one of these
fixtures. Day 2-4: replace each demo_*_analysis() with real CV + real calls
into packguard_physics.

Keep numbers realistic — judges will look at them.
"""

from typing import Any

from .models import (
    ForwardSimPrediction,
    ForwardSimStep,
    LotState,
    StepName,
    ToolCall,
    ToolType,
)

# ============================================================================
# Scenario routing — by lot_id pattern
# ============================================================================

SCENARIO_CLEAN = "clean"
SCENARIO_EARLY_KILL = "early_kill"
SCENARIO_DEBATE = "debate"


def scenario_for(lot: LotState) -> str:
    """Pick a fixture path based on lot_id. Demo-friendly."""
    lid = lot.lot_id.upper()
    if lid.endswith("002") or "KILL" in lid:
        return SCENARIO_EARLY_KILL
    if lid.endswith("003") or "DEBATE" in lid:
        return SCENARIO_DEBATE
    return SCENARIO_CLEAN


# ============================================================================
# Helper — build a PhysicsOutput-shaped dict matching Person 1's schema
# ============================================================================

def physics_output(
    *,
    p_fail: float,
    ci: tuple[float, float],
    lifetime: float,
    units: str,
    model: str,
    assumptions: list[str],
    inputs: dict[str, Any] | None = None,
    citations: list[str] | None = None,
) -> dict[str, Any]:
    """Mirror packguard_physics.ReliabilityResult shape."""
    return {
        "probability_of_failure": p_fail,
        "confidence_interval": list(ci),
        "predicted_lifetime": lifetime,
        "units": units,
        "model_used": model,
        "assumptions": assumptions,
        "inputs": inputs or {},
        "citations": citations or [],
    }


# ============================================================================
# Checkpoint 1 — Dicing
# ============================================================================

def demo_dicing_analysis(lot: LotState) -> dict[str, Any]:
    s = scenario_for(lot)

    if s == SCENARIO_EARLY_KILL:
        # Killer demo: 1.8mm crack predicted to fracture at reflow.
        forward_sim = ForwardSimPrediction(
            starting_state={"crack_length_mm": 1.8},
            steps=[
                ForwardSimStep(
                    step_name=StepName.DIE_ATTACH,
                    predicted_state={"crack_length_mm": 2.1},
                    will_fail=False,
                ),
                ForwardSimStep(
                    step_name=StepName.WIRE_BOND,
                    predicted_state={"crack_length_mm": 2.3},
                    will_fail=False,
                ),
                ForwardSimStep(
                    step_name=StepName.MOLDING,
                    predicted_state={"crack_length_mm": 2.5},
                    will_fail=False,
                ),
                ForwardSimStep(
                    step_name=StepName.REFLOW,
                    predicted_state={"crack_length_mm": 3.1},
                    will_fail=True,
                    failure_mode="catastrophic_fracture_at_thermal_shock",
                ),
            ],
            fails_at_step=StepName.REFLOW,
            failure_reason="Crack reaches critical length under 245°C thermal shock (Griffith)",
            cost_avoided_usd=1847.0,
            narrative=(
                "1.8mm crack will grow to 2.1mm at die attach, 2.3mm at wire bond, "
                "and catastrophically fracture at reflow (245°C thermal shock). "
                "Kill now — saves $1,847 per lot."
            ),
        )
        return {
            "worst_crack_mm": 1.8,
            "survival_sim": forward_sim,
            "cost_avoided_usd": 1847.0,
            "tool_calls": [
                ToolCall(
                    tool_name="edge_chip_classifier",
                    tool_type=ToolType.DETERMINISTIC,
                    output={"max_chip_um": 95, "fail_count": 14, "spec_um": 50},
                    confidence=0.99,
                    runtime_ms=8,
                ),
                ToolCall(
                    tool_name="griffith_fracture",
                    tool_type=ToolType.DETERMINISTIC,
                    output=physics_output(
                        p_fail=0.94,
                        ci=(0.89, 0.97),
                        lifetime=0.0,
                        units="cycles_to_critical_growth",
                        model="Griffith fracture (σ = √(2Eγ/πa))",
                        assumptions=["E=170GPa silicon", "γ=1.0 J/m² fracture energy"],
                        inputs={"crack_length_mm": 1.8, "applied_stress_MPa": 120},
                        citations=["Griffith 1921", "JEDEC JESD22-B116"],
                    ),
                    confidence=0.94,
                    runtime_ms=3,
                ),
                ToolCall(
                    tool_name="survival_simulator",
                    tool_type=ToolType.DETERMINISTIC,
                    output={
                        "fails_at_step": "REFLOW",
                        "trace": [s.model_dump() for s in forward_sim.steps],
                    },
                    confidence=0.92,
                    runtime_ms=42,
                ),
            ],
        }

    if s == SCENARIO_DEBATE:
        # Borderline crack — passes dicing but adds a small risk contribution.
        # Sized so that the lot ends at HOLD (not REJECT) at C7 for server app.
        return {
            "worst_crack_mm": 0.6,
            "survival_sim": None,
            "tool_calls": [
                ToolCall(
                    tool_name="edge_chip_classifier",
                    tool_type=ToolType.DETERMINISTIC,
                    output={"max_chip_um": 38, "fail_count": 0, "spec_um": 50},
                    confidence=0.97,
                    runtime_ms=7,
                ),
                ToolCall(
                    tool_name="griffith_fracture",
                    tool_type=ToolType.DETERMINISTIC,
                    output=physics_output(
                        p_fail=5e-5,
                        ci=(2e-5, 1e-4),
                        lifetime=10000.0,
                        units="cycles",
                        model="Griffith fracture",
                        assumptions=["sub-critical crack length"],
                        citations=["JEDEC JESD22-B116"],
                    ),
                    confidence=0.98,
                    runtime_ms=2,
                ),
            ],
        }

    # Clean lot — per-mode P(fail) deliberately tiny so aggregate < automotive
    # threshold (1e-5). Replace with real Person 1 outputs in Day 2-4.
    return {
        "worst_crack_mm": 0.3,
        "survival_sim": None,
        "tool_calls": [
            ToolCall(
                tool_name="edge_chip_classifier",
                tool_type=ToolType.DETERMINISTIC,
                output={"max_chip_um": 22, "fail_count": 0, "spec_um": 50},
                confidence=0.99,
                runtime_ms=6,
            ),
            ToolCall(
                tool_name="griffith_fracture",
                tool_type=ToolType.DETERMINISTIC,
                output=physics_output(
                    p_fail=1e-7,
                    ci=(5e-8, 2e-7),
                    lifetime=50000.0,
                    units="cycles",
                    model="Griffith fracture",
                    assumptions=["clean dies, σ << σ_critical"],
                    citations=["JEDEC JESD22-B116"],
                ),
                confidence=0.99,
                runtime_ms=2,
            ),
        ],
    }


# ============================================================================
# Checkpoint 2 — Die Attach
# ============================================================================

def demo_die_attach_analysis(lot: LotState) -> dict[str, Any]:
    s = scenario_for(lot)

    if s == SCENARIO_DEBATE:
        return {
            "void_ratio": 0.12,
            "is_clustered": False,
            "post_reflow_rupture": False,
            "junction_temp_exceeds_limit": False,
            "tool_calls": _da_tools(void_ratio=0.12, p_fail=5e-5),
        }

    if s == SCENARIO_EARLY_KILL:
        # Doesn't run — pipeline already KILLed at C1. But provide for completeness.
        return {
            "void_ratio": 0.05,
            "is_clustered": False,
            "post_reflow_rupture": False,
            "junction_temp_exceeds_limit": False,
            "tool_calls": _da_tools(void_ratio=0.05, p_fail=1e-7),
        }

    # Clean
    return {
        "void_ratio": 0.08,
        "is_clustered": False,
        "post_reflow_rupture": False,
        "junction_temp_exceeds_limit": False,
        "tool_calls": _da_tools(void_ratio=0.08, p_fail=3e-7),
    }


def _da_tools(*, void_ratio: float, p_fail: float) -> list[ToolCall]:
    return [
        ToolCall(
            tool_name="void_ratio_calculator",
            tool_type=ToolType.DETERMINISTIC,
            output={"void_ratio": void_ratio, "method": "U-Net segmentation + area ratio"},
            confidence=0.96,
            runtime_ms=120,
        ),
        ToolCall(
            tool_name="thermal_resistance_estimator",
            tool_type=ToolType.DETERMINISTIC,
            output=physics_output(
                p_fail=p_fail,
                ci=(max(0.0, p_fail * 0.5), p_fail * 2.0),
                lifetime=15.0,
                units="years",
                model="R_th_eff = R_th_nominal × (1/(1-void_fraction))^k",
                assumptions=["k=1.4 dispersed voids", "T_amb=70°C"],
                inputs={"void_fraction": void_ratio},
                citations=["JEDEC JESD51 series"],
            ),
            confidence=0.95,
            runtime_ms=4,
        ),
        ToolCall(
            tool_name="post_reflow_void_predictor",
            tool_type=ToolType.DETERMINISTIC,
            output={"will_rupture": False, "delta_volume_pct": 74.2, "model": "PV=nRT"},
            confidence=0.93,
            runtime_ms=5,
        ),
    ]


# ============================================================================
# Checkpoint 3 — Wire Bond (Debate Trigger lives here)
# ============================================================================

def demo_wire_bond_analysis(lot: LotState) -> dict[str, Any]:
    s = scenario_for(lot)

    if s == SCENARIO_DEBATE:
        # Vision says OK — but SPC shows Cpk=1.21 with 2.4σ drift.
        # This fires Debate Protocol Rule 2: Process beats specification.
        return {
            "cpk": 1.21,
            "predicted_imc_um": 3.2,
            "sweep_deflection_pct": 0.04,
            "vision_says_ok": True,
            "sigma_drift": 2.4,
            "tool_calls": _wb_tools(cpk=1.21, imc=3.2, sigma=2.4),
        }

    # Clean / early-kill (the latter never runs)
    return {
        "cpk": 1.78,
        "predicted_imc_um": 2.4,
        "sweep_deflection_pct": 0.02,
        "vision_says_ok": True,
        "sigma_drift": 0.6,
        "tool_calls": _wb_tools(cpk=1.78, imc=2.4, sigma=0.6),
    }


def _wb_tools(*, cpk: float, imc: float, sigma: float) -> list[ToolCall]:
    return [
        ToolCall(
            tool_name="bond_pull_shear_spc",
            tool_type=ToolType.DETERMINISTIC,
            output={
                "cpk": cpk,
                "mean_grams": 12.4,
                "sigma_drift": sigma,
                "western_electric_violations": 1 if sigma > 2.0 else 0,
            },
            confidence=0.99,
            runtime_ms=3,
        ),
        ToolCall(
            tool_name="arrhenius_imc",
            tool_type=ToolType.DETERMINISTIC,
            output=physics_output(
                p_fail=(imc / 5.0) ** 4 * 1e-6,
                ci=((imc / 5.0) ** 4 * 5e-7, (imc / 5.0) ** 4 * 5e-6),
                lifetime=imc,
                units="micrometers",
                model="Arrhenius IMC growth (x = √(D₀ exp(-Eₐ/kT) t))",
                assumptions=["Cu/Al system", "Tj=125°C continuous"],
                inputs={"temperature_C": 125, "service_years": 10},
                citations=["Kirkendall 1947", "JEDEC JESD22-A104"],
            ),
            confidence=0.94,
            runtime_ms=6,
        ),
        ToolCall(
            tool_name="vision_wire_sweep",
            tool_type=ToolType.AI,
            output={"detected_anomalies": 0, "model": "YOLOv8n-finetuned"},
            confidence=0.91,
            runtime_ms=180,
        ),
    ]


# ============================================================================
# Checkpoint 4 — Molding
# ============================================================================

def demo_molding_analysis(lot: LotState) -> dict[str, Any]:
    s = scenario_for(lot)
    if s == SCENARIO_DEBATE:
        return {
            "wire_deflection_pct": 0.05,
            "interface_stress_ratio": 0.62,
            "tool_calls": _molding_tools(deflect=0.05, stress=0.62),
        }
    return {
        "wire_deflection_pct": 0.03,
        "interface_stress_ratio": 0.40,
        "tool_calls": _molding_tools(deflect=0.03, stress=0.40),
    }


def _molding_tools(*, deflect: float, stress: float) -> list[ToolCall]:
    sweep_p = deflect * 2e-5
    cure_p = max(1e-7, stress * 1e-6)
    return [
        ToolCall(
            tool_name="wire_sweep_calculator",
            tool_type=ToolType.DETERMINISTIC,
            output=physics_output(
                p_fail=sweep_p,
                ci=(sweep_p * 0.5, sweep_p * 2.0),
                lifetime=0.0,
                units="deflection_fraction",
                model="Wire sweep (fluid dynamics)",
                assumptions=["resin viscosity 200 Pa·s", "fill velocity 1 m/s"],
                citations=["Pecht 1995"],
            ),
            confidence=0.97,
            runtime_ms=12,
        ),
        ToolCall(
            tool_name="cure_shrinkage_stress",
            tool_type=ToolType.DETERMINISTIC,
            output=physics_output(
                p_fail=cure_p,
                ci=(cure_p * 0.5, cure_p * 2.0),
                lifetime=10.0,
                units="years",
                model="Cure shrinkage stress",
                assumptions=["shrink=0.3%", "EMC CTE=15 ppm/K"],
                citations=["JEDEC JEP150"],
            ),
            confidence=0.95,
            runtime_ms=8,
        ),
    ]


# ============================================================================
# Checkpoint 5 — Reflow
# ============================================================================

def demo_reflow_analysis(lot: LotState) -> dict[str, Any]:
    s = scenario_for(lot)

    # Service life cycles vary by application
    service_life_cycles = {
        "automotive": 15000,  # 15 years × 1000 cycles/year
        "server": 7000,
        "consumer": 300,
        "industrial": 10000,
    }[lot.application.value]

    if s == SCENARIO_DEBATE:
        # Lifetime borderline
        return {
            "coffin_manson_nf": int(service_life_cycles * 1.4),
            "service_life_cycles": service_life_cycles,
            "warpage_um": 65.0,
            "warpage_spec_um": 100.0,
            "popcorn_risk": 0.08,
            "tool_calls": _reflow_tools(nf=int(service_life_cycles * 1.4), warpage=65.0),
        }

    # Clean — comfortable margin
    return {
        "coffin_manson_nf": int(service_life_cycles * 3.5),
        "service_life_cycles": service_life_cycles,
        "warpage_um": 42.0,
        "warpage_spec_um": 100.0,
        "popcorn_risk": 0.02,
        "tool_calls": _reflow_tools(nf=int(service_life_cycles * 3.5), warpage=42.0),
    }


def _reflow_tools(*, nf: int, warpage: float) -> list[ToolCall]:
    return [
        ToolCall(
            tool_name="coffin_manson",
            tool_type=ToolType.DETERMINISTIC,
            output=physics_output(
                p_fail=5e-7,
                ci=(2e-7, 1.2e-6),
                lifetime=float(nf),
                units="cycles",
                model="Coffin-Manson Nf = C × ΔT^-n",
                assumptions=["SAC305 solder", "n=2.0", "C=2200 (Engelmaier)"],
                inputs={"delta_T": 100, "cycles_per_year": 1000},
                citations=["Engelmaier 1983", "IPC-9701"],
            ),
            confidence=0.96,
            runtime_ms=4,
        ),
        ToolCall(
            tool_name="blacks_equation",
            tool_type=ToolType.DETERMINISTIC,
            output=physics_output(
                p_fail=3e-7,
                ci=(1e-7, 8e-7),
                lifetime=12.0,
                units="years",
                model="Black's equation MTTF = A × J^-n × exp(Ea/kT)",
                assumptions=["Cu interconnect", "n=2", "Ea=0.9 eV"],
                citations=["Black 1969", "JEDEC JEP119"],
            ),
            confidence=0.92,
            runtime_ms=3,
        ),
        ToolCall(
            tool_name="pecks_model",
            tool_type=ToolType.DETERMINISTIC,
            output=physics_output(
                p_fail=2e-7,
                ci=(1e-7, 5e-7),
                lifetime=11.0,
                units="years",
                model="Peck's TTF ∝ RH^-n × exp(Ea/kT)",
                assumptions=["RH=85%", "Ea=0.7 eV", "n=2.7"],
                citations=["Peck 1986", "JEDEC JESD22-A101"],
            ),
            confidence=0.91,
            runtime_ms=3,
        ),
        ToolCall(
            tool_name="warpage_calculator",
            tool_type=ToolType.DETERMINISTIC,
            output=physics_output(
                p_fail=1e-7,
                ci=(5e-8, 2e-7),
                lifetime=warpage,
                units="micrometers",
                model="Bimetallic CTE-mismatch warpage",
                assumptions=["substrate CTE=18 ppm/K", "die CTE=2.6 ppm/K"],
                citations=["Timoshenko 1925"],
            ),
            confidence=0.94,
            runtime_ms=5,
        ),
    ]


# ============================================================================
# Checkpoint 6 — Test
# ============================================================================

def demo_test_analysis(lot: LotState) -> dict[str, Any]:
    s = scenario_for(lot)
    if s == SCENARIO_DEBATE:
        return {
            "weibull_beta": 0.84,
            "weibull_eta": 4200.0,
            "critical_spc_violation": False,
            "tool_calls": _test_tools(beta=0.84),
        }
    return {
        "weibull_beta": 1.45,
        "weibull_eta": 7200.0,
        "critical_spc_violation": False,
        "tool_calls": _test_tools(beta=1.45),
    }


def _test_tools(*, beta: float) -> list[ToolCall]:
    p = 4e-7 if beta > 1.0 else 2e-5  # infant mortality rises P(fail) ~50x
    return [
        ToolCall(
            tool_name="weibull_fit",
            tool_type=ToolType.DETERMINISTIC,
            output=physics_output(
                p_fail=p,
                ci=(p * 0.5, p * 5.0),
                lifetime=7000.0,
                units="hours",
                model="Weibull MLE fit",
                assumptions=["right-censored", "uncensored fails fitted"],
                citations=["JEDEC JESD91", "Nelson 1982"],
            ),
            confidence=0.95,
            runtime_ms=15,
        ),
        ToolCall(
            tool_name="western_electric_spc",
            tool_type=ToolType.DETERMINISTIC,
            output={"violations": 0, "rules_evaluated": 4},
            confidence=0.99,
            runtime_ms=2,
        ),
    ]
