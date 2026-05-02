"""Tests for physics_adapter — verifies real Person 1 physics calls produce expected shapes."""

from packguard_pipeline import physics_adapter as pa
from packguard_pipeline.models import StepName, ToolType


def test_griffith_smoke():
    tc = pa.griffith(crack_length_mm=0.0)
    assert tc.tool_type == ToolType.DETERMINISTIC
    assert tc.tool_name == "griffith_fracture"
    assert "probability_of_failure" in tc.output
    assert tc.output["probability_of_failure"] == 0.0
    assert "citations" in tc.output
    assert len(tc.output["citations"]) >= 1


def test_griffith_kills_large_crack():
    tc = pa.griffith(crack_length_mm=1.8, applied_stress_MPa=120)
    assert tc.output["probability_of_failure"] == 1.0


def test_coffin_manson_consumer_clean():
    tc = pa.coffin_manson(40, 500, 5)
    assert tc.output["probability_of_failure"] < 0.01
    assert tc.output["units"] == "cycles"


def test_blacks_pecks_arrhenius_consume_inputs():
    for fn, kwargs in [
        (pa.blacks, {"current_density_A_per_cm2": 1e5, "temperature_celsius": 55}),
        (pa.pecks, {"relative_humidity_pct": 40, "temperature_celsius": 55, "msl_rating": 3}),
        (pa.arrhenius_imc, {"wire_bond_temps_C": [150, 175], "wire_bond_times_h": [0.5, 0.25]}),
    ]:
        tc = fn(**kwargs)
        assert tc.output["probability_of_failure"] >= 0.0
        assert tc.output["probability_of_failure"] <= 1.0


def test_survival_sim_returns_steps():
    pred, tc = pa.survival_sim(initial_crack_mm=1.8, profile="automotive")
    assert pred.fails_at_step is not None  # 1.8mm always fails
    assert pred.cost_avoided_usd > 0
    assert len(pred.steps) >= 1
    # Every step has a known StepName
    for step in pred.steps:
        assert isinstance(step.step_name, StepName)


def test_run_all_aggregates_inputs():
    state = {
        "delta_t_celsius": 40,
        "cycles_per_year": 500,
        "service_life_years": 5,
        "current_density_A_per_cm2": 1e5,
        "temperature_celsius": 55,
    }
    results = pa.run_all(state)
    assert "coffin_manson" in results
    assert "blacks_equation" in results
