"""Policy-level tests for the C6 Random Exploration floor.

Each test mirrors one mandatory condition of the C6 spec (section 5). The same
functions are executed by the c6-random stage preflight; these pytest wrappers
additionally prove the failures are visible to the standard test suite.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import c6_stage  # noqa: E402
from c6_stage import (  # noqa: E402
    tiny_dataset,
    random_sim,
    _run_random,
    count_invalid_trace_actions,
    trace_duplicate_stats,
)


@pytest.fixture(scope="module")
def policy_results():
    return {item["name"]: item for item in c6_stage.run_policy_tests(verbose=False)}


def test_all_policy_conditions_pass(policy_results):
    failed = [name for name, item in policy_results.items() if not item["passed"]]
    assert not failed, f"failed policy tests: {failed}"


def test_reproducibility(policy_results):
    assert policy_results["Reproducibility"]["passed"]


def test_policy_seed_sensitivity(policy_results):
    assert policy_results["Policy-seed sensitivity"]["passed"]


def test_n8_validity(policy_results):
    assert policy_results["N8 validity"]["passed"]


def test_uniform_support(policy_results):
    assert policy_results["Uniform support"]["passed"]


def test_prior_independence(policy_results):
    assert policy_results["Prior independence"]["passed"]


def test_belief_independence(policy_results):
    assert policy_results["Belief independence"]["passed"]


def test_observation_independence(policy_results):
    assert policy_results["Observation independence"]["passed"]


def test_no_deconfliction(policy_results):
    assert policy_results["No deconfliction"]["passed"]


def test_no_cross_uav_information(policy_results):
    assert policy_results["No cross-UAV information"]["passed"]


def test_rng_survivor_invariance(policy_results):
    assert policy_results["RNG survivor invariance"]["passed"]


def test_exposure_multiplicity(policy_results):
    assert policy_results["Exposure multiplicity"]["passed"]


def test_simultaneous_sensing_multiplicity(policy_results):
    assert policy_results["Simultaneous sensing"]["passed"]


def test_unique_L(policy_results):
    assert policy_results["Unique L"]["passed"]


def test_budget_identity(policy_results):
    assert policy_results["Budget identity"]["passed"]


def test_termination_identity(policy_results):
    assert policy_results["Termination identity"]["passed"]


def test_trace_reconstruction(policy_results):
    assert policy_results["Trace reconstruction"]["passed"]


def test_policy_never_consulted_planner_map_entries():
    """Structural no-map-access proof: every map-consumption entry point raises."""
    ds = tiny_dataset()
    sim = random_sim(ds, num_drones=3, budget=2_000.0, policy_seed=7)

    def boom(*args, **kwargs):
        raise AssertionError("random policy accessed planner map internals")

    for name in (
        "_get_planning_map", "_compute_score_map", "_score_position",
        "_bellman_implied_path", "_maximum_neighbor_filter", "_exact_path_reward",
    ):
        setattr(sim, name, boom)
    sim.setup(
        initial_positions=[(60.0, 50.0), (90.0, 80.0), (120.0, 40.0)],
        victim_positions=[],
    )
    sim.run_from_state(dt=1.0, snapshot_interval=500, heatmap_interval=500)
    assert len(sim.drone_paths[0]) > 1


def test_policy_mode_requires_policy_seed():
    ds = tiny_dataset()
    with pytest.raises(ValueError, match="policy_seed"):
        _c_build(ds)


def _c_build(ds):
    from audit_block_c import _build_sim

    return _build_sim(
        ds,
        planning_mode="random_uniform_n8_uncoordinated",
        w=0.5, p_d=0.8, tau=10_000.0, budget=6_000.0, num_drones=2,
        planning_seed=42, implementation_version="c6-test",
    )


def test_duplicate_metric_counts_cycle_multiplicity():
    ds = tiny_dataset()
    sim = random_sim(ds, num_drones=2, policy_seed=13)
    sim.setup(initial_positions=[(600.0, 500.0), (600.0, 500.0)], victim_positions=[])
    for _ in range(50):
        sim._move_drones_random(0.0)
    duplicates, fraction = trace_duplicate_stats(sim._planning_trace)
    # With two UAVs on the same cell the fraction must be non-negative and the
    # cycle-level bookkeeping complete (one row per UAV per cycle).
    cycles = {int(row["replan_id"]) for row in sim._planning_trace}
    assert len(cycles) == 50
    assert 0.0 <= fraction <= 1.0
    assert duplicates >= 0


def test_no_invalid_actions_in_realistic_window():
    ds = tiny_dataset(shape=(60, 70), radius_km=1.0)
    sim = _run_random(random_sim(ds, num_drones=4, budget=1_000.0), [
        (300.0, 250.0), (400.0, 350.0), (350.0, 320.0), (280.0, 300.0),
    ])
    invalid, holds = count_invalid_trace_actions(sim)
    assert invalid == 0
    assert holds == 0


def test_policy_seed_derivation_is_deterministic_and_separate():
    a = c6_stage._policy_seed(42, 1)
    b = c6_stage._policy_seed(42, 1)
    c = c6_stage._policy_seed(43, 1)
    d = c6_stage._policy_seed(42, 2)
    assert a == b
    assert a != c
    assert a != d
    assert isinstance(a, int) and 0 <= a < 2**32
