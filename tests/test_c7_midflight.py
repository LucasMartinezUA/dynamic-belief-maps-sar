"""C7 mid-mission failure: the 17 mandatory tests (T1-T17).

The tests are implemented in scripts/c7_stage.py (``run_pizza_tests``) so the
campaign preflight can run the same battery in-process before any scientific
run (fail-closed, like C6). This module wraps the battery in pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import c7_stage  # noqa: E402


@pytest.fixture(scope="module")
def pizza_results():
    return {item["name"]: item for item in c7_stage.run_pizza_tests(verbose=False)}


def test_all_tests_pass(pizza_results):
    failed = [name for name, item in pizza_results.items() if not item["passed"]]
    assert not failed, f"failed tests: {failed}"


def test_t1_determinism(pizza_results):
    assert pizza_results["T1 Determinism"]["passed"]


def test_t2_no_teleport(pizza_results):
    assert pizza_results["T2 No teleport"]["passed"]


def test_t3_budget_conservation(pizza_results):
    assert pizza_results["T3 Budget conservation"]["passed"]


def test_t4_dead_uav_silence(pizza_results):
    assert pizza_results["T4 Dead UAV silence"]["passed"]


def test_t5_presstep_activation_matches_c4_pipeline(pizza_results):
    assert pizza_results["T5 Pre-step activation matches C4 pipeline"]["passed"]


def test_t6_completed_tails_removed(pizza_results):
    assert pizza_results["T6 Completed tails removed"]["passed"]


def test_t7_interior_covered_kept(pizza_results):
    assert pizza_results["T7 Interior covered kept"]["passed"]


def test_t8_hold_policy_splits_and_search_inactive(pizza_results):
    assert pizza_results["T8 Hold policy splits and search-inactive"]["passed"]


def test_t9_assignment_known_case_and_tiebreak(pizza_results):
    assert pizza_results["T9 Assignment known case and tiebreak"]["passed"]


def test_t10_orientation_cheaper_end(pizza_results):
    assert pizza_results["T10 Orientation cheaper end"]["passed"]


def test_t11_fov_unseen_cell_beats_center(pizza_results):
    assert pizza_results["T11 FOV unseen cell beats center"]["passed"]


def test_t12_replanned_waypoints_valid(pizza_results):
    assert pizza_results["T12 Replanned waypoints valid"]["passed"]


def test_t13_simultaneous_k2_failure(pizza_results):
    assert pizza_results["T13 Simultaneous k=2 failure"]["passed"]


def test_t14_no_probability_leakage(pizza_results):
    assert pizza_results["T14 No probability leakage"]["passed"]


def test_t15_reproducibility(pizza_results):
    assert pizza_results["T15 Reproducibility"]["passed"]


def test_t16_checkpoint_cloning_and_shadow_belief(pizza_results):
    assert pizza_results["T16 Checkpoint cloning and shadow belief"]["passed"]


def test_t17_integration_smoke(pizza_results):
    assert pizza_results["T17 Integration smoke"]["passed"]


def test_t18_postfault_replan_cadence(pizza_results):
    assert pizza_results["T18 Post-fault replan cadence"]["passed"]


def test_t19_hold_on_task_finish(pizza_results):
    assert pizza_results["T19 Hold on task finish"]["passed"]
