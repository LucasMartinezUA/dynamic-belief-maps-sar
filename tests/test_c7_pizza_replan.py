"""C7 pizza replan: unit-level geometry tests plus integration smoke.

Mirrors tests/test_c6_random_policy.py: the battery lives in
scripts/c7_stage.py (``run_pizza_tests``); the shared fixture is the same
``pizza_results`` battery, and the extra tests below exercise the pure-geometry
module (sarenv.analytics.midflight) directly at the interface level.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import c7_stage  # noqa: E402
from sarenv.analytics import midflight  # noqa: E402


@pytest.fixture(scope="module")
def pizza_results():
    return {item["name"]: item for item in c7_stage.run_pizza_tests(verbose=False)}


def test_battery_all_pass(pizza_results):
    failed = [name for name, item in pizza_results.items() if not item["passed"]]
    assert not failed, f"failed tests: {failed}"


def test_waypoint_has_unseen_cell_requires_valid_and_unobserved():
    # A cell that is invalid or observed must not make the waypoint usable.
    visible = {(0, 0), (1, 1), (2, 2)}
    assert midflight.waypoint_has_unseen_cell(
        0.0, 0.0, visible_fn=lambda wx, wy: set(visible),
        valid_mask=np.ones((3, 3), dtype=bool), observed=set(), shape=(3, 3),
    )
    assert not midflight.waypoint_has_unseen_cell(
        0.0, 0.0, visible_fn=lambda wx, wy: set(visible),
        valid_mask=np.ones((3, 3), dtype=bool), observed=visible, shape=(3, 3),
    )
    invalid = np.zeros((3, 3), dtype=bool)
    assert not midflight.waypoint_has_unseen_cell(
        0.0, 0.0, visible_fn=lambda wx, wy: set(visible),
        valid_mask=invalid, observed=set(), shape=(3, 3),
    )


def test_trim_covered_tails_contract():
    coords = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    remaining, prefix, suffix = midflight.trim_covered_tails(
        coords, [False, True, False, True, False]
    )
    assert remaining == [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    assert (prefix, suffix) == (1, 1)
    remaining, prefix, suffix = midflight.trim_covered_tails(
        coords, [False, False, False, False, False]
    )
    assert remaining == [] and (prefix, suffix) == (5, 0)
    remaining, prefix, suffix = midflight.trim_covered_tails(
        coords, [True, True, True, True, True]
    )
    assert remaining == coords and (prefix, suffix) == (0, 0)


def test_split_route_contiguity_and_fractions():
    coords = [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)]
    segments = midflight.split_route(coords, 2)
    assert len(segments) == 2
    assert segments[0][0] == coords[0] and segments[1][-1] == coords[-1]
    # contiguous: the end of one segment is the start of the next
    assert segments[0][-1] == segments[1][0]
    # boundary by rounded cumulative-length fractions (~middle)
    assert 1 <= len(segments[0]) <= 3
    with pytest.raises(ValueError):
        midflight.split_route(coords, 1)
    assert midflight.split_route([], 2) == []


def test_route_length_and_hold_line():
    assert midflight.route_length([(0, 0), (3, 4)]) == pytest.approx(5.0)
    assert midflight.route_length([(0, 0)]) == 0.0
    line = midflight.hold_line()
    assert line.is_empty


def test_build_transit_path_orientation_and_first_point():
    task = [(50.0, 0.0), (60.0, 0.0)]
    fwd = midflight.build_transit_path((10.0, 0.0), task, "forward")
    rev = midflight.build_transit_path((10.0, 0.0), task, "reverse")
    assert list(fwd.coords[0]) == [10.0, 0.0]
    assert list(fwd.coords[1]) == [50.0, 0.0]  # entry first in forward
    assert list(rev.coords[1]) == [60.0, 0.0]  # entry first in reverse


def test_pizza_replan_end_to_end_on_tiny_domain():
    """Two-drone pizza_replan with mid-mission failure runs and stays bounded."""
    ds = c7_stage._tiny_item()
    pos = [(60.0, 50.0), (90.0, 80.0)]
    paths = c7_stage._test_pizza_paths(ds, pos, 2)
    sim = c7_stage._run_test_sim(
        c7_stage._c7_test_sim(ds, mode="pizza_replan", static_paths=paths), pos
    )
    assert sim._midflight_triggered
    assert sim._midflight_info["failed_uav_ids"] == [1]
    max_step = max(
        np.hypot(a[0] - b[0], a[1] - b[1])
        for path in sim.drone_paths
        for a, b in zip(path, path[1:])
    )
    assert max_step <= sim.drone_speed + 1e-6
    # failed UAV path frozen at the trigger step
    trigger = sim._midflight_info["trigger_step"]
    assert len(sim.drone_paths[1]) == trigger + 1
