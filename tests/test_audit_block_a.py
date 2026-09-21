"""Regression tests for the Block A audit affordances."""
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sarenv.analytics.simulation import SARSimulation


class _Dataset:
    def __init__(self, size=6):
        self.heatmap = np.full((size, size), 1.0 / (size * size))
        self.bounds = (0.0, 0.0, float(size * 10), float(size * 10))
        self.radius_km = size * 0.01
        self.size = "audit"
        self.center_point = (0.0, 0.0)
        self.environment_type = "flat"
        self.environment_climate = "temperate"
        self.features = gpd.GeoDataFrame(
            {"feature_type": [], "area_probability": []},
            geometry=[],
            crs="EPSG:3857",
        )


def _victims(self):
    return [Point(self.center_x, self.center_y) for _ in range(self.num_victims)]


def _make_sim(dataset, mode, **kwargs):
    return SARSimulation(
        dataset_item=dataset,
        num_drones=1,
        num_victims=1,
        fov_deg=10.0,
        altitude=5.0,
        budget=80.0,
        drone_speed=10.0,
        init_strategy="center",
        planning_mode=mode,
        seed=7,
        **kwargs,
    )


@patch.object(SARSimulation, "_generate_victim_positions", _victims)
def test_online_static_and_dynamic_are_identical_without_belief_updates():
    dataset = _Dataset()
    dynamic = _make_sim(dataset, "dynamic", detection_probability=0.0)
    online_static = _make_sim(dataset, "online_static", detection_probability=0.0)

    dynamic_result = dynamic.run(snapshot_interval=1)
    static_result = online_static.run(snapshot_interval=1)

    assert dynamic_result.first_action_sequence_hash == static_result.first_action_sequence_hash
    assert dynamic_result.target_trace == static_result.target_trace
    assert dynamic_result.number_of_replans == static_result.number_of_replans
    assert dynamic_result.final_drone_paths == static_result.final_drone_paths
    assert dynamic_result.final_likelihood == static_result.final_likelihood


@patch.object(SARSimulation, "_generate_victim_positions", _victims)
def test_forced_posterior_reset_makes_online_controls_identical():
    dataset = _Dataset()
    dynamic = _make_sim(dataset, "dynamic", detection_probability=0.8)
    online_static = _make_sim(dataset, "online_static", detection_probability=0.8)
    original_get_map = dynamic._get_planning_map
    dynamic._get_planning_map = lambda: dynamic.dynamic_heatmap.get_prior()

    dynamic_result = dynamic.run(snapshot_interval=1)
    static_result = online_static.run(snapshot_interval=1)

    dynamic._get_planning_map = original_get_map
    assert dynamic_result.first_action_sequence_hash == static_result.first_action_sequence_hash
    assert dynamic_result.final_drone_paths == static_result.final_drone_paths
    assert dynamic_result.final_likelihood == static_result.final_likelihood


@patch.object(SARSimulation, "_generate_victim_positions", _victims)
def test_online_static_3step_is_identical_to_dynamic_3step_without_updates():
    dataset = _Dataset()
    online_static = _make_sim(
        dataset,
        "online_static_3step",
        detection_probability=0.0,
        decay_tau=float("inf"),
    )
    dynamic = _make_sim(
        dataset,
        "dynamic_3step",
        detection_probability=0.0,
        decay_tau=float("inf"),
    )
    static_result = online_static.run(snapshot_interval=1)
    dynamic_result = dynamic.run(snapshot_interval=1)
    assert static_result.params["planning_mode"] == "online_static_3step"
    assert static_result.target_trace == dynamic_result.target_trace
    assert static_result.number_of_replans == dynamic_result.number_of_replans
    assert static_result.final_drone_paths == dynamic_result.final_drone_paths
    assert static_result.final_likelihood == dynamic_result.final_likelihood


def test_exact_reward_counts_each_fov_cell_once():
    sim = _make_sim(_Dataset(), "tree_3step")
    sim.setup()
    sim.globally_observed_cells = set()
    current_map = np.zeros_like(sim.dataset.heatmap)
    current_map[3, 3] = 1.0

    repeated_fov_path = [(3, 3), (3, 4), (3, 3)]
    assert sim._exact_path_reward(repeated_fov_path, current_map) == 1.0


def test_exact_tree_never_uses_clipped_or_implicit_wait_actions():
    sim = _make_sim(_Dataset(), "tree_3step")
    sim.setup()
    sim.globally_observed_cells = set()
    current_map = np.ones_like(sim.dataset.heatmap)
    start = (0, 0)

    candidates = sim._enumerate_exact_tree_paths(*start, current_map)
    assert candidates
    for path, _ in candidates:
        assert len(path) == 3
        previous = start
        for row, col in path:
            assert (row, col) != previous
            assert 0 <= row < 6 and 0 <= col < 6
            previous = (row, col)


def test_exact_tree_uses_unique_fov_union_for_hotspots():
    sim = _make_sim(_Dataset(), "tree_2step")
    sim.setup()
    sim.globally_observed_cells = set()
    current_map = np.zeros_like(sim.dataset.heatmap)
    current_map[2, 2] = 0.5
    current_map[4, 4] = 0.5

    candidates = sim._enumerate_exact_tree_paths(2, 2, current_map)
    assert candidates
    assert max(score for _, score in candidates) <= 1.0
