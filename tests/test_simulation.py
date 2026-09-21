# tests/test_simulation.py
"""Tests for the SARSimulation engine."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from sarenv.analytics.simulation import SARSimulation, SimulationResult, SimulationSnapshot
from sarenv.analytics.dynamic_heatmap import DynamicHeatmap
from sarenv.core.victim_models import RandomWalkModel
from shapely.geometry import Point, LineString


@pytest.fixture
def mock_dataset():
    """Create a lightweight mock dataset item for testing."""
    ds = MagicMock()
    ds.heatmap = np.random.default_rng(0).random((20, 20))
    ds.heatmap /= ds.heatmap.sum()
    ds.bounds = (0.0, 0.0, 600.0, 600.0)
    ds.radius_km = 0.3
    ds.size = "test"
    ds.center_point = (0.1, 51.5)
    ds.environment_type = "flat"
    ds.environment_climate = "temperate"

    import geopandas as gpd
    from shapely.geometry import LineString
    ds.features = gpd.GeoDataFrame(
        {
            "feature_type": ["road", "woodland"],
            "geometry": [
                LineString([(100, 300), (500, 300)]),
                LineString([(300, 100), (300, 500)]),
            ],
            "area_probability": [0.15, 0.1],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    return ds


def _fake_victim_positions(self):
    """Generate simple victim positions within bounds for testing."""
    rng = np.random.default_rng(self.seed)
    minx, miny, maxx, maxy = self.bounds
    positions = []
    for _ in range(self.num_victims):
        x = rng.uniform(minx + 50, maxx - 50)
        y = rng.uniform(miny + 50, maxy - 50)
        positions.append(Point(x, y))
    return positions


class TestSimulationSetup:
    def test_init_creates_simulation(self, mock_dataset):
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=2,
            num_victims=2,
            budget=500.0,
            seed=42,
        )
        assert sim.num_drones == 2
        assert sim.num_victims == 2
        assert sim.detection_radius > 0

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_setup_initializes_state(self, mock_dataset):
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=2,
            num_victims=2,
            budget=500.0,
            seed=42,
        )
        sim.setup()
        assert len(sim.drone_positions) == 2
        assert len(sim.victim_positions) == 2
        assert len(sim.victims_found) == 2
        assert all(v is False for v in sim.victims_found)
        assert sim.dynamic_heatmap is not None

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_init_strategies(self, mock_dataset):
        for strategy in ["circle", "center", "grid", "random"]:
            sim = SARSimulation(
                dataset_item=mock_dataset,
                num_drones=3,
                num_victims=1,
                budget=500.0,
                init_strategy=strategy,
                seed=42,
            )
            sim.setup()
            assert len(sim.drone_positions) == 3

    def test_unknown_strategy_raises(self, mock_dataset):
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=1,
            num_victims=1,
            budget=500.0,
            init_strategy="invalid",
        )
        with pytest.raises(ValueError, match="Unknown init_strategy"):
            sim.setup()

    def test_unknown_victim_model_raises(self, mock_dataset):
        with pytest.raises(ValueError, match="Unknown victim model"):
            SARSimulation(
                dataset_item=mock_dataset,
                num_drones=1,
                num_victims=1,
                victim_model="nonexistent",
            )


class TestSimulationRun:
    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_budget_termination(self, mock_dataset):
        """Simulation should stop when budget is exhausted."""
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=2,
            num_victims=1,
            budget=500.0,
            seed=42,
        )
        result = sim.run(dt=1.0, snapshot_interval=5, heatmap_interval=10)
        assert isinstance(result, SimulationResult)
        assert result.total_distance >= sim.budget or sim._done

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_result_has_snapshots(self, mock_dataset):
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=1,
            num_victims=1,
            budget=200.0,
            seed=42,
        )
        result = sim.run(dt=1.0, snapshot_interval=5)
        assert len(result.snapshots) > 0
        # Last snapshot should have heatmap
        assert result.snapshots[-1].heatmap is not None

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_result_params(self, mock_dataset):
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=2,
            num_victims=3,
            budget=300.0,
            seed=7,
        )
        result = sim.run(dt=1.0)
        assert result.params["num_drones"] == 2
        assert result.params["num_victims"] == 3
        assert result.params["seed"] == 7

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_final_heatmap_shape(self, mock_dataset):
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=1,
            num_victims=1,
            budget=200.0,
        )
        result = sim.run()
        assert result.final_heatmap.shape == mock_dataset.heatmap.shape

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_drone_paths_recorded(self, mock_dataset):
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=2,
            num_victims=1,
            budget=300.0,
        )
        result = sim.run(dt=1.0)
        assert len(result.final_drone_paths) == 2
        # Each path should have at least the initial position
        for path in result.final_drone_paths:
            assert len(path) >= 1


class TestSimulationSnapshot:
    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_snapshot_structure(self, mock_dataset):
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=2,
            num_victims=3,
            budget=100.0,
        )
        result = sim.run(dt=1.0, snapshot_interval=5)
        snap = result.snapshots[0]
        assert isinstance(snap, SimulationSnapshot)
        assert len(snap.drone_positions) == 2
        assert len(snap.victim_positions) == 3
        assert len(snap.victims_found) == 3
        assert isinstance(snap.likelihood_score, float)
        assert isinstance(snap.cells_observed, int)


class TestPlanningModes:
    def test_invalid_planning_mode_raises(self, mock_dataset):
        with pytest.raises(ValueError, match="Unknown planning_mode"):
            SARSimulation(
                dataset_item=mock_dataset,
                num_drones=1,
                num_victims=1,
                budget=500.0,
                planning_mode="invalid",
            )

    def test_static_without_paths_raises(self, mock_dataset):
        with pytest.raises(ValueError, match="static_paths"):
            SARSimulation(
                dataset_item=mock_dataset,
                num_drones=1,
                num_victims=1,
                budget=500.0,
                planning_mode="static",
            )

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_static_mode_runs(self, mock_dataset):
        """Static mode should follow pre-computed paths."""
        minx, miny, maxx, maxy = mock_dataset.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        paths = [
            LineString([(cx, cy), (cx + 100, cy), (cx + 200, cy), (cx + 200, cy + 100)]),
        ]
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=1,
            num_victims=1,
            budget=500.0,
            planning_mode="static",
            static_paths=paths,
            seed=42,
        )
        result = sim.run(dt=1.0)
        assert isinstance(result, SimulationResult)
        assert result.total_distance > 0
        assert result.params["planning_mode"] == "static"

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_static_2step_runs(self, mock_dataset):
        """Static 2-step lookahead should generate paths with N-step planning."""
        from sarenv.analytics.paths import generate_greedy_path
        minx, miny, maxx, maxy = mock_dataset.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        max_radius = mock_dataset.radius_km * 1000
        
        # Generate paths with 2-step lookahead
        paths_2step = generate_greedy_path(
            center_x=cx,
            center_y=cy,
            num_drones=1,
            probability_map=mock_dataset.heatmap,
            bounds=mock_dataset.bounds,
            max_radius=max_radius,
            fov_deg=45.0,
            altitude=80.0,
            budget=500.0,
            lookahead_steps=2,
        )
        
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=1,
            num_victims=1,
            budget=500.0,
            planning_mode="static_2step",
            static_paths=paths_2step,
            seed=42,
        )
        result = sim.run(dt=1.0)
        assert isinstance(result, SimulationResult)
        assert result.total_distance > 0
        assert result.params["planning_mode"] == "static_2step"

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_dynamic_2step_runs(self, mock_dataset):
        """2-step lookahead should complete and produce valid results."""
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=1,
            num_victims=1,
            budget=300.0,
            planning_mode="dynamic_2step",
            seed=42,
        )
        result = sim.run(dt=1.0)
        assert isinstance(result, SimulationResult)
        assert result.total_distance > 0
        assert result.params["planning_mode"] == "dynamic_2step"

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_dynamic_3step_runs(self, mock_dataset):
        """3-step lookahead via convolution should complete."""
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=1,
            num_victims=1,
            budget=300.0,
            planning_mode="dynamic_3step",
            seed=42,
        )
        result = sim.run(dt=1.0)
        assert result.params["planning_mode"] == "dynamic_3step"
        assert result.total_distance > 0

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_dynamic_4step_runs(self, mock_dataset):
        """4-step lookahead via convolution should complete."""
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=1,
            num_victims=1,
            budget=300.0,
            planning_mode="dynamic_4step",
            seed=42,
        )
        result = sim.run(dt=1.0)
        assert result.params["planning_mode"] == "dynamic_4step"
        assert result.total_distance > 0

    def test_1step_nstep_rejected(self, mock_dataset):
        """dynamic_1step should be rejected (use 'dynamic' instead)."""
        with pytest.raises(ValueError, match="N >= 2"):
            SARSimulation(
                dataset_item=mock_dataset,
                num_drones=1,
                num_victims=1,
                planning_mode="dynamic_1step",
            )

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_default_mode_is_dynamic(self, mock_dataset):
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=1,
            num_victims=1,
            budget=200.0,
            seed=42,
        )
        assert sim.planning_mode == "dynamic"
        result = sim.run(dt=1.0)
        assert result.params["planning_mode"] == "dynamic"

    @patch.object(SARSimulation, '_generate_victim_positions', _fake_victim_positions)
    def test_static_victims_move(self, mock_dataset):
        """In static mode, victims should still move (fair comparison)."""
        minx, miny, maxx, maxy = mock_dataset.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        paths = [
            LineString([(cx, cy), (cx + 200, cy), (cx + 200, cy + 200)]),
        ]
        sim = SARSimulation(
            dataset_item=mock_dataset,
            num_drones=1,
            num_victims=2,
            budget=500.0,
            planning_mode="static",
            static_paths=paths,
            seed=42,
        )
        sim.setup()
        initial_positions = [Point(v.x, v.y) for v in sim.victim_positions]

        # Run several steps
        for _ in range(50):
            if sim._done:
                break
            sim.step(dt=1.0)

        # At least one victim should have moved
        moved = any(
            abs(sim.victim_positions[i].x - initial_positions[i].x) > 0.1
            or abs(sim.victim_positions[i].y - initial_positions[i].y) > 0.1
            for i in range(len(initial_positions))
        )
        assert moved, "Victims should move in static mode"
