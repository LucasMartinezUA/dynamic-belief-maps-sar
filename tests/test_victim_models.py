# tests/test_victim_models.py
"""Tests for victim pedestrian movement models."""
import numpy as np
import pytest
from shapely.geometry import Point
import geopandas as gpd
from shapely.geometry import LineString, Polygon

from sarenv.core.victim_models import (
    RandomWalkModel,
    RouteFollowingModel,
    LostPersonBehaviorModel,
    VICTIM_MODELS,
)


@pytest.fixture
def bounds():
    """Rectangular bounds (minx, miny, maxx, maxy)."""
    return (0.0, 0.0, 1000.0, 1000.0)


@pytest.fixture
def empty_features():
    """Empty GeoDataFrame with required columns."""
    return gpd.GeoDataFrame(
        {"feature_type": [], "geometry": [], "area_probability": []},
        geometry="geometry",
    )


@pytest.fixture
def features_with_roads():
    """GeoDataFrame with some road features."""
    roads = [
        LineString([(100, 500), (900, 500)]),
        LineString([(500, 100), (500, 900)]),
    ]
    return gpd.GeoDataFrame(
        {
            "feature_type": ["road", "road"],
            "geometry": roads,
            "area_probability": [0.15, 0.15],
        },
        geometry="geometry",
    )


class TestRandomWalkModel:
    def test_stays_in_bounds(self, bounds, empty_features):
        model = RandomWalkModel(speed=5.0, seed=42)
        pos = Point(500, 500)
        for _ in range(100):
            pos = model.move(pos, dt=1.0, features_gdf=empty_features, bounds=bounds)
        assert bounds[0] <= pos.x <= bounds[2]
        assert bounds[1] <= pos.y <= bounds[3]

    def test_moves_from_origin(self, bounds, empty_features):
        model = RandomWalkModel(speed=5.0, seed=42)
        start = Point(500, 500)
        pos = start
        for _ in range(10):
            pos = model.move(pos, dt=1.0, features_gdf=empty_features, bounds=bounds)
        # Should have moved at least somewhat
        dist = start.distance(pos)
        assert dist > 0

    def test_respects_speed(self, bounds, empty_features):
        speed = 2.0
        model = RandomWalkModel(speed=speed, seed=42)
        pos = Point(500, 500)
        new_pos = model.move(pos, dt=1.0, features_gdf=empty_features, bounds=bounds)
        dist = pos.distance(new_pos)
        assert dist <= speed * 1.0 + 1e-6  # Allow small floating-point tolerance

    def test_clips_to_bounds(self, bounds, empty_features):
        """Starting at edge should clip to bounds."""
        model = RandomWalkModel(speed=100.0, seed=42)
        pos = Point(0, 0)
        for _ in range(50):
            pos = model.move(pos, dt=1.0, features_gdf=empty_features, bounds=bounds)
            assert bounds[0] <= pos.x <= bounds[2]
            assert bounds[1] <= pos.y <= bounds[3]


class TestRouteFollowingModel:
    def test_stays_in_bounds(self, bounds, features_with_roads):
        model = RouteFollowingModel(speed=3.0, seed=42)
        pos = Point(500, 500)
        for _ in range(100):
            pos = model.move(pos, dt=1.0, features_gdf=features_with_roads, bounds=bounds)
        assert bounds[0] <= pos.x <= bounds[2]
        assert bounds[1] <= pos.y <= bounds[3]

    def test_falls_back_without_routes(self, bounds, empty_features):
        """Should still work (random walk fallback) when no linear features exist."""
        model = RouteFollowingModel(speed=3.0, seed=42)
        pos = Point(500, 500)
        new_pos = model.move(pos, dt=1.0, features_gdf=empty_features, bounds=bounds)
        assert isinstance(new_pos, Point)


class TestLostPersonBehaviorModel:
    def test_stays_in_bounds(self, bounds, features_with_roads):
        model = LostPersonBehaviorModel(speed=1.0, seed=42)
        pos = Point(500, 500)
        for _ in range(50):
            pos = model.move(pos, dt=1.0, features_gdf=features_with_roads, bounds=bounds)
        assert bounds[0] <= pos.x <= bounds[2]
        assert bounds[1] <= pos.y <= bounds[3]

    def test_falls_back_without_features(self, bounds, empty_features):
        model = LostPersonBehaviorModel(speed=1.0, seed=42)
        pos = Point(500, 500)
        new_pos = model.move(pos, dt=1.0, features_gdf=empty_features, bounds=bounds)
        assert isinstance(new_pos, Point)


class TestModelRegistry:
    def test_all_models_registered(self):
        assert "random_walk" in VICTIM_MODELS
        assert "route_following" in VICTIM_MODELS
        assert "lost_person" in VICTIM_MODELS

    def test_instantiation_from_registry(self):
        for name, cls in VICTIM_MODELS.items():
            instance = cls(speed=1.0, seed=0)
            assert hasattr(instance, "move")
