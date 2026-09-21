# sarenv/core/victim_models.py
"""
Pedestrian movement models for simulating lost-person motion during SAR operations.

Three models with a common interface:
  - RandomWalkModel: Brownian motion (disoriented person)
  - RouteFollowingModel: Follows roads / linear features
  - LostPersonBehaviorModel: Koester-inspired, feature-weighted drift
"""
from abc import ABC, abstractmethod

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from sarenv.utils.lost_person_behavior import FEATURE_PROBABILITIES


# Speed multipliers by feature type (relative to base speed)
_TERRAIN_SPEED_FACTORS = {
    "road": 1.0,
    "linear": 0.9,
    "field": 0.8,
    "structure": 0.7,
    "drainage": 0.4,
    "brush": 0.5,
    "scrub": 0.4,
    "woodland": 0.35,
    "rock": 0.3,
    "water": 0.05,  # nearly impassable
}


class VictimModel(ABC):
    """Abstract base class for victim movement models."""

    def __init__(self, speed: float = 0.5, seed: int | None = None):
        """
        Args:
            speed: Base movement speed in m/s.
            seed: Random seed for reproducibility.
        """
        self.speed = speed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def move(
        self,
        position: Point,
        dt: float,
        features_gdf: gpd.GeoDataFrame,
        bounds: tuple,
    ) -> Point:
        """
        Compute the next position of a victim.

        Args:
            position: Current victim position (projected coordinates).
            dt: Timestep in seconds.
            features_gdf: Geographic features with geometry and feature_type columns.
            bounds: (minx, miny, maxx, maxy) of the search area.

        Returns:
            New Point position clipped to bounds.
        """

    def _clip_to_bounds(self, x: float, y: float, bounds: tuple) -> Point:
        minx, miny, maxx, maxy = bounds
        x = np.clip(x, minx, maxx)
        y = np.clip(y, miny, maxy)
        return Point(x, y)


class RandomWalkModel(VictimModel):
    """
    Brownian motion — models a disoriented person wandering without clear direction.

    At each step the victim moves ``speed × dt`` in a random direction with
    Gaussian angular noise.
    """

    def move(self, position, dt, features_gdf, bounds):
        step_size = self.speed * dt
        angle = self.rng.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        return self._clip_to_bounds(position.x + dx, position.y + dy, bounds)


class RouteFollowingModel(VictimModel):
    """
    Follows the nearest road or linear feature, with occasional deviations.

    The victim moves toward the closest point on road/linear features with some
    noise.  If no features are nearby it falls back to a random walk.
    """

    def __init__(self, speed: float = 0.5, deviation_prob: float = 0.15, seed: int | None = None):
        """
        Args:
            deviation_prob: Probability of deviating from the route at each step.
        """
        super().__init__(speed, seed)
        self.deviation_prob = deviation_prob
        self._route_features = None

    def _get_route_features(self, features_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Cache road + linear features for faster lookup."""
        if self._route_features is None:
            from shapely.geometry import LineString, MultiLineString
            mask = features_gdf["feature_type"].isin(["road", "linear"])
            candidates = features_gdf[mask]
            # Only keep actual linear geometries (not polygons)
            if not candidates.empty:
                linear_mask = candidates.geometry.apply(lambda g: isinstance(g, (LineString, MultiLineString)))
                self._route_features = candidates[linear_mask]
            else:
                self._route_features = gpd.GeoDataFrame()
        return self._route_features

    def move(self, position, dt, features_gdf, bounds):
        step_size = self.speed * dt

        # Occasional random deviation
        if self.rng.random() < self.deviation_prob:
            angle = self.rng.uniform(0, 2 * np.pi)
            dx = step_size * np.cos(angle)
            dy = step_size * np.sin(angle)
            return self._clip_to_bounds(position.x + dx, position.y + dy, bounds)

        routes = self._get_route_features(features_gdf)
        if routes.empty:
            # Fallback to random walk
            angle = self.rng.uniform(0, 2 * np.pi)
            dx = step_size * np.cos(angle)
            dy = step_size * np.sin(angle)
            return self._clip_to_bounds(position.x + dx, position.y + dy, bounds)

        # Find nearest route geometry
        distances = routes.geometry.distance(position)
        nearest_idx = distances.idxmin()
        nearest_geom = routes.loc[nearest_idx, "geometry"]

        # Project position onto the nearest route and walk along it
        from shapely.geometry import LineString, MultiLineString
        if isinstance(nearest_geom, (LineString, MultiLineString)):
            nearest_point = nearest_geom.interpolate(nearest_geom.project(position))
        else:
            # Fallback for non-linear geometries (shouldn't happen after filtering, but be safe)
            nearest_point = nearest_geom.centroid
            
        dx_to_route = nearest_point.x - position.x
        dy_to_route = nearest_point.y - position.y
        dist_to_route = np.sqrt(dx_to_route**2 + dy_to_route**2)

        if dist_to_route < step_size:
            # Already on the route — walk along it
            proj_dist = nearest_geom.project(position)
            direction = 1.0 if self.rng.random() > 0.5 else -1.0
            new_proj = proj_dist + direction * step_size
            new_proj = np.clip(new_proj, 0, nearest_geom.length)
            target = nearest_geom.interpolate(new_proj)
            # Add small noise
            noise = self.rng.normal(0, step_size * 0.05, size=2)
            return self._clip_to_bounds(target.x + noise[0], target.y + noise[1], bounds)
        else:
            # Walk toward the route
            ratio = step_size / dist_to_route
            new_x = position.x + dx_to_route * ratio
            new_y = position.y + dy_to_route * ratio
            return self._clip_to_bounds(new_x, new_y, bounds)


class LostPersonBehaviorModel(VictimModel):
    """
    Koester-inspired model: person drifts toward high-probability features with
    speed modulated by terrain type.

    At each step:
    1. Sample a target feature type weighted by FEATURE_PROBABILITIES.
    2. Find the nearest feature of that type.
    3. Move toward it at speed modulated by the underlying terrain.
    4. Add Gaussian noise to simulate disorientation.
    """

    def __init__(self, speed: float = 0.5, seed: int | None = None):
        super().__init__(speed, seed)
        self._feature_types = list(FEATURE_PROBABILITIES.keys())
        self._feature_weights = np.array([FEATURE_PROBABILITIES[t] for t in self._feature_types])
        self._feature_weights = self._feature_weights / self._feature_weights.sum()
        self._type_cache: dict[str, gpd.GeoDataFrame] = {}

    def _features_by_type(self, features_gdf, ftype):
        if ftype not in self._type_cache:
            self._type_cache[ftype] = features_gdf[features_gdf["feature_type"] == ftype]
        return self._type_cache[ftype]

    def move(self, position, dt, features_gdf, bounds):
        # Choose target feature type
        target_type = self.rng.choice(self._feature_types, p=self._feature_weights)

        candidates = self._features_by_type(features_gdf, target_type)
        if candidates.empty:
            # Fallback: random walk
            angle = self.rng.uniform(0, 2 * np.pi)
            step = self.speed * dt
            return self._clip_to_bounds(
                position.x + step * np.cos(angle),
                position.y + step * np.sin(angle),
                bounds,
            )

        # Nearest feature of chosen type
        distances = candidates.geometry.distance(position)
        nearest_idx = distances.idxmin()
        nearest_geom = candidates.loc[nearest_idx, "geometry"]
        
        # Get nearest point on geometry (handle both linear and areal features)
        from shapely.geometry import LineString, MultiLineString
        if isinstance(nearest_geom, (LineString, MultiLineString)):
            # For linear features, project onto the line
            nearest_point = nearest_geom.interpolate(nearest_geom.project(position))
        else:
            # For polygons/other shapes, use the nearest point on boundary or centroid
            try:
                nearest_point = nearest_geom.boundary.interpolate(nearest_geom.boundary.project(position))
            except:
                # Final fallback: use centroid
                nearest_point = nearest_geom.centroid

        # Speed factor based on underlying terrain
        speed_factor = _TERRAIN_SPEED_FACTORS.get(target_type, 0.6)
        step_size = self.speed * speed_factor * dt

        dx = nearest_point.x - position.x
        dy = nearest_point.y - position.y
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1e-6:
            # Already at feature — wander randomly
            angle = self.rng.uniform(0, 2 * np.pi)
            nx = position.x + step_size * np.cos(angle)
            ny = position.y + step_size * np.sin(angle)
        elif dist < step_size:
            nx = nearest_point.x
            ny = nearest_point.y
        else:
            ratio = step_size / dist
            nx = position.x + dx * ratio
            ny = position.y + dy * ratio

        # Gaussian noise for disorientation
        noise = self.rng.normal(0, step_size * 0.15, size=2)
        return self._clip_to_bounds(nx + noise[0], ny + noise[1], bounds)


# Registry for easy CLI lookup
VICTIM_MODELS = {
    "random_walk": RandomWalkModel,
    "route_following": RouteFollowingModel,
    "lost_person": LostPersonBehaviorModel,
}
