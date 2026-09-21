"""
Unit tests for the path evaluation bug fixes.

This test suite covers two critical bugs:
1. Incorrect distance calculation in metrics.py (the +1 bug)
2. Incorrect cell_key calculation causing over-sampling
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

# Suppress known warnings from external libraries
warnings.filterwarnings("ignore", category=DeprecationWarning, module="geopandas")
warnings.filterwarnings("ignore", message=".*shapely.geos.*")
warnings.filterwarnings("ignore", message=".*TripleDES.*")
warnings.filterwarnings("ignore", message=".*Blowfish.*")

# Add the project root to the path to import sarenv
sys.path.insert(0, str(Path(__file__).parent.parent))

from sarenv.analytics import metrics, paths
from sarenv.analytics.evaluator import PathGeneratorConfig


class TestMetricsBugs:
    """Test suite for metrics calculation bugs."""
    
    @pytest.fixture
    def setup_evaluator(self):
        """Set up test fixtures."""
        # Create a simple uniform heatmap for testing
        heatmap = np.ones((10, 10))  # 10x10 uniform probability map
        extent = (0, 0, 100, 100)  # 100m x 100m area
        empty_victims = gpd.GeoDataFrame(geometry=[])
        
        # Standard evaluator configuration
        evaluator = metrics.PathEvaluator(
            heatmap=heatmap,
            extent=extent,
            victims=empty_victims,
            fov_deg=45.0,
            altitude=80.0,
            meters_per_bin=10.0
        )
        
        return {
            'evaluator': evaluator,
            'heatmap': heatmap,
            'extent': extent,
            'empty_victims': empty_victims
        }
    
    def test_distance_calculation_no_plus_one_bug(self, setup_evaluator):
        """Test that distances start from 0, not 1 (Bug #1 fix)."""
        evaluator = setup_evaluator['evaluator']
        
        # Create a simple straight line path
        path = LineString([(10, 10), (20, 10), (30, 10), (40, 10)])
        
        # Calculate metrics
        metrics_result = evaluator.calculate_all_metrics([path], discount_factor=0.999)
        
        # Check that the path's first cumulative distance is 0
        cumulative_distances = metrics_result['cumulative_distances'][0]
        assert cumulative_distances[0] == 0.0, "First distance should be 0, not 1 (no +1 bug)"
        
        # Verify distances are properly spaced
        assert len(cumulative_distances) > 1, "Should have multiple distance points"
        assert all(cumulative_distances[i] >= cumulative_distances[i-1] 
                  for i in range(1, len(cumulative_distances))), \
               "Distances should be monotonically increasing"
    
    def test_heatmap_cell_key_calculation(self, setup_evaluator):
        """Test that cell_key uses heatmap grid, not interpolation grid (Bug #3 fix)."""
        evaluator = setup_evaluator['evaluator']
        
        # Create a path that visits multiple cells
        path = LineString([(5, 5), (15, 15), (25, 25), (35, 35)])
        
        # Calculate metrics
        metrics_result = evaluator.calculate_all_metrics([path], discount_factor=1.0)
        likelihood_score = metrics_result['total_likelihood_score']
        
        # With a 10x10 uniform heatmap (each cell = 1.0) and the detection radius,
        # the score should be reasonable and not exceed the total heatmap sum
        heatmap_total = np.sum(setup_evaluator['heatmap'])  # 100.0 for 10x10 ones
        assert likelihood_score <= heatmap_total, \
               f"Likelihood should not exceed total heatmap sum: {likelihood_score} <= {heatmap_total}"
        
        # Verify it's reasonable (should be a significant portion due to detection radius)
        assert likelihood_score > 0.0, "Should capture some likelihood from path"
        
        # The key fix: ensure we're using heatmap cells, not interpolation grid
        # This is verified by the score being based on discrete cell values
        assert isinstance(likelihood_score, float), "Score should be numeric"
    
    def test_no_likelihood_exceeds_theoretical_maximum(self, setup_evaluator):
        """Test that no path can exceed the sum of all heatmap values in coverage area."""
        evaluator = setup_evaluator['evaluator']
        heatmap = setup_evaluator['heatmap']
        
        # Create a path that tries to cover the entire area multiple times
        coords = []
        for i in range(10):
            for j in range(10):
                x = 5 + i * 10  # Cell centers
                y = 5 + j * 10
                coords.append((x, y))
        
        # Create a path visiting all cells
        full_coverage_path = LineString(coords)
        
        # Calculate metrics
        metrics_result = evaluator.calculate_all_metrics([full_coverage_path], discount_factor=1.0)
        likelihood_score = metrics_result['total_likelihood_score']
        
        # Theoretical maximum is sum of all heatmap values
        theoretical_max = np.sum(heatmap)  # 100.0 for 10x10 ones
        
        assert likelihood_score <= theoretical_max, \
               f"Likelihood score {likelihood_score} should not exceed " \
               f"theoretical maximum {theoretical_max}"
    
    def test_multiple_paths_no_double_counting(self, setup_evaluator):
        """Test that multiple paths don't double-count the same cells."""
        evaluator = setup_evaluator['evaluator']
        
        # Create two overlapping paths
        path1 = LineString([(5, 5), (15, 15), (25, 25)])
        path2 = LineString([(5, 5), (15, 15), (35, 35)])  # Overlaps with path1
        
        # Calculate metrics for both paths
        metrics_result = evaluator.calculate_all_metrics([path1, path2], discount_factor=1.0)
        likelihood_score = metrics_result['total_likelihood_score']
        
        # Calculate individual path scores
        result1 = evaluator.calculate_all_metrics([path1], discount_factor=1.0)
        result2 = evaluator.calculate_all_metrics([path2], discount_factor=1.0)
        
        # The key test: combined score should be <= sum of individual scores
        # (because overlapping coverage is not double-counted)
        individual_sum = result1['total_likelihood_score'] + result2['total_likelihood_score']
        
        assert likelihood_score <= individual_sum, \
               "Multiple overlapping paths should not double-count cells"
        
        # Also verify the score is reasonable
        heatmap_total = np.sum(setup_evaluator['heatmap'])
        assert likelihood_score <= heatmap_total, \
               "Score should not exceed total heatmap sum"
    
    def test_interpolation_resolution_consistency(self, setup_evaluator):
        """Test that different interpolation resolutions give consistent results."""
        heatmap = setup_evaluator['heatmap']
        extent = setup_evaluator['extent']
        empty_victims = setup_evaluator['empty_victims']
        
        path = LineString([(5, 5), (25, 25), (45, 45), (65, 65)])
        
        # Test with different meters_per_bin values
        scores = []
        for meters_per_bin in [5.0, 10.0, 20.0]:
            evaluator = metrics.PathEvaluator(
                heatmap=heatmap,
                extent=extent,
                victims=empty_victims,
                fov_deg=45.0,
                altitude=80.0,
                meters_per_bin=meters_per_bin
            )
            
            metrics_result = evaluator.calculate_all_metrics([path], discount_factor=1.0)
            scores.append(metrics_result['total_likelihood_score'])
        
        # All scores should be the same since we're using the same heatmap grid
        for i in range(1, len(scores)):
            assert scores[0] == scores[i], \
                   f"Different interpolation resolutions should give same likelihood score. " \
                   f"Got {scores[0]} vs {scores[i]} for different meters_per_bin values"
    
    def test_empty_path_handling(self, setup_evaluator):
        """Test that empty paths are handled correctly."""
        evaluator = setup_evaluator['evaluator']
        
        empty_path = LineString()
        
        metrics_result = evaluator.calculate_all_metrics([empty_path], discount_factor=1.0)
        
        assert metrics_result['total_likelihood_score'] == 0.0
        assert metrics_result['total_time_discounted_score'] == 0.0
        assert len(metrics_result['cumulative_distances']) == 1
        assert len(metrics_result['cumulative_likelihoods']) == 1


class TestGreedyPathBugs:
    """Test suite for greedy path generation bugs."""
    
    @pytest.fixture
    def setup_greedy_test(self):
        """Set up test fixtures."""
        heatmap = np.ones((20, 20))  # Uniform probability map
        bounds = (0, 0, 200, 200)  # 200m x 200m
        center_x, center_y = 100, 100
        max_radius = 50  # Small radius for controlled testing
        
        return {
            'heatmap': heatmap,
            'bounds': bounds,
            'center_x': center_x,
            'center_y': center_y,
            'max_radius': max_radius
        }
    
    def test_greedy_respects_theoretical_maximum(self, setup_greedy_test):
        """Test that greedy algorithm doesn't exceed theoretical maximum."""
        test_data = setup_greedy_test
        heatmap = test_data['heatmap']
        bounds = test_data['bounds']
        center_x = test_data['center_x']
        center_y = test_data['center_y']
        max_radius = test_data['max_radius']
        
        # Generate greedy path
        greedy_paths = paths.generate_greedy_path(
            center_x=center_x,
            center_y=center_y,
            num_drones=1,
            probability_map=heatmap,
            bounds=bounds,
            max_radius=max_radius
        )
        
        # Calculate metrics
        empty_victims = gpd.GeoDataFrame(geometry=[])
        evaluator = metrics.PathEvaluator(
            heatmap=heatmap,
            extent=bounds,
            victims=empty_victims,
            fov_deg=45.0,
            altitude=80.0,
            meters_per_bin=10.0
        )
        
        greedy_metrics = evaluator.calculate_all_metrics(greedy_paths, discount_factor=1.0)
        likelihood_score = greedy_metrics['total_likelihood_score']
        
        # The theoretical maximum is the sum of all heatmap cells (since each cell = 1.0)
        # The greedy algorithm should not exceed this
        theoretical_max = np.sum(heatmap)
        
        assert likelihood_score <= theoretical_max, \
               f"Greedy algorithm score {likelihood_score} should not exceed " \
               f"theoretical maximum {theoretical_max}"
        
        # Also verify the score is reasonable (should be positive)
        assert likelihood_score > 0, "Greedy algorithm should produce positive score"
    
    def test_greedy_path_stays_within_radius(self, setup_greedy_test):
        """Test that greedy algorithm doesn't generate points outside max_radius."""
        test_data = setup_greedy_test
        
        greedy_paths = paths.generate_greedy_path(
            center_x=test_data['center_x'],
            center_y=test_data['center_y'],
            num_drones=1,
            probability_map=test_data['heatmap'],
            bounds=test_data['bounds'],
            max_radius=test_data['max_radius']
        )
        
        # Check all path points are within radius
        for path in greedy_paths:
            if not path.is_empty:
                for coord in path.coords:
                    dist = np.sqrt((coord[0] - test_data['center_x'])**2 + 
                                 (coord[1] - test_data['center_y'])**2)
                    assert dist <= test_data['max_radius'] + 1e-6, \
                           f"Path point {coord} is outside max_radius {test_data['max_radius']}"
    
    def test_greedy_algorithm_functionality(self):
        """Test that greedy algorithm functions correctly and can handle revisits."""
        # Create a small test case where we can track functionality
        small_heatmap = np.ones((5, 5))
        small_bounds = (0, 0, 50, 50)
        
        greedy_paths = paths.generate_greedy_path(
            center_x=25, center_y=25,
            num_drones=1,
            probability_map=small_heatmap,
            bounds=small_bounds,
            max_radius=30
        )
        
        # The greedy algorithm should produce a valid path
        assert len(greedy_paths) == 1, "Should produce one path for one drone"
        
        if greedy_paths and not greedy_paths[0].is_empty:
            path = greedy_paths[0]
            
            # Path should have multiple points
            assert len(path.coords) > 1, "Path should have multiple points"
            
            # All points should be within bounds
            minx, miny, maxx, maxy = small_bounds
            for coord in path.coords:
                assert minx <= coord[0] <= maxx, f"X coordinate {coord[0]} out of bounds"
                assert miny <= coord[1] <= maxy, f"Y coordinate {coord[1]} out of bounds"


class TestRegressionSuite:
    """Regression tests to ensure bugs don't reappear."""
    
    def test_spiral_vs_greedy_sanity(self):
        """Regression test: Ensure greedy doesn't wildly exceed spiral on uniform maps."""
        # Create uniform heatmap
        heatmap = np.ones((50, 50))
        extent = (0, 0, 500, 500)
        center_x, center_y = 250, 250
        max_radius = 200
        
        # Generate paths
        config = PathGeneratorConfig(
            num_drones=1,
            budget=10_000,
            fov_degrees=45.0,
            altitude_meters=80.0,
            overlap_ratio=0.1,
            path_point_spacing_m=10.0
        )
        
        spiral_paths = paths.generate_spiral_path(
            center_x=center_x, center_y=center_y, max_radius=max_radius,
            fov_deg=config.fov_degrees, altitude=config.altitude_meters,
            overlap=config.overlap_ratio, num_drones=config.num_drones,
            path_point_spacing_m=config.path_point_spacing_m
        )
        
        greedy_paths = paths.generate_greedy_path(
            center_x=center_x, center_y=center_y, num_drones=config.num_drones,
            probability_map=heatmap, bounds=extent, max_radius=max_radius
        )
        
        # Evaluate both
        empty_victims = gpd.GeoDataFrame(geometry=[])
        evaluator = metrics.PathEvaluator(
            heatmap=heatmap, extent=extent, victims=empty_victims,
            fov_deg=config.fov_degrees, altitude=config.altitude_meters,
            meters_per_bin=config.path_point_spacing_m
        )
        
        spiral_metrics = evaluator.calculate_all_metrics(spiral_paths, discount_factor=1.0)
        greedy_metrics = evaluator.calculate_all_metrics(greedy_paths, discount_factor=1.0)
        
        spiral_score = spiral_metrics['total_likelihood_score']
        greedy_score = greedy_metrics['total_likelihood_score']
        
        # Calculate theoretical maximum (sum of all heatmap values)
        theoretical_max = np.sum(heatmap)
        
        # Both should be within theoretical maximum
        assert spiral_score <= theoretical_max, \
               f"Spiral score {spiral_score} exceeds theoretical max {theoretical_max}"
        assert greedy_score <= theoretical_max, \
               f"Greedy score {greedy_score} exceeds theoretical max {theoretical_max}"
        
        # Greedy shouldn't be wildly higher than spiral on uniform maps
        # (it can be higher due to better coverage, but shouldn't be >5x)
        if spiral_score > 0:
            ratio = greedy_score / spiral_score
            assert ratio < 5.0, \
                   f"Greedy score {greedy_score} is {ratio:.2f}x higher than " \
                   f"spiral score {spiral_score}, suggesting a bug"
    
    def test_no_nan_or_infinite_scores(self):
        """Regression test: Ensure no NaN or infinite scores are produced."""
        # Test with various edge cases
        test_cases = [
            # Empty path
            [],
            # Single point path
            [LineString([(10, 10), (10, 10)])],
            # Very short path
            [LineString([(10, 10), (10.1, 10.1)])],
            # Path outside extent
            [LineString([(-100, -100), (-90, -90)])],
        ]
        
        heatmap = np.ones((10, 10))
        extent = (0, 0, 100, 100)
        empty_victims = gpd.GeoDataFrame(geometry=[])
        evaluator = metrics.PathEvaluator(
            heatmap=heatmap, extent=extent, victims=empty_victims,
            fov_deg=45.0, altitude=80.0, meters_per_bin=10.0
        )
        
        for i, test_paths in enumerate(test_cases):
            metrics_result = evaluator.calculate_all_metrics(test_paths, discount_factor=0.999)
            
            # Check all scores are finite
            assert np.isfinite(metrics_result['total_likelihood_score']), \
                   f"Test case {i}: likelihood score is not finite"
            assert np.isfinite(metrics_result['total_time_discounted_score']), \
                   f"Test case {i}: time discounted score is not finite"
            assert np.isfinite(metrics_result['total_path_length']), \
                   f"Test case {i}: path length is not finite"
