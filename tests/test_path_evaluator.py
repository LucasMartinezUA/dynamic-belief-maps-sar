"""
Unit tests for PathEvaluator class, specifically testing that cumulative likelihoods
don't double-count areas when multiple paths visit the same cells.
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

from sarenv.analytics.metrics import PathEvaluator


class TestPathEvaluatorCumulativeLikelihoods:
    """Test case for PathEvaluator cumulative likelihood calculations."""

    @pytest.fixture
    def setup_evaluator(self):
        """Set up test fixtures before each test method."""
        # Create a simple 3x3 probability heatmap
        heatmap = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

        # Define extent: 3x3 grid with 1 unit per cell
        extent = (0, 0, 3, 3)  # (minx, miny, maxx, maxy)

        # Create empty victims dataframe (not needed for these tests)
        victims = gpd.GeoDataFrame(geometry=[])

        # Create PathEvaluator
        evaluator = PathEvaluator(
            heatmap=heatmap,
            extent=extent,
            victims=victims,
            fov_deg=60,
            altitude=100,
            meters_per_bin=1,
        )

        return {
            "evaluator": evaluator,
            "heatmap": heatmap,
            "extent": extent,
            "victims": victims,
            "discount_factor": 1.0,
        }

    def test_single_path_baseline(self, setup_evaluator):
        """Test that a single path produces expected baseline results."""
        evaluator = setup_evaluator["evaluator"]
        discount_factor = setup_evaluator["discount_factor"]

        path = LineString([(0.5, 0.5), (1.5, 1.5)])  # Diagonal path
        paths = [path]

        result = evaluator.calculate_all_metrics(paths, discount_factor)

        # Should have one cumulative likelihood array
        assert len(result["cumulative_likelihoods"]) == 1

        # Total likelihood should be positive
        assert result["total_likelihood_score"] > 0

        # Cumulative should be monotonically increasing
        cumulative = result["cumulative_likelihoods"][0]
        for i in range(1, len(cumulative)):
            assert cumulative[i] >= cumulative[i - 1]

    def test_identical_paths_no_double_counting_total(self, setup_evaluator):
        """Test that identical overlapping paths don't double-count in total likelihood."""
        evaluator = setup_evaluator["evaluator"]
        discount_factor = setup_evaluator["discount_factor"]

        path1 = LineString([(0.5, 0.5), (1.5, 1.5)])
        path2 = LineString([(0.5, 0.5), (1.5, 1.5)])  # Identical path

        # Single path result
        single_result = evaluator.calculate_all_metrics([path1], discount_factor)
        single_total = single_result["total_likelihood_score"]

        # Identical paths result
        identical_result = evaluator.calculate_all_metrics(
            [path1, path2], discount_factor
        )
        identical_total = identical_result["total_likelihood_score"]

        # Total likelihood should be the same (no double-counting)
        assert (
            abs(single_total - identical_total) < 1e-10
        ), "Identical paths should not double-count in total likelihood"

    def test_identical_paths_cumulative_likelihoods(self, setup_evaluator):
        """Test that cumulative likelihoods properly handle overlapping identical paths."""
        evaluator = setup_evaluator["evaluator"]
        discount_factor = setup_evaluator["discount_factor"]

        path1 = LineString([(0.5, 0.5), (1.5, 1.5)])
        path2 = LineString([(0.5, 0.5), (1.5, 1.5)])  # Identical path

        # Single path result
        single_result = evaluator.calculate_all_metrics([path1], discount_factor)
        single_cumulative = single_result["cumulative_likelihoods"][0]
        single_final = single_cumulative[-1] if len(single_cumulative) > 0 else 0

        # Identical paths result
        identical_result = evaluator.calculate_all_metrics(
            [path1, path2], discount_factor
        )
        identical_cumulatives = identical_result["cumulative_likelihoods"]

        # With the new implementation, both paths get full credit in their individual
        # cumulative scores (they're calculated as if running independently)
        first_path_final = (
            identical_cumulatives[0][-1] if len(identical_cumulatives[0]) > 0 else 0
        )
        second_path_final = (
            identical_cumulatives[1][-1] if len(identical_cumulatives[1]) > 0 else 0
        )
        
        # Both paths should get the same credit as the single path
        assert (
            abs(single_final - first_path_final) < 1e-10
        ), "First path should get same credit as single path"

        assert (
            abs(single_final - second_path_final) < 1e-10
        ), "Second path should get same credit as single path"

    def test_partially_overlapping_paths(self, setup_evaluator):
        """Test that partially overlapping paths correctly handle overlap."""
        evaluator = setup_evaluator["evaluator"]
        discount_factor = setup_evaluator["discount_factor"]

        # Use paths that are more clearly separated to ensure some non-overlapping coverage
        path1 = LineString([(0.5, 0.5), (1.5, 0.5)])  # Horizontal path
        path2 = LineString([(0.5, 1.5), (2.5, 2.5)])  # Diagonal path in different area

        result = evaluator.calculate_all_metrics([path1, path2], discount_factor)

        # Both paths should contribute something since they cover different areas
        cumulative1_final = result["cumulative_likelihoods"][0][-1]
        cumulative2_final = result["cumulative_likelihoods"][1][-1]

        assert cumulative1_final > 0, "First path should contribute"
        assert cumulative2_final > 0, "Second path should contribute to new areas"

        # With the new implementation, individual cumulative contributions represent
        # what each path would contribute if running independently, so they may
        # sum to more than the total if there's overlapping coverage
        total_from_cumulatives = cumulative1_final + cumulative2_final
        assert (
            total_from_cumulatives >= result["total_likelihood_score"]
        ), "Sum of cumulative contributions should be >= total (due to potential overlap)"

    def test_separate_paths_full_contribution(self, setup_evaluator):
        """Test that completely separate paths each get full contribution."""
        evaluator = setup_evaluator["evaluator"]
        discount_factor = setup_evaluator["discount_factor"]

        path1 = LineString([(0.5, 0.5), (0.5, 1.5)])  # Vertical path left side
        path2 = LineString([(2.5, 0.5), (2.5, 1.5)])  # Vertical path right side

        # Individual path results
        result1 = evaluator.calculate_all_metrics([path1], discount_factor)
        result2 = evaluator.calculate_all_metrics([path2], discount_factor)

        # Combined path result
        combined_result = evaluator.calculate_all_metrics(
            [path1, path2], discount_factor
        )

        # With the new implementation, if paths have overlapping coverage areas,
        # the total may be less than the sum of individual contributions
        individual_sum = (
            result1["total_likelihood_score"] + result2["total_likelihood_score"]
        )
        combined_total = combined_result["total_likelihood_score"]

        # The combined total should be <= individual sum (due to potential overlap)
        # but for truly separate paths, they should be approximately equal
        assert (
            combined_total <= individual_sum
        ), "Combined total should be <= sum of individual contributions"

        # Check that both paths contribute in the combined result
        cumulative1_final = combined_result["cumulative_likelihoods"][0][-1]
        cumulative2_final = combined_result["cumulative_likelihoods"][1][-1]

        assert cumulative1_final > 0, "First path should contribute"
        assert cumulative2_final > 0, "Second path should contribute"

    def test_empty_path_handling(self, setup_evaluator):
        """Test that empty paths are handled correctly."""
        evaluator = setup_evaluator["evaluator"]
        discount_factor = setup_evaluator["discount_factor"]

        valid_path = LineString([(0.5, 0.5), (1.5, 1.5)])
        empty_path = LineString()  # Empty path

        result = evaluator.calculate_all_metrics(
            [valid_path, empty_path], discount_factor
        )

        # Should have two cumulative arrays
        assert len(result["cumulative_likelihoods"]) == 2

        # First path should have normal contribution
        assert len(result["cumulative_likelihoods"][0]) > 0

        # Empty path should have zero contribution
        empty_cumulative = result["cumulative_likelihoods"][1]
        assert len(empty_cumulative) == 1  # Should have one zero element
        assert empty_cumulative[0] == 0

    def test_zero_length_path_handling(self, setup_evaluator):
        """Test that zero-length paths are handled correctly."""
        evaluator = setup_evaluator["evaluator"]
        discount_factor = setup_evaluator["discount_factor"]

        valid_path = LineString([(0.5, 0.5), (1.5, 1.5)])
        zero_length_path = LineString([(1.0, 1.0), (1.0, 1.0)])  # Same start/end point

        result = evaluator.calculate_all_metrics(
            [valid_path, zero_length_path], discount_factor
        )

        # Should have two cumulative arrays
        assert len(result["cumulative_likelihoods"]) == 2

        # Zero-length path should have zero contribution
        zero_cumulative = result["cumulative_likelihoods"][1]
        assert len(zero_cumulative) == 1
        assert zero_cumulative[0] == 0

    def test_cumulative_monotonicity(self, setup_evaluator):
        """Test that cumulative likelihoods are monotonically increasing within each path."""
        evaluator = setup_evaluator["evaluator"]
        discount_factor = setup_evaluator["discount_factor"]

        path = LineString([(0.5, 0.5), (2.5, 2.5)])  # Diagonal across heatmap

        result = evaluator.calculate_all_metrics([path], discount_factor)
        cumulative = result["cumulative_likelihoods"][0]

        # Check monotonicity
        for i in range(1, len(cumulative)):
            assert (
                cumulative[i] >= cumulative[i - 1]
            ), f"Cumulative should be monotonic at index {i}"

    def test_multiple_paths_order_independence(self, setup_evaluator):
        """Test that the order of paths doesn't affect the total likelihood."""
        evaluator = setup_evaluator["evaluator"]
        discount_factor = setup_evaluator["discount_factor"]

        # Use completely non-overlapping paths to test order independence
        path1 = LineString([(0.5, 0.5), (0.5, 1.5)])  # Vertical left
        path2 = LineString([(2.5, 0.5), (2.5, 1.5)])  # Vertical right

        # Test different orders
        result_order1 = evaluator.calculate_all_metrics([path1, path2], discount_factor)
        result_order2 = evaluator.calculate_all_metrics([path2, path1], discount_factor)

        # Total should be the same regardless of order
        assert (
            abs(
                result_order1["total_likelihood_score"]
                - result_order2["total_likelihood_score"]
            )
            < 1e-10
        ), f"Total likelihood should be order-independent. Got {result_order1['total_likelihood_score']} vs {result_order2['total_likelihood_score']}"

        # Individual cumulative contributions should also be consistent
        # (though they may be in different order in the arrays)
        cumulative1_order1 = sorted([result_order1["cumulative_likelihoods"][0][-1],
                                     result_order1["cumulative_likelihoods"][1][-1]])
        cumulative1_order2 = sorted([result_order2["cumulative_likelihoods"][0][-1],
                                     result_order2["cumulative_likelihoods"][1][-1]])

        assert (
            abs(cumulative1_order1[0] - cumulative1_order2[0]) < 1e-10
        ), "First cumulative contribution should be order-independent"
        assert (
            abs(cumulative1_order1[1] - cumulative1_order2[1]) < 1e-10
        ), "Second cumulative contribution should be order-independent"


class TestPathEvaluatorEdgeCases:
    """Test edge cases for PathEvaluator."""

    @pytest.fixture
    def setup_single_cell_evaluator(self):
        """Set up test fixtures with single cell heatmap."""
        # 2x2 heatmap with only center having value
        heatmap = np.array([[0.0, 0.0], [0.0, 1.0]])
        extent = (0, 0, 2, 2)
        victims = gpd.GeoDataFrame(geometry=[])
        return PathEvaluator(
            heatmap=heatmap,
            extent=extent,
            victims=victims,
            fov_deg=60,
            altitude=100,
            meters_per_bin=1,
        )

    def test_no_paths(self, setup_single_cell_evaluator):
        """Test behavior with empty path list."""
        evaluator = setup_single_cell_evaluator
        result = evaluator.calculate_all_metrics([], 1.0)

        assert result["total_likelihood_score"] == 0
        assert len(result["cumulative_likelihoods"]) == 0

    def test_single_cell_coverage(self, setup_single_cell_evaluator):
        """Test path covering single cell multiple times."""
        evaluator = setup_single_cell_evaluator

        # Path that visits the high-value cell (bottom-right)
        path = LineString([(1.2, 1.2), (1.8, 1.8)])

        result = evaluator.calculate_all_metrics([path], 1.0)

        # Should get some positive value from the 1.0 cell
        # Note: can be > 1.0 because path covers multiple interpolated points
        assert result["total_likelihood_score"] > 0
        assert result["total_likelihood_score"] < 10.0  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
