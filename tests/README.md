# Test Suite for SAR Dataset Project

This directory contains unit tests for the SAR dataset project, with a focus on testing the PathEvaluator class to ensure cumulative likelihoods don't double-count areas when multiple paths visit the same cells.

## Running Tests

### Prerequisites

Install the test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

### Run All Tests

```bash
# From the project root directory
python -m pytest tests/ -v
```

### Run Specific Test File

```bash
python -m pytest tests/test_path_evaluator.py -v
```

### Run Specific Test

```bash
python -m pytest tests/test_path_evaluator.py::TestPathEvaluatorCumulativeLikelihoods::test_identical_paths_no_double_counting_total -v
```

### Run with Coverage

```bash
python -m pytest tests/ --cov=sarenv --cov-report=html
```

## Test Structure

### TestPathEvaluatorCumulativeLikelihoods

Tests the core functionality of cumulative likelihood calculations:

- **test_single_path_baseline**: Verifies basic single path behavior
- **test_identical_paths_no_double_counting_total**: Ensures identical overlapping paths don't double-count in total likelihood
- **test_identical_paths_cumulative_likelihoods**: Verifies proper handling of cumulative likelihoods for overlapping paths
- **test_partially_overlapping_paths**: Tests partial overlap scenarios
- **test_separate_paths_full_contribution**: Ensures non-overlapping paths get full individual contributions
- **test_empty_path_handling**: Tests handling of empty/invalid paths
- **test_zero_length_path_handling**: Tests handling of zero-length paths
- **test_cumulative_monotonicity**: Ensures cumulative values increase monotonically
- **test_multiple_paths_order_independence**: Verifies that path order doesn't affect total likelihood for non-overlapping paths

### TestPathEvaluatorEdgeCases

Tests edge cases and boundary conditions:

- **test_no_paths**: Tests behavior with empty path lists
- **test_single_cell_coverage**: Tests path behavior within single cells

## Key Testing Principles

### Double-Counting Prevention

The core principle being tested is that areas in the probability map should not be counted multiple times when multiple paths visit the same cells:

1. **Total Likelihood**: Should represent the sum of unique cells visited across all paths
2. **Cumulative Likelihoods**: Each path's cumulative should only include new areas not yet visited by previous paths
3. **Order Independence**: For non-overlapping paths, the total likelihood should be the same regardless of path order

### Test Scenarios

1. **Identical Paths**: Second path should contribute 0 to cumulative likelihood
2. **Partial Overlap**: Each path should only get credit for new areas it covers
3. **No Overlap**: Each path should get full credit for its areas
4. **Edge Cases**: Empty paths, zero-length paths, single cells

## Configuration

The tests use pytest with the following configuration (see `pytest.ini`):

- Warnings filtered for geopandas/shapely deprecation warnings
- Verbose output by default
- Short traceback format

## Fixtures

- **setup_evaluator**: Creates a PathEvaluator with a 3x3 test heatmap
- **setup_single_cell_evaluator**: Creates a PathEvaluator with a 2x2 test heatmap for edge case testing
