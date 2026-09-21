# tests/test_detection_metrics.py
"""Unit + Monte Carlo tests for Block B stationary detection metrics."""
import numpy as np
import pytest

from sarenv.analytics.detection_metrics import (
    exposure_counts,
    exposure_events,
    expected_detection,
    expected_rmst,
    stationary_victim_cell,
)


def test_exact_unit():
    """1x2 grid: drone visits cell0 at t=0, cell1 at t=1."""
    def visible_fn(wx, wy):
        return {(0, 0)} if wx == 0.0 else {(0, 1)}

    drone_paths = [[(0.0, 0.0), (1.0, 0.0)]]  # t=0 at A, t=1 at B
    shape = (1, 2)
    prior = np.array([0.5, 0.5])
    p_d = 0.8
    T = 1

    counts = exposure_counts(drone_paths, visible_fn, shape)
    np.testing.assert_array_equal(counts.ravel(), [1, 1])
    assert np.isclose(expected_detection(prior, counts.ravel(), p_d), 0.8)

    cells, ts, offsets = exposure_events(drone_paths, visible_fn, shape)
    # cell 0 exposed only at t=0 -> RMST_0 = 0.4; cell 1 only at t=1 -> RMST_1 = 1.2
    assert np.isclose(expected_rmst(prior, cells, ts, offsets, p_d, T), 0.8)


def test_multiplicity():
    def visible_fn(wx, wy):
        return {(0, 0)}

    drone_paths = [[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (0.0, 0.0)]]
    shape = (1, 1)
    prior = np.array([1.0])
    p_d = 0.8

    counts = exposure_counts(drone_paths, visible_fn, shape)
    assert counts[0, 0] == 4
    assert np.isclose(expected_detection(prior, counts.ravel(), p_d), 1 - (1 - 0.8) ** 4)


def test_cross_drone_ordering():
    """Cell (0,0) receives interleaved timestamps from two drones; lexsort fixes order."""
    def visible_fn(wx, wy):
        return {(0, 0)} if wx == 0.0 else {(0, 1)}

    P = (0.0, 0.0)   # -> cell (0,0)
    Q = (1.0, 0.0)   # -> cell (0,1)
    T = 10
    drone1 = [P if t in {0, 5, 10} else Q for t in range(T + 1)]
    drone2 = [P if t in {1, 4, 8} else Q for t in range(T + 1)]
    drone_paths = [drone1, drone2]
    shape = (1, 2)
    prior = np.array([1.0, 0.0])  # all mass in cell (0,0)
    p_d = 0.8
    q = 1 - p_d

    counts = exposure_counts(drone_paths, visible_fn, shape)
    assert counts[0, 0] == 6
    assert np.isclose(expected_detection(prior, counts.ravel(), p_d), 1 - q ** 6)

    cells, ts, offsets = exposure_events(drone_paths, visible_fn, shape)
    cell0_ts = ts[offsets[0]:offsets[1]]
    np.testing.assert_array_equal(cell0_ts, [0, 1, 4, 5, 8, 10])

    # Direct per-timestep sum for comparison.
    k = np.zeros(T + 1, dtype=np.int64)
    for tau in cell0_ts:
        k[tau] += 1
    S = np.ones(T + 1)
    c = 0
    for t in range(T + 1):
        c += k[t]
        S[t] = q ** c
    direct_rmst = float(S.sum())
    assert np.isclose(expected_rmst(prior, cells, ts, offsets, p_d, T), direct_rmst)


def test_mc_detection():
    """4x4 grid, one exposure per cell; validate exact detection formula."""
    shape = (4, 4)
    prior = np.full(16, 1.0 / 16.0)
    p_d = 0.8

    def visible_fn(wx, wy):
        return {(int(wx), int(wy))}

    drone_paths = [[(r, c) for r in range(4) for c in range(4)]]  # 16 cells, T=15
    counts = exposure_counts(drone_paths, visible_fn, shape)
    exact = expected_detection(prior, counts.ravel(), p_d)
    assert np.all(counts.ravel() == 1)
    assert np.isclose(exact, 0.8)

    M = 200000
    cells_sample = np.random.default_rng(42).choice(16, size=M, p=prior)
    hits = np.random.default_rng(9999).random(M) < (1 - (1 - p_d) ** 1)
    mc = hits.mean()
    bound = 3 * np.sqrt(0.25 / M)
    assert abs(mc - exact) < bound

    # stationary_victim_cell returns an in-range index.
    assert 0 <= stationary_victim_cell(prior, seed=7) < 16


def test_mc_rmst():
    """4x4 grid, one exposure per cell; validate exact RMST (sensor + position MC)."""
    shape = (4, 4)
    prior = np.full(16, 1.0 / 16.0)
    p_d = 0.8
    q = 1 - p_d
    T = 15

    def visible_fn(wx, wy):
        return {(int(wx), int(wy))}

    drone_paths = [[(r, c) for r in range(4) for c in range(4)]]
    cells, ts, offsets = exposure_events(drone_paths, visible_fn, shape)
    exact = expected_rmst(prior, cells, ts, offsets, p_d, T)

    M = 200000
    cells_sample = np.random.default_rng(42).choice(16, size=M, p=prior)
    taus = ts[offsets[cells_sample]].astype(np.float64)  # exposure timestep per cell
    rng = np.random.default_rng(9999)
    hits = rng.random(M) < p_d
    ys = np.where(hits, taus, float(T + 1))  # never detected -> censored at T+1
    mc = ys.mean()
    bound = 3 * (T + 1) / np.sqrt(M)
    assert abs(mc - exact) < bound
