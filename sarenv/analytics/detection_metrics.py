# sarenv/analytics/detection_metrics.py
"""Exact stationary single-victim detection metrics (Block B, handoff §16/§17).

Given a search trajectory (per-drone world positions), these functions integrate
the sensor noise analytically for a single stationary victim sampled from the
prior. ``expected_detection`` is the exact probability of at least one detection
before the horizon; ``expected_rmst`` is the exact expected restricted mean
survival time (censored time-to-detection). Sensor independence and multiplicity
(k drones in a cell in one timestep) are preserved.
"""
from __future__ import annotations

import numpy as np


def exposure_counts(drone_paths: list, visible_fn, shape: tuple) -> np.ndarray:
    """Total independent sensor exposures per cell (sum over drone x timestep).

    This is ``n_c`` in handoff §17: the number of times cell ``c`` was observed.
    """
    n_cells = shape[0] * shape[1]
    counts = np.zeros(n_cells, dtype=np.int64)
    for path in drone_paths:
        for wx, wy in path:
            for r, c in visible_fn(wx, wy):
                if 0 <= r < shape[0] and 0 <= c < shape[1]:
                    counts[r * shape[1] + c] += 1
    return counts.reshape(shape)


def expected_detection(prior_flat: np.ndarray, n_c: np.ndarray, p_d: float) -> float:
    """Exact P(detected at least once) for a stationary victim from the prior.

    ``ExpectedDetection = sum_c P_prior(c) * [1 - (1 - p_d)^n_c]``.
    """
    n_c = np.asarray(n_c, dtype=np.float64).ravel()
    if n_c.size == 0:
        return 0.0
    return float(np.sum(prior_flat * (1.0 - (1.0 - p_d) ** n_c)))


def exposure_events(drone_paths: list, visible_fn, shape: tuple):
    """Flatten per-cell exposure events sorted by ``(cell, timestep)``.

    Returns ``(cell_flat_sorted, timestep_sorted, offsets)`` where ``offsets`` are
    ``np.searchsorted`` boundaries so that cell ``c`` occupies
    ``timestep_sorted[offsets[c]:offsets[c+1]]`` (ascending timesteps).

    The sort is ``np.lexsort((timestep, cell))``: a stable ``argsort`` by cell
    alone does NOT guarantee ascending timestamps within a cell, because events
    are collected by walking each drone path in turn, so a cell can receive
    interleaved timestamps (e.g. ``[0,5,10,1,4,8]``) from two drones.
    """
    n_cells = shape[0] * shape[1]
    cells = []
    ts = []
    for path in drone_paths:
        for t, (wx, wy) in enumerate(path):
            for r, c in visible_fn(wx, wy):
                if 0 <= r < shape[0] and 0 <= c < shape[1]:
                    cells.append(r * shape[1] + c)
                    ts.append(t)
    cells = np.asarray(cells, dtype=np.int32) if cells else np.empty(0, dtype=np.int32)
    ts = np.asarray(ts, dtype=np.int32) if ts else np.empty(0, dtype=np.int32)
    if cells.size == 0:
        return (cells, ts, np.zeros(n_cells + 1, dtype=np.int32))
    order = np.lexsort((ts, cells))
    cells_sorted = cells[order]
    ts_sorted = ts[order]
    offsets = np.searchsorted(cells_sorted, np.arange(n_cells + 1, dtype=np.int32))
    return cells_sorted, ts_sorted, offsets


def _cell_rmst(ts_c: np.ndarray, q: float, T: int) -> float:
    """RMST for one cell with sorted (non-decreasing) exposure timesteps.

    Computes ``sum_{t=0}^{T} q^{sum_{s<=t} k_s(c)}`` in closed form by grouping
    equal timesteps (sensor multiplicity) into cumulative exposure counts.
    """
    m = len(ts_c)
    if m == 0:
        return float(T + 1)
    rmst = 0.0
    count = 0
    prev_tau = 0
    i = 0
    while i < m:
        tau = int(ts_c[i])
        if tau > prev_tau:
            rmst += (tau - prev_tau) * (q ** count)
        while i < m and int(ts_c[i]) == tau:
            count += 1
            i += 1
        prev_tau = tau
    rmst += (T - prev_tau + 1) * (q ** count)
    return rmst


def expected_rmst(
    prior_flat: np.ndarray,
    cell_flat_sorted: np.ndarray,
    timestep_sorted: np.ndarray,
    offsets: np.ndarray,
    p_d: float,
    T: int,
) -> float:
    """Exact expected restricted mean survival time (censored TTD).

    ``RMST = sum_c P_prior(c) * sum_{t=0}^{T} q^{sum_{s<=t} k_s(c)}``; a cell with
    no exposures contributes ``T + 1``. ``T`` is the last timestep index (number of
    drone-path positions minus 1), so timesteps are ``0..T`` inclusive.
    """
    q = 1.0 - p_d
    n_cells = prior_flat.size
    total = 0.0
    for c in range(n_cells):
        lo = int(offsets[c])
        hi = int(offsets[c + 1])
        if lo == hi:
            total += prior_flat[c] * (T + 1)
        else:
            total += prior_flat[c] * _cell_rmst(timestep_sorted[lo:hi], q, T)
    return float(total)


def stationary_victim_cell(prior_flat: np.ndarray, seed: int) -> int:
    """Sample one cell from the prior PMF — the matched-prior single victim."""
    p = np.asarray(prior_flat, dtype=np.float64)
    p = p / p.sum()   # dataset heatmaps sum to ~0.9997, not exactly 1
    rng = np.random.default_rng(seed)
    return int(rng.choice(p.size, p=p))
