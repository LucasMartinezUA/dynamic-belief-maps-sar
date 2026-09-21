# sarenv/analytics/midflight.py
"""Pure-geometry helpers for C7 fault-triggered pizza repartitioning.

No probability source is ever accessed: these functions operate exclusively on
geometry (waypoint coordinates), the valid-domain mask, the observed-cell set,
and the FOV visible-cell function. All functions are deterministic (no RNG).

The replan pipeline (used by ``SARSimulation._replan_pizza_midflight``):

1. generate K pizza sectors (same generator/params as the frozen C4 pizza arm);
2. ``waypoint_has_unseen_cell`` per waypoint -> ``trim_covered_tails`` removes
   fully-observed prefix/suffix waypoints (interior covered waypoints KEPT so
   the path stays physically continuous);
3. ``split_route`` splits the longest remaining route into contiguous halves
   until every survivor has a task (min segment length 20 m);
4. ``min_entry_cost_assignment`` (Hungarian + deterministic lexicographic
   tie-break) assigns tasks to survivor positions with min-entry orientation;
5. ``build_transit_path`` prepends the survivor position (physical transit;
   no teleport).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString


def waypoint_has_unseen_cell(
    wx: float,
    wy: float,
    *,
    visible_fn,
    valid_mask: np.ndarray,
    observed: set,
    shape: tuple,
) -> bool:
    """True if the waypoint's FOV contains a valid, unobserved cell."""
    for r, c in visible_fn(wx, wy):
        if 0 <= r < shape[0] and 0 <= c < shape[1] and valid_mask[r, c] and (r, c) not in observed:
            return True
    return False


def trim_covered_tails(coords: list, useful: list) -> tuple:
    """Remove fully-observed prefix/suffix waypoints; keep interior covered.

    Returns ``(remaining, prefix_removed, suffix_removed)``. ``remaining`` is
    ``coords[first:last+1]`` where ``first``/``last`` are the first/last index
    with ``useful == True``; empty if there are no useful waypoints.
    """
    if not coords:
        return [], 0, 0
    first = None
    last = None
    for idx, flag in enumerate(useful):
        if flag:
            if first is None:
                first = idx
            last = idx
    if first is None:
        return [], len(coords), 0
    return (
        [tuple(float(v) for v in point) for point in coords[first : last + 1]],
        int(first),
        int(len(coords) - last - 1),
    )


def split_route(coords: list, n: int) -> list:
    """Split ``coords`` into ``n`` contiguous sub-paths by cumulative length.

    Boundary waypoint indexes are chosen by rounding cumulative-length
    fractions (``target_k = total * k / n``). Duplicate boundaries are merged,
    so fewer than ``n`` segments are returned for degenerate short routes.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if not coords:
        return []
    points = [tuple(float(v) for v in point) for point in coords]
    cum = [0.0]
    for prev, nxt in zip(points, points[1:]):
        cum.append(cum[-1] + float(np.hypot(nxt[0] - prev[0], nxt[1] - prev[1])))
    total = cum[-1]
    boundaries = [0]
    if total > 0.0:
        for k in range(1, n):
            target = total * k / n
            idx = int(np.argmin(np.abs(np.asarray(cum, dtype=float) - target)))
            boundaries.append(idx)
    else:
        step = max(1, len(points) // n)
        for k in range(1, n):
            boundaries.append(min(k * step, len(points) - 1))
    boundaries.append(len(points) - 1)
    boundaries = sorted(set(boundaries))
    segments = []
    for a, b in zip(boundaries, boundaries[1:]):
        segments.append(list(points[a : b + 1]))
    return segments


def route_length(coords: list) -> float:
    """Polyline length of ``coords`` in meters."""
    total = 0.0
    for prev, nxt in zip(coords, coords[1:]):
        total += float(np.hypot(nxt[0] - prev[0], nxt[1] - prev[1]))
    return total


def min_entry_cost_assignment(positions: list, tasks: list) -> tuple:
    """Assign tasks to positions by minimum straight-line entry cost.

    Cost basis: distance from the position to the task's entry waypoint, where
    the entry is the first waypoint (forward orientation) or the last waypoint
    (reverse orientation); the cheaper orientation is selected per (position,
    task). Rectangular form: ``len(tasks) <= len(positions)`` is allowed, so
    when fewer tasks than survivors remain, the assignment also CHOOSES which
    survivors get a task (the unassigned ones become search-inactive) - the
    Hungarian optimum over all injective task->survivor matchings; ties are
    broken deterministically by enumerating every injective matching (<= 120
    for a 5-survivor fleet) whose cost equals the optimum (relative tolerance
    1e-9) and selecting the lexicographically smallest canonical tuple
    (sorted (survivor_index, task_index) pairs).

    Returns ``(assignment, cost)`` where ``assignment`` maps the assigned
    position index to ``(task_index, orientation)`` and ``orientation`` is
    ``"forward"`` or ``"reverse"``. Unassigned position indices are absent
    (the caller converts them to search-inactive).
    """
    if len(tasks) > len(positions):
        raise ValueError(
            f"tasks ({len(tasks)}) cannot exceed positions ({len(positions)})"
        )
    total = len(tasks)
    if total == 0:
        return {}, 0.0
    pos_xy = [(float(p[0]), float(p[1])) for p in positions]
    # costs[survivor, task]
    costs = np.zeros((len(pos_xy), total), dtype=np.float64)
    orient = {}
    for i, (px, py) in enumerate(pos_xy):
        for j, task in enumerate(tasks):
            d_f = float(np.hypot(task[0][0] - px, task[0][1] - py))
            d_r = float(np.hypot(task[-1][0] - px, task[-1][1] - py))
            costs[i, j] = min(d_f, d_r)
            orient[(i, j)] = "forward" if d_f <= d_r else "reverse"
    # Hungarian on the T x M view: every task is matched, survivors may be left
    # unmatched (rows=tasks, columns=survivors).
    task_mat = costs.T
    row_idx, col_idx = linear_sum_assignment(task_mat)
    optimum = float(task_mat[row_idx, col_idx].sum())
    tolerance = 1e-9 * max(abs(optimum), 1.0)

    def canonical(f_map):
        # f_map: task j -> survivor g[j]; canonical = sorted (survivor, task)
        return tuple(sorted((int(f_map[j]), int(j)) for j in range(total)))

    best = None
    best_pairs = None
    for injective in itertools.permutations(range(len(pos_xy)), total):
        candidate = float(
            sum(costs[injective[j], j] for j in range(total))
        )
        if abs(candidate - optimum) <= tolerance:
            pairs = canonical(injective)
            if best_pairs is None or pairs < best_pairs:
                best_pairs = pairs
                best = injective
    assignment = {
        int(best[j]): (int(j), orient[(int(best[j]), int(j))])
        for j in range(total)
    }
    return assignment, optimum


def hold_line() -> LineString:
    """Empty LineString used for failed/search-inactive (no move)."""
    return LineString()


def build_transit_path(position: tuple, task_coords: list, orientation: str) -> LineString:
    """Build the post-replan path: transit segment + task in orientation.

    The first coordinate is the current position (no teleport); the task
    coordinates follow in the requested orientation.
    """
    coords = (
        list(task_coords)
        if orientation == "forward"
        else list(reversed(task_coords))
    )
    return LineString([tuple(float(v) for v in position)] + coords)
