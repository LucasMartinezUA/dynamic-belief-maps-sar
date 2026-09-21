#!/usr/bin/env python3
"""A2 microbenchmark: isolate value-map cost from per-UAV selection.

Frozen states are collected from one real mission (DS1). For each state:
  t_map(N)   = _compute_score_map + (N-1) Bellman backups, N in {1..7}
  t_select(U)= per-UAV target selection over a cached V, U in {1,2,5,10,20}

Each measurement repeats 50 times after 2 warm-ups; results are averaged over
the frozen states. No shadow tree is involved (planning profile n8 core ops).

Usage:
  pixi run python scripts/a2_microbench.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sarenv.analytics.simulation import SARSimulation
from sarenv.core.loading import DatasetLoader


HORIZONS = [1, 2, 3, 4, 5, 6, 7]
FLEETS = [1, 2, 5, 10, 20]
REPEATS = 50
WARMUP = 2
STATE_STEPS = [100, 500, 1000, 2000, 3000]


def _timed(fn, repeats=REPEATS, warmup=WARMUP) -> float:
    for _ in range(warmup):
        fn()
    best = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best.append(time.perf_counter() - start)
    return float(np.mean(best))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/audit_block_A/a2_microbench.csv"))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()

    loader = DatasetLoader(f"sarenv_dataset/{args.dataset}")
    item = loader.load_environment("xlarge")
    sim = SARSimulation(
        dataset_item=item,
        num_drones=5,
        num_victims=5,
        budget=200_000.0,
        planning_mode="dynamic",  # cheap 1-step just to harvest realistic states
        seed=42,
    )
    sim.setup()

    states = []
    for step in range(1, max(STATE_STEPS) + 1):
        sim.step(1.0)
        if step in STATE_STEPS:
            states.append({
                "step": step,
                "current_map": sim.dynamic_heatmap.get_current_map().copy(),
                "observed": set(sim.globally_observed_cells),
                "drone_positions": [p for p in sim.drone_positions],
            })
    print(f"harvested {len(states)} frozen states from mission steps {STATE_STEPS}")

    # --- t_map(N): score map + (N-1) backups ---
    rows = []
    for state in states:
        sim.globally_observed_cells = state["observed"]
        # precompute kernel-derived artifacts once per state (not timed)
        for n in HORIZONS:
            def map_op(n=n):
                score_map = sim._compute_score_map(state["current_map"])
                value = score_map
                for _ in range(n - 1):
                    value = score_map + sim._maximum_neighbor_filter(value)
                return value
            elapsed = _timed(map_op, repeats=args.repeats)
            rows.append({"kind": "map", "state_step": state["step"], "N": n,
                         "U": None, "seconds": elapsed})
        print(f"  state step {state['step']}: t_map done")

    # --- t_select(U): target selection over a cached V (N=3) ---
    sim.globally_observed_cells = states[0]["observed"]
    score_map = sim._compute_score_map(states[0]["current_map"])
    V = score_map
    for _ in range(2):
        V = score_map + sim._maximum_neighbor_filter(V)
    sim._cached_V = V

    for state in states:
        positions = state["drone_positions"]
        for u in FLEETS:
            # pad/truncate positions to u drones
            fleet_pos = [positions[i % len(positions)] for i in range(u)]

            def select_op(u=u, fleet_pos=fleet_pos):
                for wx, wy in fleet_pos:
                    row, col = sim._world_to_grid(wx, wy)
                    neighbors = sim._get_valid_neighbors(row, col)
                    if not neighbors:
                        continue
                    best = neighbors[0]
                    best_score = sim._cached_V[best]
                    for nr, nc in neighbors[1:]:
                        if sim._cached_V[nr, nc] > best_score:
                            best_score = sim._cached_V[nr, nc]
                            best = (nr, nc)
            elapsed = _timed(select_op, repeats=args.repeats)
            rows.append({"kind": "select", "state_step": state["step"], "N": 3,
                         "U": u, "seconds": elapsed})
        print(f"  state step {state['step']}: t_select done")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    map_summary = (
        df[df.kind == "map"].groupby("N")["seconds"].agg(["mean", "std"])
        .rename(columns={"mean": "t_map_s", "std": "sd"})
        .round(6)
    )
    sel_summary = (
        df[df.kind == "select"].groupby("U")["seconds"].agg(["mean", "std"])
        .rename(columns={"mean": "t_select_s", "std": "sd"})
        .round(6)
    )
    print("\n=== t_map(N): score map + (N-1) backups (mean over 5 states) ===")
    print(map_summary.to_string())
    print("\n=== t_select(U): per-UAV selection over cached V (N=3) ===")
    print(sel_summary.to_string())
    t_select_10 = float(sel_summary.loc[10, "t_select_s"])
    t_map_3 = float(map_summary.loc[3, "t_map_s"])
    print(f"\nratio t_select(U=10) / t_map(N=3) = {t_select_10 / t_map_3:.4f}")


if __name__ == "__main__":
    main()
