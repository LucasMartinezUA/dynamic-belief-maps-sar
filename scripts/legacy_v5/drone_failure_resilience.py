#!/usr/bin/env python3
# scripts/drone_failure_resilience.py
"""
Drone‑failure resilience analysis (K‑drone loss) — Q1‑ready methodology.

**Static / geometric modes** (pizza, static, static_3step):
   Post‑hoc path analysis.  Paths are deterministic → removing K drone
   paths yields the exact result of K drones failing mid‑mission.
   Fast, no re‑simulation needed.

 **Dynamic modes** (dynamic, dynamic_3step):
    Full SARSimulation runs with N-K drones, preserving the same seed so
    initial positions & victims are identical across drone counts.  The
    remaining drones re‑plan over the evolving posterior heatmap.
    Budget scales proportionally (budget × (N-K)/N) so per‑drone autonomy
    stays constant — modelling the realistic case where battery is per‑drone
    and is lost with the drone.  This is the accurate approach — post‑hoc
    would underestimate resilience.

Usage:
  pixi run python scripts/drone_failure_resilience.py
  pixi run python scripts/drone_failure_resilience.py --dataset 1 5 10 --seeds 42
"""

from __future__ import annotations

import argparse
import re
import sys
import time as _time
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

import sarenv
from sarenv.core.loading import DatasetLoader
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.analytics.simulation import SARSimulation
from sarenv.analytics.paths import (
    generate_greedy_path,
    generate_spiral_path,
    generate_concentric_circles_path,
    generate_pizza_zigzag_path,
)

log = sarenv.get_logger()

# ── Constants ────────────────────────────────────────────────────────────
_GEOMETRIC_BASELINES = {"spiral", "concentric", "pizza"}
_STATIC_MODES = {"pizza", "static", "static_3step"}
_DYNAMIC_MODES = {"dynamic", "dynamic_3step"}

DEFAULT_MODES = ["pizza", "static", "dynamic", "static_3step", "dynamic_3step"]
DEFAULT_DATASETS = [1, 5, 10]          # 3 representative datasets
DEFAULT_SEEDS = [42]                   # single seed for speed
DEFAULT_FAILURE_COUNTS = [1, 2, 3]

DEFAULT_ALTITUDE = 80.0
DEFAULT_INIT_STRATEGY = "random"
DEFAULT_INIT_RADIUS = 100.0
DEFAULT_OVERLAP = 0.1


# ── Helpers (shared) ─────────────────────────────────────────────────────

def _resolve_gen_mode(planning_mode: str) -> str:
    """Map a planning mode to the path‑generator mode."""
    if planning_mode in _GEOMETRIC_BASELINES:
        return planning_mode
    if planning_mode in ("dynamic", "static"):
        return "static"
    if planning_mode in ("dynamic_3step", "static_3step"):
        return "static_3step"
    m = re.match(r"^(?:dynamic|static)_(\d+)step$", planning_mode)
    if m:
        return f"static_{m.group(1)}step"
    raise ValueError(f"Unknown planning mode: {planning_mode}")


def _initial_positions(
    num_drones: int,
    center_x: float, center_y: float,
    bounds: tuple[float, float, float, float],
    init_strategy: str, init_radius: float, seed: int,
) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bounds
    positions: list[tuple[float, float]] = []
    if init_strategy == "circle":
        for i in range(num_drones):
            angle = 2 * np.pi * i / num_drones
            x = np.clip(center_x + init_radius * np.cos(angle), minx, maxx)
            y = np.clip(center_y + init_radius * np.sin(angle), miny, maxy)
            positions.append((float(x), float(y)))
    elif init_strategy == "center":
        for i in range(num_drones):
            positions.append((
                float(np.clip(center_x + i * 5.0, minx, maxx)),
                float(np.clip(center_y + i * 5.0, miny, maxy)),
            ))
    elif init_strategy == "random":
        for _ in range(num_drones):
            angle = rng.uniform(0, 2 * np.pi)
            r = init_radius * np.sqrt(rng.uniform())
            positions.append((
                float(np.clip(center_x + r * np.cos(angle), minx, maxx)),
                float(np.clip(center_y + r * np.sin(angle), miny, maxy)),
            ))
    else:
        for i in range(num_drones):
            angle = 2 * np.pi * i / num_drones
            x = np.clip(center_x + init_radius * np.cos(angle), minx, maxx)
            y = np.clip(center_y + init_radius * np.sin(angle), miny, maxy)
            positions.append((float(x), float(y)))
    return positions


def _generate_paths(
    mode: str, dataset_item, num_drones: int,
    fov_deg: float, altitude: float, budget: float,
    init_strategy: str, init_radius: float, seed: int,
) -> list[LineString]:
    minx, miny, maxx, maxy = dataset_item.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    max_radius = dataset_item.radius_km * 1000
    det_r = altitude * np.tan(np.radians(fov_deg / 2))
    spacing = det_r * 0.5
    overlap = DEFAULT_OVERLAP
    gen_mode = _resolve_gen_mode(mode)

    if gen_mode in _GEOMETRIC_BASELINES:
        if gen_mode == "spiral":
            return generate_spiral_path(
                center_x=cx, center_y=cy, max_radius=max_radius,
                fov_deg=fov_deg, altitude=altitude,
                overlap=overlap, num_drones=num_drones,
                path_point_spacing_m=spacing, budget=budget,
            )
        elif gen_mode == "concentric":
            return generate_concentric_circles_path(
                center_x=cx, center_y=cy, max_radius=max_radius,
                fov_deg=fov_deg, altitude=altitude,
                overlap=overlap, num_drones=num_drones,
                path_point_spacing_m=spacing, transition_distance_m=50.0,
                budget=budget,
            )
        else:  # pizza
            return generate_pizza_zigzag_path(
                center_x=cx, center_y=cy, max_radius=max_radius,
                num_drones=num_drones,
                fov_deg=fov_deg, altitude=altitude,
                overlap=overlap, path_point_spacing_m=spacing,
                border_gap_m=10.0, budget=budget,
            )

    m = re.match(r"^static_(\d+)step$", gen_mode)
    lookahead = int(m.group(1)) if m else 1
    initial_positions = _initial_positions(
        num_drones, cx, cy, dataset_item.bounds,
        init_strategy, init_radius, seed,
    )
    return generate_greedy_path(
        center_x=cx, center_y=cy,
        num_drones=num_drones,
        probability_map=dataset_item.heatmap,
        bounds=dataset_item.bounds,
        max_radius=max_radius,
        fov_deg=fov_deg, altitude=altitude,
        budget=budget,
        initial_positions=initial_positions,
        lookahead_steps=lookahead,
        seed=seed,
    )


def _drone_likelihood(
    path: LineString,
    heatmap: np.ndarray,
    bounds: tuple[float, float, float, float],
    detection_radius: float,
) -> float:
    if path.is_empty or path.length < 1.0:
        return 0.0
    minx, miny, maxx, maxy = bounds
    height, width = heatmap.shape
    dx = (maxx - minx) / width
    dy = (maxy - miny) / height
    sample_spacing = max(detection_radius * 0.5, 5.0)
    num_samples = max(int(path.length / sample_spacing) + 1, 1)
    det2 = detection_radius * detection_radius
    cells_x = int(np.ceil(detection_radius / dx)) + 1
    cells_y = int(np.ceil(detection_radius / dy)) + 1
    observed = np.zeros((height, width), dtype=bool)
    for s in np.linspace(0.0, path.length, num_samples):
        pt = path.interpolate(s)
        wx, wy = pt.x, pt.y
        col_c = int((wx - minx) / dx)
        row_c = int((wy - miny) / dy)
        r0 = max(0, row_c - cells_y)
        r1 = min(height, row_c + cells_y + 1)
        c0 = max(0, col_c - cells_x)
        c1 = min(width, col_c + cells_x + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        cell_x = minx + (cc + 0.5) * dx
        cell_y = miny + (rr + 0.5) * dy
        dist2 = (cell_x - wx) ** 2 + (cell_y - wy) ** 2
        observed[r0:r1, c0:c1] |= (dist2 <= det2)
    return float(np.sum(heatmap[observed]))


# ── Dataset cache ────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _load_dataset_cached(ds_id: int):
    ds_dir = f"sarenv_dataset/{ds_id}"
    return DatasetLoader(ds_dir).load_environment("xlarge")


# ══════════════════════════════════════════════════════════════════════════
#  STATIC modes — post‑hoc per‑drone path analysis
# ══════════════════════════════════════════════════════════════════════════

def _analyse_static_modes(
    dataset_ids: Sequence[int],
    static_modes: Sequence[str],
    num_drones: int,
    budget: float, fov_deg: float, altitude: float,
    init_strategy: str, init_radius: float,
    seeds: Sequence[int],
    failure_counts: Sequence[int],
    pbar: tqdm,
) -> list[dict]:
    """Post‑hoc: generate N‑drone paths, compute per‑drone lh, remove K."""
    detection_radius = altitude * np.tan(np.radians(fov_deg / 2))
    rows: list[dict] = []

    for ds_id in dataset_ids:
        dataset_item = _load_dataset_cached(ds_id)
        assert dataset_item is not None

        for mode in static_modes:
            gen_mode = _resolve_gen_mode(mode)
            eff_seeds = seeds if gen_mode not in _GEOMETRIC_BASELINES else seeds[:1]

            for seed in eff_seeds:
                paths = _generate_paths(
                    mode, dataset_item, num_drones,
                    fov_deg, altitude, budget,
                    init_strategy, init_radius, seed,
                )
                if len(paths) != num_drones:
                    pbar.update(1)
                    continue

                drone_lh = [
                    _drone_likelihood(p, dataset_item.heatmap,
                                      dataset_item.bounds, detection_radius)
                    for p in paths
                ]
                total_lh = sum(drone_lh)
                if total_lh <= 0:
                    pbar.update(1)
                    continue

                sorted_lh = sorted(drone_lh, reverse=True)
                avg_per_drone = total_lh / num_drones

                row: dict = {
                    "dataset": ds_id,
                    "planning_mode": mode,
                    "seed": seed,
                    "method": "posthoc",
                    "baseline_likelihood": total_lh,
                }
                for i, lh in enumerate(drone_lh):
                    row[f"drone_{i}_likelihood"] = lh

                for k in failure_counts:
                    if k >= num_drones:
                        continue
                    worst_k = sum(sorted_lh[:k])
                    row[f"fail{k}_worst_lh"] = total_lh - worst_k
                    row[f"fail{k}_worst_pct"] = 100.0 * worst_k / total_lh
                    row[f"fail{k}_avg_lh"] = total_lh - k * avg_per_drone
                    row[f"fail{k}_avg_pct"] = 100.0 * k * avg_per_drone / total_lh

                rows.append(row)
                pbar.update(1)

    return rows


# ══════════════════════════════════════════════════════════════════════════
#  DYNAMIC modes — full SARSimulation with N-K drones
# ══════════════════════════════════════════════════════════════════════════

def _run_one_simulation(
    ds_id: int, mode: str, num_drones_now: int,
    budget: float, fov_deg: float, altitude: float,
    seed: int, init_strategy: str, init_radius: float,
    dt: float = 1.0,
) -> dict | None:
    """Run one SARSimulation and return {likelihood, victims_found, ...}."""
    dataset_item = _load_dataset_cached(ds_id)
    if dataset_item is None:
        return None

    minx, miny, maxx, maxy = dataset_item.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    positions = _initial_positions(
        num_drones_now, cx, cy, dataset_item.bounds,
        init_strategy, init_radius, seed,
    )

    vgen = LostPersonLocationGenerator(dataset_item, seed=seed)
    victim_positions = vgen.generate_locations(
        n=5, percent_random_samples=0,
    )

    # Static paths only for static‑simulated modes (not used here, dynamic
    # modes don't need pre‑computed paths).
    sim = SARSimulation(
        dataset_item=dataset_item,
        num_drones=num_drones_now,
        num_victims=5,
        fov_deg=fov_deg,
        altitude=altitude,
        detection_probability=0.8,
        decay_tau=10_000.0,
        victim_speed=0.5,
        victim_model="random_walk",
        budget=budget,
        init_strategy=init_strategy,
        init_radius=init_radius,
        drone_speed=5.0,
        planning_mode=mode,
        static_paths=None,
        revisit_weight=0.5,
        seed=seed,
    )
    sim.setup()
    assert sim.dynamic_heatmap is not None
    sim.drone_positions = list(positions)
    sim.drone_paths = [[pos] for pos in positions]
    sim.victim_positions = list(victim_positions)
    sim.globally_observed_cells = set()
    for pos in sim.drone_positions:
        cells = sim._get_visible_cells_world(pos[0], pos[1])
        sim.globally_observed_cells.update(cells)
        sim.dynamic_heatmap.update(cells, current_time=0.0)

    sim._record_snapshot(store_heatmap=True)
    result = sim.run_from_state(
        dt=dt, snapshot_interval=100, heatmap_interval=200,
    )

    observed = list(sim.globally_observed_cells)
    likelihood = (
        sum(sim.dataset.heatmap[r, c] for r, c in observed)
        if observed else 0.0
    )
    return {
        "likelihood": likelihood,
        "victims_found": result.victims_found_count,
        "path_length": result.total_distance,
        "sim_time": result.total_time,
    }


def _analyse_dynamic_modes(
    dataset_ids: Sequence[int],
    dynamic_modes: Sequence[str],
    num_drones_full: int,
    budget: float, fov_deg: float, altitude: float,
    init_strategy: str, init_radius: float,
    seeds: Sequence[int],
    failure_counts: Sequence[int],
    pbar: tqdm,
) -> list[dict]:
    """Simulation‑based: run with N, N-1, N-2, N-3 drones."""
    rows: list[dict] = []

    drone_variants = [num_drones_full] + [
        num_drones_full - k for k in failure_counts
        if num_drones_full - k > 0
    ]  # [5, 4, 3, 2] for K=[1,2,3]

    for ds_id in dataset_ids:
        for mode in dynamic_modes:
            for seed in seeds:
                sim_results: dict[int, dict] = {}  # nd → result

                for nd in drone_variants:
                    scaled_budget = budget * nd / num_drones_full
                    res = _run_one_simulation(
                        ds_id, mode, nd, scaled_budget, fov_deg, altitude,
                        seed, init_strategy, init_radius,
                    )
                    if res is not None:
                        sim_results[nd] = res
                    pbar.update(1)

                if num_drones_full not in sim_results:
                    continue

                baseline_lh = sim_results[num_drones_full]["likelihood"]
                if baseline_lh <= 0:
                    continue

                row: dict = {
                    "dataset": ds_id,
                    "planning_mode": mode,
                    "seed": seed,
                    "method": "simulation",
                    "baseline_likelihood": baseline_lh,
                }

                for k in failure_counts:
                    nd = num_drones_full - k
                    if nd in sim_results and nd > 0:
                        fail_lh = sim_results[nd]["likelihood"]
                        row[f"fail{k}_worst_lh"] = fail_lh
                        row[f"fail{k}_worst_pct"] = (
                            100.0 * (baseline_lh - fail_lh) / baseline_lh
                        )
                        row[f"fail{k}_sim_lh"] = fail_lh
                    else:
                        row[f"fail{k}_worst_lh"] = float("nan")
                        row[f"fail{k}_worst_pct"] = float("nan")

                rows.append(row)

    return rows


# ══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ══════════════════════════════════════════════════════════════════════════

def run_resilience_analysis(
    dataset_ids: Sequence[int],
    modes: Sequence[str],
    num_drones: int,
    budget: float, fov_deg: float, altitude: float,
    seeds: Sequence[int],
    init_strategy: str, init_radius: float,
    failure_counts: Sequence[int],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    static_modes = [m for m in modes if m in _STATIC_MODES]
    dynamic_modes = [m for m in modes if m in _DYNAMIC_MODES]

    # Count total jobs
    n_static = 0
    for ds_id in dataset_ids:
        for mode in static_modes:
            gen_mode = _resolve_gen_mode(mode)
            eff = len(seeds) if gen_mode not in _GEOMETRIC_BASELINES else 1
            n_static += eff

    n_dynamic = 0
    drone_variants = 1 + len([k for k in failure_counts if num_drones - k > 0])
    for ds_id in dataset_ids:
        for mode in dynamic_modes:
            n_dynamic += len(seeds) * drone_variants

    total = n_static + n_dynamic
    pbar = tqdm(total=total, desc="Analyzing", unit="job")

    all_rows: list[dict] = []

    if static_modes:
        print(f"\n  Static (post‑hoc): {n_static} jobs — fast")
        rows_s = _analyse_static_modes(
            dataset_ids, static_modes, num_drones,
            budget, fov_deg, altitude,
            init_strategy, init_radius, seeds, failure_counts, pbar,
        )
        all_rows.extend(rows_s)

    if dynamic_modes:
        print(f"  Dynamic (simulation): {n_dynamic} jobs — ~5 s/job")
        rows_d = _analyse_dynamic_modes(
            dataset_ids, dynamic_modes, num_drones,
            budget, fov_deg, altitude,
            init_strategy, init_radius, seeds, failure_counts, pbar,
        )
        all_rows.extend(rows_d)

    pbar.close()

    if not all_rows:
        log.error("No results.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # ── Per‑mode summary ──
    summary = df.groupby("planning_mode").agg(
        baseline=("baseline_likelihood", "mean"),
        baseline_std=("baseline_likelihood", "std"),
        **{f"fail{k}_worst": (f"fail{k}_worst_lh", "mean")
           for k in failure_counts},
        **{f"fail{k}_worst_std": (f"fail{k}_worst_lh", "std")
           for k in failure_counts},
        **{f"fail{k}_worst_pct": (f"fail{k}_worst_pct", "mean")
           for k in failure_counts},
        **{f"fail{k}_worst_pct_std": (f"fail{k}_worst_pct", "std")
           for k in failure_counts},
        n=("dataset", "count"),
    ).reset_index()

    # ── Console report ──
    print("\n" + "=" * 100)
    print(f"RESILIENCE — Impact of losing K drones out of {num_drones}")
    print("=" * 100)

    for k in failure_counts:
        print(f"\n── Losing {k} drone(s) ──")
        print(f"  {'Mode':<18} {'Baseline':>10} {'-{k} worst':>12} {'Δ%':>8}")
        print("  " + "-" * 55)
        for _, r in summary.iterrows():
            print(f"  {r['planning_mode']:<18} {r['baseline']:>10.4f}  "
                  f"{r[f'fail{k}_worst']:>12.4f}  {r[f'fail{k}_worst_pct']:>7.1f}%")

    # ── Save ──
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_d = output_dir / f"resilience_detail_{ts}.csv"
    df.to_csv(csv_d, index=False)
    log.info(f"Detail → {csv_d}")

    csv_s = output_dir / f"resilience_summary_{ts}.csv"
    summary.to_csv(csv_s, index=False)
    log.info(f"Summary → {csv_s}")

    return df, summary


# ══════════════════════════════════════════════════════════════════════════
#  Plot
# ══════════════════════════════════════════════════════════════════════════

def _generate_kdrone_plot(summary: pd.DataFrame, failure_counts: list[int],
                          output_dir: Path) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("matplotlib not available — skipping plot")
        return Path()

    mode_order = ["pizza", "static", "dynamic", "static_3step", "dynamic_3step"]
    modes_present = [m for m in mode_order if m in summary["planning_mode"].values]
    sdf = summary.set_index("planning_mode").loc[modes_present].reset_index()

    n_modes = len(sdf)
    n_bars = 1 + len(failure_counts)
    x = np.arange(n_modes)
    width = 0.8 / n_bars

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"][:n_bars]
    labels_map = {1: "-1 drone", 2: "-2 drones", 3: "-3 drones"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                     gridspec_kw={"height_ratios": [2.2, 1]})
    offsets = np.linspace(-(n_bars - 1) * width / 2,
                          (n_bars - 1) * width / 2, n_bars)

    ebar_kw = dict(capsize=3, elinewidth=1, alpha=0.6)

    # ── Top: absolute likelihood ──
    ax1.bar(x + offsets[0], sdf["baseline"].values, width,
            yerr=sdf["baseline_std"].values, error_kw=ebar_kw,
            label="5 drones", color=colors[0],
            alpha=0.85, edgecolor="white", linewidth=0.5)

    for i, k in enumerate(failure_counts):
        col = f"fail{k}_worst"
        col_std = f"fail{k}_worst_std"
        ax1.bar(x + offsets[i + 1], sdf[col].values, width,
                yerr=sdf[col_std].values, error_kw=ebar_kw,
                label=labels_map.get(k, f"-{k}"),
                color=colors[i + 1],
                alpha=0.75, edgecolor="white", linewidth=0.5)

    for i, k in enumerate(failure_counts):
        pct_col = f"fail{k}_worst_pct"
        val_col = f"fail{k}_worst"
        for j in range(n_modes):
            val = sdf[val_col].values[j]
            pct = sdf[pct_col].values[j]
            if val > 0:
                ax1.text(x[j] + offsets[i + 1], val + 0.002,
                         f"-{pct:.0f}%", ha="center", va="bottom",
                         fontsize=7, fontweight="bold", color=colors[i + 1])

    ax1.set_ylabel("Likelihood (accumulated probability)")
    ax1.set_title("Drone-failure resilience — worst case",
                  fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sdf["planning_mode"])
    ax1.legend(loc="upper left", fontsize=8, ncol=n_bars)
    ax1.grid(axis="y", alpha=0.3)

    # ── Bottom: % loss ──
    for i, k in enumerate(failure_counts):
        pct_col = f"fail{k}_worst_pct"
        pct_std = f"fail{k}_worst_pct_std"
        ax2.bar(x + offsets[i + 1], sdf[pct_col].values, width,
                yerr=sdf[pct_std].values, error_kw=ebar_kw,
                label=labels_map.get(k, f"-{k}"),
                color=colors[i + 1],
                alpha=0.75, edgecolor="white", linewidth=0.5)
        for j in range(n_modes):
            pct_val = sdf[pct_col].values[j]
            if not np.isnan(pct_val):
                ax2.text(x[j] + offsets[i + 1], pct_val + 0.8,
                         f"{pct_val:.0f}%", ha="center", va="bottom",
                         fontsize=7, fontweight="bold", color=colors[i + 1])

    ax2.set_ylabel("% likelihood lost")
    ax2.set_xlabel("Planning mode")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sdf["planning_mode"])
    ax2.legend(loc="upper left", fontsize=8, ncol=len(failure_counts))
    ax2.grid(axis="y", alpha=0.3)
    all_pcts = [sdf[f"fail{k}_worst_pct"].max() for k in failure_counts]
    ax2.set_ylim(0, max(all_pcts) * 1.2)

    method_note = "Static: post‑hoc (exact)  |  Dynamic: simulation with N-K drones (re-planning on posterior)"
    fig.text(0.5, 0.01, method_note, ha="center", fontsize=8,
             style="italic", color="#555")

    plt.tight_layout(rect=(0, 0.035, 1, 1))
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"resilience_kdrone_{ts}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot → {plot_path}")
    return plot_path


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Drone‑failure resilience — K‑drone loss (Q1‑ready).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=int, nargs="+",
                   default=DEFAULT_DATASETS,
                   help="Dataset IDs")
    p.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                   help="Planning modes to compare")
    p.add_argument("--num-drones", type=int, default=5)
    p.add_argument("--budget", type=float, default=200_000)
    p.add_argument("--fov-deg", type=float, default=45.0)
    p.add_argument("--altitude", type=float, default=DEFAULT_ALTITUDE)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                   help="Seeds (geometric modes use first only)")
    p.add_argument("--failure-counts", type=int, nargs="+",
                   default=DEFAULT_FAILURE_COUNTS,
                   help="K drones lost")
    p.add_argument("--init-strategy", default=DEFAULT_INIT_STRATEGY,
                   choices=["circle", "center", "grid", "random"])
    p.add_argument("--init-radius", type=float, default=DEFAULT_INIT_RADIUS)
    p.add_argument("--output-dir", default="results/resilience")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-from", type=str, default=None,
                   help="Regenerate plot from existing detail CSV (skip analysis)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)

    failure_counts = [k for k in args.failure_counts if k < args.num_drones]

    # ── Plot-only mode: regenerate from existing CSV ──
    if args.plot_from:
        csv_path = Path(args.plot_from)
        if not csv_path.exists():
            log.error(f"CSV not found: {csv_path}")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        # Re-derive failure counts from columns if not explicitly provided
        if not failure_counts:
            avail = [int(c.replace("fail","").replace("_worst_lh",""))
                     for c in df.columns if c.startswith("fail") and c.endswith("_worst_lh")]
            failure_counts = sorted(avail)
        summary = df.groupby("planning_mode").agg(
            baseline=("baseline_likelihood", "mean"),
            baseline_std=("baseline_likelihood", "std"),
            **{f"fail{k}_worst": (f"fail{k}_worst_lh", "mean") for k in failure_counts},
            **{f"fail{k}_worst_std": (f"fail{k}_worst_lh", "std") for k in failure_counts},
            **{f"fail{k}_worst_pct": (f"fail{k}_worst_pct", "mean") for k in failure_counts},
            **{f"fail{k}_worst_pct_std": (f"fail{k}_worst_pct", "std") for k in failure_counts},
        ).reset_index()
        p = _generate_kdrone_plot(summary, failure_counts, out)
        if p.exists():
            print(f"Plot → {p}")
        return

    if not failure_counts:
        log.error(f"All failure_counts >= num_drones ({args.num_drones})")
        sys.exit(1)

    budget_per_drone = args.budget / args.num_drones
    print(f"Datasets: {args.dataset}")
    print(f"Modes:   {args.modes}")
    print(f"Drones:  {args.num_drones} → losing {failure_counts}")
    print(f"Budget:  {args.budget:,} total  |  {budget_per_drone:,.0f}/drone  "
          f"|  FOV: {args.fov_deg}°  |  Alt: {args.altitude}m  |  "
          f"Seeds: {args.seeds}")
    print(f"(In dynamic modes, budget scales with N-K drones: "
          f"{budget_per_drone:,.0f} m/drone constant)")
    print()

    df, summary = run_resilience_analysis(
        dataset_ids=args.dataset,
        modes=args.modes,
        num_drones=args.num_drones,
        budget=args.budget,
        fov_deg=args.fov_deg,
        altitude=args.altitude,
        seeds=args.seeds,
        init_strategy=args.init_strategy,
        init_radius=args.init_radius,
        failure_counts=failure_counts,
        output_dir=out,
    )

    if df.empty:
        log.error("No results.")
        sys.exit(1)

    if not args.no_plot:
        p = _generate_kdrone_plot(summary, failure_counts, out)
        if p.exists():
            print(f"\nPlot → {p}")

    print(f"\nCSVs in: {out}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
