#!/usr/bin/env python3
# scripts/05_comprehensive_study.py
"""
Comprehensive study comparing all search strategies across multiple
experimental configurations with optional parallelization via joblib.

Studies:
  1. Effect of number of drones
  2. Effect of budget
  3. Effect of victim model
  4. Effect of FOV angle
  5. All planning modes (main comparison)
  6. Generalization across datasets

Planning modes:
  Geometric baselines: spiral, concentric, pizza
  Greedy on prior:     static, static_2step, static_3step
  Greedy on posterior:  dynamic, dynamic_2step, dynamic_3step

Usage:
  # Default: parallel execution using all cores
  python scripts/05_comprehensive_study.py

  # Sequential execution
  python scripts/05_comprehensive_study.py --single

  # Parallel with 4 workers
  python scripts/05_comprehensive_study.py -j 4

  # Run only specific studies
  python scripts/05_comprehensive_study.py --studies 5 6

  # Verbose subprocess logging
  python scripts/05_comprehensive_study.py --debug

  # Custom settings
  python scripts/05_comprehensive_study.py --runs 50 --datasets 1 5 10 15 20 -j 8

Q1 Journal Publication (recommended):
  # Main comparison (Study 5)
  python scripts/05_comprehensive_study.py --studies 5 --runs 30 --budget 200000

  # Generalization (Study 6) - use 10-15+ datasets
  python scripts/05_comprehensive_study.py --studies 6 --runs 30 --budget 200000 \\
    --datasets 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25 30

  # Complete study (all 6 studies)
  python scripts/05_comprehensive_study.py --runs 30 --budget 200000 \\
    --datasets 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25 30
"""
import argparse
import json
import logging
import os
import re
import time as _time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point
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

ALL_MODES = [
    "spiral", "concentric", "pizza",
    "static", "static_2step", "static_3step",
    "dynamic", "dynamic_2step", "dynamic_3step",
]

_GEOMETRIC_BASELINES = {"spiral", "concentric", "pizza"}


@dataclass
class StudyConfig:
    """Configuration for a single simulation run (mode × trial)."""
    study_name: str
    dataset_id: int
    size: str
    num_drones: int
    num_victims: int
    budget: float
    fov_deg: float
    altitude: float
    detection_prob: float
    decay_tau: float
    victim_speed: float
    victim_model: str
    drone_speed: float
    init_strategy: str
    init_radius: float
    revisit_weight: float
    planning_mode: str
    dt: float
    trial_seed: int
    trial_num: int
    implementation_version: str = "working_tree"
    fix_profile: str = "original"

# ── Core simulation logic (importable, pickle-safe) ─────────────────────

def _generate_initial_positions(num_drones, center_x, center_y, bounds, init_strategy, init_radius, seed):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bounds
    positions = []
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
    elif init_strategy == "grid":
        side = int(np.ceil(np.sqrt(num_drones)))
        spacing = 2 * init_radius / max(side - 1, 1)
        idx = 0
        for r in range(side):
            for c in range(side):
                if idx >= num_drones:
                    break
                x = center_x - init_radius + c * spacing
                y = center_y - init_radius + r * spacing
                positions.append((
                    float(np.clip(x, minx, maxx)),
                    float(np.clip(y, miny, maxy)),
                ))
                idx += 1
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


def _generate_baseline_paths(mode, dataset_item, cfg):
    minx, miny, maxx, maxy = dataset_item.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    max_radius = dataset_item.radius_km * 1000
    det_r = cfg.altitude * np.tan(np.radians(cfg.fov_deg / 2))
    spacing = det_r * 0.5
    overlap = 0.1

    if mode == "spiral":
        return generate_spiral_path(
            center_x=cx, center_y=cy, max_radius=max_radius,
            fov_deg=cfg.fov_deg, altitude=cfg.altitude,
            overlap=overlap, num_drones=cfg.num_drones,
            path_point_spacing_m=spacing, budget=cfg.budget,
        )
    elif mode == "concentric":
        return generate_concentric_circles_path(
            center_x=cx, center_y=cy, max_radius=max_radius,
            fov_deg=cfg.fov_deg, altitude=cfg.altitude,
            overlap=overlap, num_drones=cfg.num_drones,
            path_point_spacing_m=spacing, transition_distance_m=50.0,
            budget=cfg.budget,
        )
    elif mode == "pizza":
        return generate_pizza_zigzag_path(
            center_x=cx, center_y=cy, max_radius=max_radius,
            num_drones=cfg.num_drones,
            fov_deg=cfg.fov_deg, altitude=cfg.altitude,
            overlap=overlap, path_point_spacing_m=spacing,
            border_gap_m=10.0, budget=cfg.budget,
        )
    raise ValueError(f"Unknown geometric baseline: {mode}")


def run_single_job(cfg: StudyConfig) -> dict:
    """Run ONE (mode, trial) combination. Designed for joblib.Parallel.

    Loads the dataset internally so each worker is self-contained.
    Returns a flat dict with all result columns.
    """
    dataset_dir = f"sarenv_dataset/{cfg.dataset_id}"
    loader = DatasetLoader(dataset_dir)
    dataset_item = loader.load_environment(cfg.size)

    minx, miny, maxx, maxy = dataset_item.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    initial_positions = _generate_initial_positions(
        cfg.num_drones, cx, cy, dataset_item.bounds,
        cfg.init_strategy, cfg.init_radius, cfg.trial_seed,
    )

    # Path generation
    static_paths = None
    sim_mode = cfg.planning_mode

    if cfg.planning_mode in _GEOMETRIC_BASELINES:
        static_paths = _generate_baseline_paths(cfg.planning_mode, dataset_item, cfg)
        sim_mode = "static"
    elif cfg.planning_mode.startswith("static"):
        m = re.match(r"^static_(\d+)step$", cfg.planning_mode)
        lookahead = int(m.group(1)) if m else 1
        static_paths = generate_greedy_path(
            center_x=cx, center_y=cy,
            num_drones=cfg.num_drones,
            probability_map=dataset_item.heatmap,
            bounds=dataset_item.bounds,
            max_radius=dataset_item.radius_km * 1000,
            fov_deg=cfg.fov_deg, altitude=cfg.altitude,
            budget=cfg.budget,
            initial_positions=initial_positions,
            lookahead_steps=lookahead,
            seed=cfg.trial_seed,
        )
        sim_mode = cfg.planning_mode

    sim = SARSimulation(
        dataset_item=dataset_item,
        num_drones=cfg.num_drones,
        num_victims=cfg.num_victims,
        fov_deg=cfg.fov_deg,
        altitude=cfg.altitude,
        detection_probability=cfg.detection_prob,
        decay_tau=cfg.decay_tau,
        victim_speed=cfg.victim_speed,
        victim_model=cfg.victim_model,
        budget=cfg.budget,
        init_strategy=cfg.init_strategy,
        init_radius=cfg.init_radius,
        drone_speed=cfg.drone_speed,
        planning_mode=sim_mode,
        static_paths=static_paths,
        revisit_weight=cfg.revisit_weight,
        seed=cfg.trial_seed,
        dataset_id=cfg.dataset_id,
        implementation_version=cfg.implementation_version,
        fix_profile=cfg.fix_profile,
    )

    t0 = _time.time()
    sim.setup(initial_positions=initial_positions)
    result = sim.run_from_state(dt=cfg.dt, snapshot_interval=50, heatmap_interval=100)
    wall_time = _time.time() - t0

    observed = list(sim.globally_observed_cells)
    prior_likelihood = (
        sum(sim.dataset.heatmap[r, c] for r, c in observed)
        if observed else 0.0
    )

    return {
        "study": cfg.study_name,
        "dataset": cfg.dataset_id,
        "planner": cfg.planning_mode,
        "planning_mode": cfg.planning_mode,
        "trial": cfg.trial_num,
        "seed": cfg.trial_seed,
        "num_drones": cfg.num_drones,
        "num_victims": cfg.num_victims,
        "budget": cfg.budget,
        "pd": cfg.detection_prob,
        "tau": cfg.decay_tau,
        "w": cfg.revisit_weight,
        "fov_deg": cfg.fov_deg,
        "victim_model": cfg.victim_model,
        "implementation_version": cfg.implementation_version,
        "fix_profile": cfg.fix_profile,
        "likelihood": prior_likelihood,
        "L": prior_likelihood,
        "victims_found": result.victims_found_count,
        "path_length": result.total_distance,
        "cells_observed": result.snapshots[-1].cells_observed,
        "sim_time": result.total_time,
        "total_time_sim": result.total_time,
        "wall_time": wall_time,
        "planner_wall_time": result.planner_wall_time,
        "shadow_wall_time": result.shadow_wall_time,
        "number_of_replans": result.number_of_replans,
        "number_of_target_conflicts": result.number_of_target_conflicts,
        "reservation_blocked": result.reservation_blocked,
        "first_action_sequence_hash": result.first_action_sequence_hash,
        "planning_trace_json": json.dumps(result.planning_trace, separators=(",", ":")),
        "target_trace_json": json.dumps(result.target_trace, separators=(",", ":")),
    }

# ── Study definitions ────────────────────────────────────────────────────

def _base_cfg(args, study_name, mode, trial_num):
    return StudyConfig(
        study_name=study_name,
        dataset_id=args.dataset,
        size=args.size,
        num_drones=args.num_drones,
        num_victims=args.num_victims,
        budget=args.budget,
        fov_deg=args.fov_deg,
        altitude=args.altitude,
        detection_prob=args.detection_prob,
        decay_tau=args.decay_tau,
        victim_speed=args.victim_speed,
        victim_model=args.victim_model,
        drone_speed=args.drone_speed,
        init_strategy=args.init_strategy,
        init_radius=args.init_radius,
        revisit_weight=args.revisit_weight,
        planning_mode=mode,
        trial_seed=args.base_seed + trial_num,
        trial_num=trial_num + 1,
        dt=args.dt,
    )


def build_study_1(args) -> list[StudyConfig]:
    """Effect of number of drones with PROPORTIONAL budget."""
    jobs = []
    BUDGET_PER_DRONE = 40_000  # Constant budget per drone for fair scaling comparison
    for nd in [3, 5, 10, 15]:
        for mode in ALL_MODES:
            for t in range(args.runs):
                cfg = _base_cfg(args, "drones", mode, t)
                cfg.num_drones = nd
                cfg.budget = BUDGET_PER_DRONE * nd  # Scale budget proportionally
                jobs.append(cfg)
    return jobs


def build_study_2(args) -> list[StudyConfig]:
    """Effect of budget."""
    jobs = []
    for b in [100_000, 200_000, 300_000, 500_000]:
        for mode in ALL_MODES:
            for t in range(args.runs):
                cfg = _base_cfg(args, "budget", mode, t)
                cfg.budget = b
                jobs.append(cfg)
    return jobs


def build_study_3(args) -> list[StudyConfig]:
    """Effect of victim model."""
    jobs = []
    for vm in ["random_walk", "route_following", "lost_person"]:
        for mode in ALL_MODES:
            for t in range(args.runs):
                cfg = _base_cfg(args, "victim_model", mode, t)
                cfg.victim_model = vm
                # budget inherited from args via _base_cfg
                jobs.append(cfg)
    return jobs


def build_study_4(args) -> list[StudyConfig]:
    """Effect of FOV angle."""
    jobs = []
    for fov in [30, 45, 60, 90]:
        for mode in ALL_MODES:
            for t in range(args.runs):
                cfg = _base_cfg(args, "fov", mode, t)
                cfg.fov_deg = float(fov)
                # budget inherited from args via _base_cfg
                jobs.append(cfg)
    return jobs


def build_study_5(args) -> list[StudyConfig]:
    """All planning modes — main comparison."""
    jobs = []
    for mode in ALL_MODES:
        for t in range(args.runs):
            cfg = _base_cfg(args, "main", mode, t)
            # budget inherited from args via _base_cfg
            jobs.append(cfg)
    return jobs


def build_study_6(args) -> list[StudyConfig]:
    """Generalization across datasets."""
    jobs = []
    for ds in args.datasets:
        for mode in ALL_MODES:
            for t in range(args.runs):
                cfg = _base_cfg(args, "datasets", mode, t)
                cfg.dataset_id = ds
                # budget inherited from args via _base_cfg
                jobs.append(cfg)
    return jobs


STUDY_BUILDERS = {
    1: ("Effect of Number of Drones", build_study_1),
    2: ("Effect of Budget", build_study_2),
    3: ("Effect of Victim Model", build_study_3),
    4: ("Effect of FOV Angle", build_study_4),
    5: ("All Planning Modes (main)", build_study_5),
    6: ("Generalization Across Datasets", build_study_6),
}


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Comprehensive SAR comparison study with optional parallelization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=int, default=1, help="Default dataset ID")
    p.add_argument("--size", type=str, default="xlarge")
    p.add_argument("--num-drones", type=int, default=5)
    p.add_argument("--num-victims", type=int, default=5)
    p.add_argument("--budget", type=float, default=200_000,
                    help="Default budget for studies 1,3,4,5,6 (Study 2 tests multiple budgets)")
    p.add_argument("--fov-deg", type=float, default=45.0)
    p.add_argument("--altitude", type=float, default=80.0)
    p.add_argument("--detection-prob", type=float, default=0.8)
    p.add_argument("--decay-tau", type=float, default=10_000.0)
    p.add_argument("--victim-speed", type=float, default=0.5)
    p.add_argument("--victim-model", type=str, default="random_walk",
                    choices=["random_walk", "route_following", "lost_person"])
    p.add_argument("--drone-speed", type=float, default=5.0)
    p.add_argument("--init-strategy", type=str, default="random",
                    choices=["circle", "center", "grid", "random"])
    p.add_argument("--init-radius", type=float, default=100.0)
    p.add_argument("--revisit-weight", type=float, default=0.5)
    p.add_argument("--dt", type=float, default=1.0,
                    help="Simulation timestep in seconds (default: 1.0, use 3.0-5.0 for faster exploration)")
    p.add_argument("--runs", type=int, default=30, help="Trials per configuration")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--datasets", type=int, nargs="+", default=[1, 5, 10, 15, 20, 25, 30],
                    help="Dataset IDs for Study 6")
    p.add_argument("--studies", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                    choices=[1, 2, 3, 4, 5, 6], help="Which studies to run")
    p.add_argument("--single", action="store_true",
                    help="Run sequentially (default is parallel)")
    p.add_argument("-j", "--n-jobs", type=int, default=-1,
                    help="Number of parallel workers (-1 = all cores)")
    p.add_argument("--backend", type=str, default="loky",
                    choices=["loky", "threading", "multiprocessing"],
                    help="Joblib parallel backend")
    p.add_argument("--debug", action="store_true",
                    help="Show verbose subprocess logging (default: quiet)")
    p.add_argument("--output-dir", type=str, default="results/comprehensive_study",
                    help="Directory for output CSVs")
    p.add_argument("--no-resume", action="store_true",
                    help="Do not resume from checkpoint, start fresh")
    p.add_argument("-y", "--yes", action="store_true", default=True,
                    help="Auto-confirm checkpoint resume without prompting (default: True)")
    p.add_argument("--no-confirm", action="store_false", dest="yes",
                    help="Prompt before resuming from checkpoint")
    p.add_argument("--checkpoint-interval", type=int, default=10,
                    help="Save checkpoint every N completed jobs (default: 10)")
    return p.parse_args()


# ── Checkpoint helpers ───────────────────────────────────────────────────

def _get_checkpoint_path(output_dir: Path, studies: list[int]) -> Path:
    """Get checkpoint file path based on studies being run."""
    study_suffix = "_".join(map(str, sorted(studies)))
    return output_dir / f"_checkpoint_studies_{study_suffix}.csv"


def _job_key(cfg: StudyConfig) -> str:
    """Create unique key for a job to identify if already completed."""
    return f"{cfg.study_name}|{cfg.dataset_id}|{cfg.planning_mode}|{cfg.trial_seed}"


def _result_key(row: dict) -> str:
    """Create unique key from a result row."""
    return f"{row['study']}|{row['dataset']}|{row['planning_mode']}|{row['seed']}"


def _load_checkpoint(checkpoint_path: Path) -> tuple[list[dict], set[str]]:
    """Load checkpoint file if exists. Returns (results list, set of completed job keys)."""
    if not checkpoint_path.exists():
        return [], set()
    
    df = pd.read_csv(checkpoint_path)
    results = df.to_dict('records')
    completed_keys = {_result_key(row) for row in results}
    return results, completed_keys


def _save_checkpoint(results: list[dict], checkpoint_path: Path):
    """Save all results to checkpoint file."""
    if results:
        df = pd.DataFrame(results)
        df.to_csv(checkpoint_path, index=False)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set log level via env var so worker processes inherit it too
    os.environ["SARENV_LOG_LEVEL"] = "DEBUG" if args.debug else "WARNING"

    use_parallel = not args.single
    checkpoint_path = _get_checkpoint_path(output_dir, args.studies)
    
    # Check for existing checkpoint
    completed_results, completed_keys = [], set()
    if checkpoint_path.exists() and not args.no_resume:
        completed_results, completed_keys = _load_checkpoint(checkpoint_path)
        n_completed = len(completed_keys)
        print(f"\n⚠️  Found checkpoint with {n_completed} completed jobs.")
        if args.yes:
            print(f"Auto-resuming from checkpoint (--yes active): will skip {n_completed} already completed jobs.\n")
            response = 'y'
        else:
            try:
                response = input("Resume from checkpoint? [Y/n]: ").strip().lower()
            except EOFError:
                response = 'y'  # Default to yes in non-interactive mode
        if response in ('n', 'no'):
            print("Starting fresh (checkpoint will be overwritten).")
            completed_results, completed_keys = [], set()
        else:
            if not args.yes:
                print(f"Resuming: will skip {n_completed} already completed jobs.\n")

    # Build all jobs
    all_jobs: list[StudyConfig] = []
    for study_num in args.studies:
        name, builder = STUDY_BUILDERS[study_num]
        jobs = builder(args)
        all_jobs.extend(jobs)
        print(f"  Study {study_num} ({name}): {len(jobs)} jobs")

    # Filter out completed jobs
    if completed_keys:
        pending_jobs = [j for j in all_jobs if _job_key(j) not in completed_keys]
        print(f"\n  Already completed: {len(all_jobs) - len(pending_jobs)}")
        print(f"  Remaining jobs: {len(pending_jobs)}")
    else:
        pending_jobs = all_jobs

    if not pending_jobs:
        print("\n✓ All jobs already completed!")
        results = completed_results
    else:
        total = len(pending_jobs)
        mode_str = f"parallel (n_jobs={args.n_jobs})" if use_parallel else "sequential"
        print(f"\n  Total pending: {total}")
        print(f"  Mode: {mode_str}")
        if use_parallel:
            import multiprocessing
            n_cores = multiprocessing.cpu_count()
            effective_jobs = n_cores if args.n_jobs == -1 else args.n_jobs
            print(f"  Workers: {effective_jobs} (of {n_cores} cores)")
        print()

        t0 = _time.time()
        new_results = []
        checkpoint_interval = args.checkpoint_interval

        if use_parallel:
            from joblib import Parallel, delayed
            import multiprocessing
            n_cores = multiprocessing.cpu_count()
            effective_jobs = n_cores if args.n_jobs == -1 else args.n_jobs
            
            # Process in batches for checkpointing with total progress bar
            batch_size = max(checkpoint_interval, effective_jobs * 2)
            total_jobs = len(all_jobs)
            already_completed = len(completed_results)
            
            with tqdm(total=len(pending_jobs), desc="Progress", unit="job", initial=0) as pbar:
                for batch_start in range(0, len(pending_jobs), batch_size):
                    batch = pending_jobs[batch_start:batch_start + batch_size]
                    batch_results = Parallel(
                        n_jobs=args.n_jobs,
                        backend=args.backend,
                        verbose=0,
                    )(delayed(run_single_job)(cfg) for cfg in batch)
                    new_results.extend(batch_results)
                    pbar.update(len(batch_results))
                    
                    # Save checkpoint after each batch
                    _save_checkpoint(completed_results + new_results, checkpoint_path)
                    completed_so_far = already_completed + len(new_results)
                    print(f"  💾 Checkpoint: {completed_so_far}/{total_jobs} jobs ({100*completed_so_far/total_jobs:.0f}%)")
        else:
            for i, cfg in enumerate(tqdm(pending_jobs, desc="Running", unit="job")):
                result = run_single_job(cfg)
                new_results.append(result)
                
                # Save checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    _save_checkpoint(completed_results + new_results, checkpoint_path)
                    completed_so_far = len(completed_results) + len(new_results)
                    total_jobs = len(all_jobs)
                    print(f"  💾 Checkpoint: {completed_so_far}/{total_jobs} jobs ({100*completed_so_far/total_jobs:.0f}%)")
            
            # Final checkpoint save
            _save_checkpoint(completed_results + new_results, checkpoint_path)

        elapsed = _time.time() - t0
        print(f"\nNew jobs completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        results = completed_results + new_results

    # Aggregate into DataFrame and save per study
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for study_name, group in df.groupby("study"):
        csv_path = output_dir / f"study_{study_name}_{timestamp}.csv"
        group.to_csv(csv_path, index=False)
        print(f"Saved {len(group)} rows → {csv_path}")

    # Also save the complete results
    all_path = output_dir / f"all_studies_{timestamp}.csv"
    df.to_csv(all_path, index=False)
    print(f"Complete results → {all_path}")

    # Remove checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("✓ Checkpoint removed (study complete)")

    # Print summary per study
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for study_name, group in df.groupby("study"):
        print(f"\n--- {study_name} ---")
        summary = group.groupby("planning_mode")["likelihood"].agg(["mean", "std", "count"])
        summary = summary.sort_values("mean", ascending=False)
        ref_mean = summary["mean"].iloc[-1]  # worst as reference
        for mode, row in summary.iterrows():
            delta = 100 * (row["mean"] - ref_mean) / ref_mean if ref_mean > 0 else 0
            print(f"  {mode:<18s}  L={row['mean']:.4f} ± {row['std']:.4f}  "
                  f"(n={int(row['count'])})  Δ={delta:+.1f}%")

    print(f"\nOutput directory: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
