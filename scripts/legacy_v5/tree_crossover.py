#!/usr/bin/env python3
# scripts/tree_crossover.py
"""
B9 — Empirical Bellman/tree wall-time crossover on DS1.

Measures both planners (tree_Nstep, dynamic_Nstep) paired by seed for
N in {3,4,5,6}. N=3 anchors against Table VI. Single-threaded by default.

Idempotent via checkpoint: safe to interrupt and resume.

Output: results/tree_crossover/tree_crossover.csv (appended incrementally)

Usage:
  pixi run python scripts/tree_crossover.py
  pixi run python scripts/tree_crossover.py --no-n6
  pixi run python scripts/tree_crossover.py --reset
  pixi run python scripts/tree_crossover.py --threads 4  # NON-faithful timing
"""

import os
import sys as _sys

_threads = "1"
if "--threads" in _sys.argv:
    _threads = _sys.argv[_sys.argv.index("--threads") + 1]
for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = _threads

import time as _time
import json
from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
_comp = __import__("comprehensive_study")
StudyConfig = _comp.StudyConfig
run_single_job = _comp.run_single_job

# ── Configuration ────────────────────────────────────────────────────────
DATASET_ID = 1
NUM_TRIALS = 5
BASE_SEED = 42
N_VALUES = [3, 4, 5, 6]
PLANNERS = ["dynamic", "tree"]

DEFAULTS = dict(
    num_drones=5,
    budget=200_000.0,
    fov_deg=45.0,
    altitude=80.0,
    detection_prob=0.8,
    decay_tau=10_000.0,
    victim_speed=0.5,
    victim_model="random_walk",
    drone_speed=5.0,
    init_strategy="random",
    init_radius=100.0,
    revisit_weight=0.5,
    dt=1.0,
    size="xlarge",
    num_victims=5,
)

OUTDIR = Path("results/tree_crossover")
CHECKPOINT = OUTDIR / "_checkpoint.json"
CSV_PATH = OUTDIR / "tree_crossover.csv"


# ── Checkpoint helpers ───────────────────────────────────────────────────
def _job_key(planner: str, N: int, trial: int) -> str:
    return f"{planner}_{N}step|seed={BASE_SEED + trial}"


def load_checkpoint() -> set:
    if not CHECKPOINT.exists():
        return set()
    try:
        return set(json.loads(CHECKPOINT.read_text()).get("completed", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def save_checkpoint(completed: set):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps({
        "completed": sorted(completed),
        "updated": datetime.now().isoformat(),
    }, indent=2))


def append_result(result: dict):
    """Append a single result row to the CSV (create if missing)."""
    df_row = pd.DataFrame([result])
    if CSV_PATH.exists():
        df_row.to_csv(CSV_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(CSV_PATH, index=False)


# ── Core ─────────────────────────────────────────────────────────────────
def run_one(planner: str, N: int, trial: int) -> dict:
    mode = f"{planner}_{N}step"
    cfg = StudyConfig(
        study_name="tree_crossover",
        dataset_id=DATASET_ID,
        trial_num=trial,
        trial_seed=BASE_SEED + trial,
        planning_mode=mode,
        **DEFAULTS,
    )
    result = run_single_job(cfg)
    result["planner"] = planner
    result["N"] = N
    result["n_paths"] = 8**N if planner == "tree" else N - 1
    return result


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="B9 tree/Bellman crossover (paired, single-thread, idempotent)"
    )
    p.add_argument("--no-n6", action="store_true", help="skip the slow N=6 point")
    p.add_argument("--threads", default="1", help="thread cap (default 1)")
    p.add_argument("--trials", type=int, default=NUM_TRIALS)
    p.add_argument("--reset", action="store_true", help="delete checkpoint and start fresh")
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.reset:
        if CHECKPOINT.exists():
            CHECKPOINT.unlink()
        if CSV_PATH.exists():
            CSV_PATH.unlink()
        print("Checkpoint and CSV deleted.\n")

    if args.threads != "1":
        print(
            f"  WARNING: running with {args.threads} threads — wall-times are NOT "
            f"comparable to Table VI, and the crossover may shift."
        )

    n_values = [n for n in N_VALUES if not (args.no_n6 and n == 6)]
    completed = load_checkpoint()

    # Build job list
    jobs = []
    for N in n_values:
        for trial in range(args.trials):
            for planner in PLANNERS:
                key = _job_key(planner, N, trial)
                if key not in completed:
                    jobs.append((planner, N, trial, key))

    n_total = len(n_values) * args.trials * len(PLANNERS)
    n_done = n_total - len(jobs)

    print(f"B9 Tree/Bellman crossover — DS{DATASET_ID}, {args.trials} paired trials")
    print(f"  N values: {n_values}")
    print(f"  Threads: {args.threads}")
    if n_done > 0:
        print(f"  Resuming: {n_done}/{n_total} done, {len(jobs)} pending")
    else:
        print(f"  Fresh start: {len(jobs)} jobs")
    print()

    if not jobs:
        print("All complete.")
    else:
        t_start = _time.time()
        for i, (planner, N, trial, key) in enumerate(jobs):
            tag = f"{planner}_{N}step"
            r = run_one(planner, N, trial)

            append_result(r)
            completed.add(key)
            save_checkpoint(completed)

            elapsed_total = _time.time() - t_start
            rate = (i + 1) / elapsed_total if elapsed_total > 0 else 0
            remaining = (len(jobs) - i - 1) / rate if rate > 0 else 0

            print(
                f"  [{n_done + i + 1}/{n_total}] {tag:<16} seed={BASE_SEED + trial}  "
                f"L={r.get('likelihood', float('nan')):.4f}  "
                f"wall={r.get('wall_time', float('nan')):.0f}s  "
                f"ETA={remaining / 60:.0f}min"
            )

        print(f"\nAll done in {(_time.time() - t_start) / 60:.1f} min")

    # ── Summary ──────────────────────────────────────────────────────────
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print(f"\n{'=' * 64}")
        print(f"Results: {CSV_PATH}")
        print(f"{'=' * 64}")
        print(
            f"\n  {'N':>3}  {'paths':>10}  {'dyn_wall':>9}  {'tree_wall':>9}  "
            f"{'ratio t/d':>9}  {'faster':>8}  {'dyn_L':>7}  {'tree_L':>7}"
        )
        crossover = None
        for N in n_values:
            d = df[(df.N == N) & (df.planner == "dynamic")]
            t = df[(df.N == N) & (df.planner == "tree")]
            if d.empty or t.empty:
                continue
            dw, tw = d.wall_time.mean(), t.wall_time.mean()
            ratio = tw / dw if dw else float("nan")
            faster = "dynamic" if tw > dw else "tree"
            if faster == "dynamic" and crossover is None:
                crossover = N
            print(
                f"  {N:>3}  {8**N:>10,}  {dw:>8.0f}s  {tw:>8.0f}s  {ratio:>9.2f}  "
                f"{faster:>8}  {d.likelihood.mean():>7.4f}  {t.likelihood.mean():>7.4f}"
            )

        print(
            f"\n  ANCHOR (N=3): expect tree~72s / dynamic~132s (Table VI)."
        )
        if crossover:
            print(
                f"  CROSSOVER at N={crossover} (tree becomes slower than Bellman)."
            )
        else:
            print("  No crossover within tested range.")


if __name__ == "__main__":
    main()
