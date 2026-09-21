#!/usr/bin/env python3
# scripts/nstep_sweep.py
"""
B1 — N-step horizon sweep for dynamic_Nstep planner.

Sweeps N ∈ {2, 3, 4, 5, 6, 7} on Dataset 1 (baseline, 50 trials each)
to generate the performance-vs-N curve. Idempotent: checkpoint skips
already-completed jobs on restart.

Output: results/nstep_sweep/nstep_sweep_<timestamp>.csv
        results/nstep_sweep/_checkpoint.json

Usage:
  pixi run python scripts/nstep_sweep.py -j 10          # run / resume
  pixi run python scripts/nstep_sweep.py -j 10 --reset   # start fresh
"""

import time as _time
from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
_comp = __import__("comprehensive_study")
StudyConfig = _comp.StudyConfig
run_single_job = _comp.run_single_job

# ── Configuration ────────────────────────────────────────────────────────
DATASET_ID = 1
NUM_TRIALS = 50
BASE_SEED = 42
N_VALUES = [2, 3, 4, 5, 6, 7]
MODE_TEMPLATE = "dynamic_{n}step"

DEFAULTS = dict(
    num_drones=5, budget=200_000.0, fov_deg=45.0, altitude=80.0,
    detection_prob=0.8, decay_tau=10_000.0, victim_speed=0.5,
    victim_model="random_walk", drone_speed=5.0, init_strategy="random",
    init_radius=100.0, revisit_weight=0.5, dt=1.0,
    size="xlarge", num_victims=1,
)

OUTDIR = Path("results/nstep_sweep")
CHECKPOINT = OUTDIR / "_checkpoint.json"


# ── Helpers ──────────────────────────────────────────────────────────────
def _job_key(cfg: StudyConfig) -> str:
    return f"{cfg.planning_mode}|seed={cfg.trial_seed}"


def build_jobs():
    jobs = []
    for n in N_VALUES:
        mode_name = MODE_TEMPLATE.format(n=n)
        for trial in range(NUM_TRIALS):
            cfg = StudyConfig(
                study_name=f"nstep_N{n}",
                dataset_id=DATASET_ID,
                trial_num=trial,
                trial_seed=BASE_SEED + trial,   # same seed scheme as 05_comprehensive_study (seed 42–91), paired with Study 5
                planning_mode=mode_name,
                **DEFAULTS,
            )
            jobs.append(cfg)
    return jobs


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


# ── Progress counter (file-backed, shared across workers) ────────────────
class Progress:
    """Thread/process-safe-ish progress counter via atomic file writes."""
    def __init__(self, path: Path, offset: int = 0):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(offset))

    def inc(self) -> int:
        try:
            cur = int(self.path.read_text().strip())
        except (ValueError, FileNotFoundError):
            cur = 0
        cur += 1
        self.path.write_text(str(cur))
        return cur


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser(description="B1 N-step horizon sweep")
    p.add_argument("-j", "--n-jobs", type=int, default=-1,
                   help="Parallel workers (default: all cores)")
    p.add_argument("--reset", action="store_true",
                   help="Delete checkpoint and start fresh")
    args = p.parse_args()

    if args.reset and CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint deleted — starting fresh.\n")

    all_jobs = build_jobs()
    completed = load_checkpoint()
    pending = [j for j in all_jobs if _job_key(j) not in completed]

    n_total = len(all_jobs)
    n_done = len(completed)
    n_pending = len(pending)

    print(f"B1 N-sweep: {len(N_VALUES)} modes × {NUM_TRIALS} trials = {n_total} jobs")
    print(f"  N values: {N_VALUES}  Dataset: {DATASET_ID}")
    print(f"  Workers: {args.n_jobs if args.n_jobs > 0 else 'all'}")
    if n_done > 0:
        print(f"  Resuming: {n_done} done, {n_pending} pending "
              f"({n_done/n_total*100:.0f}% complete)")
    else:
        print(f"  Fresh start: {n_pending} jobs")
    print()

    if n_pending == 0:
        print("All jobs already complete.")
        latest = sorted(OUTDIR.glob("nstep_sweep_*.csv"))
        if latest:
            df = pd.read_csv(latest[-1])
            print(f"Latest results: {latest[-1]}")
            print_summary(df)
        return

    # ── Run ──────────────────────────────────────────────────────────────
    progress = Progress(OUTDIR / "_progress.txt", offset=n_done)
    t_start = _time.time()

    def _run_one(cfg):
        key = _job_key(cfg)
        t1 = _time.time()
        result = run_single_job(cfg)
        elapsed = _time.time() - t1
        result["_job_key"] = key

        cur = progress.inc()
        pct = cur / n_total * 100
        elapsed_total = _time.time() - t_start
        speed = (cur - n_done) / max(elapsed_total, 0.1)
        eta = (n_total - cur) / max(speed, 0.01)
        print(f"  [{cur:3d}/{n_total} {pct:5.1f}%  |  "
              f"ETA {eta/60:5.1f}min]  "
              f"{cfg.planning_mode:<16} trial {cfg.trial_num:2d}  "
              f"LH={result.get('likelihood', float('nan')):.4f}  "
              f"wall={result.get('wall_time', float('nan')):.0f}s  "
              f"Δ{elapsed:.0f}s")
        return result

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
        delayed(_run_one)(cfg) for cfg in pending
    )

    # ── Save ─────────────────────────────────────────────────────────────
    # Mark all dispatched jobs as completed
    for r in results:
        completed.add(r["_job_key"])
    save_checkpoint(completed)

    df = pd.DataFrame(results)
    total_elapsed = _time.time() - t_start
    print(f"\nDone. {len(df)} trials in {total_elapsed/60:.1f} min "
          f"({total_elapsed/len(df):.1f} s/trial avg)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = OUTDIR / f"nstep_sweep_{ts}.csv"
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath}")

    # Cleanup
    pf = OUTDIR / "_progress.txt"
    if pf.exists():
        pf.unlink()

    print_summary(df)


def print_summary(df: pd.DataFrame):
    print("\nMean likelihood by N:")
    for col in ["planning_mode", "likelihood", "wall_time"]:
        if col not in df.columns:
            print(f"  (no '{col}' column)")
            return
    summary = df.groupby("planning_mode")["likelihood"].agg(["mean", "std", "count"])
    print(summary.to_string())
    print("\nMean wall_time (s) by N:")
    wt = df.groupby("planning_mode")["wall_time"].agg(["mean", "std"])
    print(wt.to_string())


if __name__ == "__main__":
    main()
