#!/usr/bin/env python3
# scripts/tree_search_baseline.py
"""
B2 — Exhaustive tree search baseline at N=3.

Runs tree_3step on Dataset 1 (30 trials) and compares against the
existing dynamic_3step results from Study 5 to measure:
  (a) speedup (wall-clock tree_3step / wall-clock dynamic_3step)
  (b) fidelity (Cliff's delta of likelihood — is Bellman as good as exact? → B6)

Idempotent via checkpoint.

Output: results/tree_search/tree_search_<timestamp>.csv

Usage:
  pixi run python scripts/tree_search_baseline.py -j 10
  pixi run python scripts/tree_search_baseline.py -j 10 --reset
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
NUM_TRIALS = 30
BASE_SEED = 42
MODE = "tree_3step"

DEFAULTS = dict(
    num_drones=5, budget=200_000.0, fov_deg=45.0, altitude=80.0,
    detection_prob=0.8, decay_tau=10_000.0, victim_speed=0.5,
    victim_model="random_walk", drone_speed=5.0, init_strategy="random",
    init_radius=100.0, revisit_weight=0.5, dt=1.0,
    size="xlarge", num_victims=5,
)

OUTDIR = Path("results/tree_search")
CHECKPOINT = OUTDIR / "_checkpoint.json"


# ── Helpers ──────────────────────────────────────────────────────────────
def _job_key(cfg: StudyConfig) -> str:
    return f"{cfg.planning_mode}|seed={cfg.trial_seed}"


class Progress:
    def __init__(self, path: Path, offset: int = 0):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(offset))

    def inc(self) -> int:
        try: cur = int(self.path.read_text().strip())
        except (ValueError, FileNotFoundError): cur = 0
        cur += 1
        self.path.write_text(str(cur))
        return cur


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


def build_jobs():
    jobs = []
    for trial in range(NUM_TRIALS):
        cfg = StudyConfig(
            study_name="tree_baseline",
            dataset_id=DATASET_ID,
            trial_num=trial,
            trial_seed=BASE_SEED + trial,
            planning_mode=MODE,
            **DEFAULTS,
        )
        jobs.append(cfg)
    return jobs


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    import argparse
    p = argparse.ArgumentParser(description="B2 tree search baseline")
    p.add_argument("-j", "--n-jobs", type=int, default=-1)
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()

    if args.reset and CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint deleted.\n")

    all_jobs = build_jobs()
    completed = load_checkpoint()
    pending = [j for j in all_jobs if _job_key(j) not in completed]

    n_total, n_done, n_pending = len(all_jobs), len(completed), len(pending)
    print(f"B2 Tree search: {MODE}, Dataset {DATASET_ID}, {NUM_TRIALS} trials")
    if n_done > 0:
        print(f"  Resuming: {n_done} done, {n_pending} pending")
    else:
        print(f"  Fresh start: {n_pending} jobs")
    print(f"  Estimated: ~1.5 h (10 cores, DS1, 30 trials)")
    print()

    if n_pending == 0:
        print("All complete.")
        latest = sorted(OUTDIR.glob("tree_search_*.csv"))
        if latest:
            df = pd.read_csv(latest[-1])
            print(f"Latest: {latest[-1]}")
            print_summary(df)
        return

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
        eta = (n_total - cur) / max((cur - n_done) / max(_time.time() - t_start, 0.1), 0.01)
        print(f"  [{cur:3d}/{n_total} {pct:5.1f}%  |  ETA {eta/60:5.1f}min]  "
              f"LH={result.get('likelihood', float('nan')):.4f}  "
              f"wall={result.get('wall_time', float('nan')):.0f}s  "
              f"Δ{elapsed:.0f}s")
        return result

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
        delayed(_run_one)(cfg) for cfg in pending
    )

    for r in results:
        completed.add(r["_job_key"])
    save_checkpoint(completed)

    df = pd.DataFrame(results)
    total_elapsed = _time.time() - t_start
    print(f"\nDone. {len(df)} trials in {total_elapsed/60:.1f} min")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = OUTDIR / f"tree_search_{ts}.csv"
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath}")

    pf = OUTDIR / "_progress.txt"
    if pf.exists(): pf.unlink()

    print_summary(df)


def print_summary(df: pd.DataFrame):
    print(f"\n  {MODE} summary ({len(df)} trials):")
    if "likelihood" in df.columns:
        lh = df["likelihood"]
        print(f"    Likelihood:  mean={lh.mean():.4f}  std={lh.std():.4f}")
    if "wall_time" in df.columns:
        wt = df["wall_time"]
        print(f"    Wall time:   mean={wt.mean():.0f}s  std={wt.std():.0f}s")


if __name__ == "__main__":
    main()
