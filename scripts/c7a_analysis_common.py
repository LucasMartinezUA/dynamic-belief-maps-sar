"""Shared loaders/helpers for the C7A immediate post-run analyses.

Frozen artifacts only: the C7A campaign runs CSV, the recovery curves and the
trajectory archive.  No new simulations, no policy/metric changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT / "scripts"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

OUTPUT_DIR = ROOT / "results" / "audit_block_C"
FIG_DIR = OUTPUT_DIR / "C7_FIGURES"

RUNS_NAME = "c7_midflight_runs.csv"
RECOVERY_NAME = "c7_midflight_recovery.csv"
STAGE = "c7-midflight"

NUMERIC_COLS = [
    "P_detect", "RMST", "RMST_at_own_T", "RMST_at_fault", "L", "Pdet_at_fault",
    "PostFault_PdetGain", "ConditionalPostFaultPdet", "exposure_multiplicity",
    "revisit_fraction", "T", "H", "P_detect_at_Tref", "RMST_at_Tref",
    "NRMST_at_Tref", "NRMST", "actual_distance_total", "budget_utilization",
    "failure_distance", "failure_time", "trigger_step", "retention_Pdet",
    "degradation_NRMST", "T_common", "H_common", "budget", "num_drones",
    "number_of_replans", "planner_wall_time", "failed_uav_post_fault_distance",
    "search_inactive_post_fault_exposures", "failed_uav_post_fault_exposures",
]

# Frozen C7 constants (c7_stage.py) — do not change.
T_REF = 8000.0
B_FAIL = 100_000.0
BUDGET = 200_000.0
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_ITERS = 10_000


def find_campaign(runs_name: str = RUNS_NAME) -> Path:
    candidates = []
    for candidate in sorted(OUTPUT_DIR.iterdir()):
        config_path = candidate / "config.json"
        if not config_path.exists() or not (candidate / runs_name).exists():
            continue
        try:
            existing = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if existing.get("stage") == STAGE:
            candidates.append((existing.get("timestamp", ""), candidate))
    if not candidates:
        raise SystemExit(f"no completed {STAGE} campaign with {runs_name}")
    candidates.sort()
    return candidates[-1][1]


def load_runs() -> pd.DataFrame:
    frame = pd.read_csv(find_campaign() / RUNS_NAME, low_memory=False)
    for column in NUMERIC_COLS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_recovery() -> pd.DataFrame:
    frame = pd.read_csv(find_campaign() / RECOVERY_NAME, low_memory=False)
    for column in ("step", "time", "distance_absolute", "distance_post_fault",
                   "b_post", "P_detect_cumulative", "L_cumulative"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def trial_metric(frame: pd.DataFrame, mode: str, column: str,
                 condition: str = "midflight_failure") -> pd.Series:
    """Per-trial series indexed by (dataset, planning_seed)."""
    sub = frame[(frame.planning_mode == mode) & (frame.condition == condition)]
    series = sub.set_index(["dataset", "planning_seed"])[column]
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series


def paired_vector(frame: pd.DataFrame, method: str, control: str, column: str,
                  condition: str = "midflight_failure") -> pd.Series:
    """Per-trial (method - control) differences, NaN pairs dropped."""
    left = trial_metric(frame, method, column, condition)
    right = trial_metric(frame, control, column, condition)
    diff = (left - right).dropna()
    diff.name = "diff"
    return diff


def cluster_summary(diff: pd.Series) -> dict:
    """Trial diffs -> dataset means -> equal-weight cluster bootstrap + TOST.

    Mirrors c7_stage._contrast_summary (same seed/iterations/unit).
    """
    import audit_block_c as cmod
    values = diff.to_numpy(dtype=float)
    datasets = np.array([int(key[0]) for key in diff.index])
    frame = pd.DataFrame({"dataset": datasets, "value": values})
    per_dataset = frame.groupby("dataset", sort=True)["value"].mean()
    ds_values = per_dataset.to_numpy(dtype=float)
    boot = cmod.cluster_bootstrap_ci(
        pd.DataFrame({"dataset": per_dataset.index.astype(int), "value": ds_values}),
        "value", "dataset", seed=BOOTSTRAP_SEED, iterations=BOOTSTRAP_ITERS,
    )
    return {
        "n_pairs": int(len(values)),
        "n_datasets": int(len(ds_values)),
        "mean": float(boot["mean"]),
        "sd_datasets": float(np.std(ds_values, ddof=1)) if len(ds_values) >= 2 else float("nan"),
        "ci_lo": float(boot["ci_lo"]),
        "ci_hi": float(boot["ci_hi"]),
        "per_dataset": per_dataset,
        "dataset_values": ds_values,
        "datasets_positive": int(np.sum(ds_values > 0)),
        "datasets_negative": int(np.sum(ds_values < 0)),
        "trials_positive": int(np.sum(values > 0)),
        "trials_negative": int(np.sum(values < 0)),
        "boot": boot,
    }


def save_figure(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    return path
