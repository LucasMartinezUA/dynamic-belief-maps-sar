#!/usr/bin/env python3
"""Block C robustness/generalisation campaign orchestrator.

The runner is intentionally campaign-oriented: every execution freezes its
configuration and provenance, emits long arm-level rows, validates pairing,
and writes summaries only inside the input campaign directory.  The simulation
core remains the frozen Block B implementation; C adds overlays and
reweighting at this boundary.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from shapely.geometry import LineString, Point
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - remote pixi supplies tqdm
    def tqdm(iterable, **_kwargs):
        return iterable


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from paired_stats import (  # noqa: E402
    classify_endpoint,
    classify_paired_endpoints,
    cluster_bootstrap_ci,
    paired_mean_ci,
    summarize_paired,
    tost_equivalence,
    validate_paired_keys,
)
from sarenv.analytics.detection_metrics import (  # noqa: E402
    expected_detection,
    expected_rmst,
    exposure_counts,
    exposure_events,
    stationary_victim_cell,
)
from sarenv.analytics.paths import generate_pizza_zigzag_path  # noqa: E402
from sarenv.analytics.simulation import SARSimulation  # noqa: E402
from sarenv.audit import stable_hash  # noqa: E402
from sarenv.core.loading import DatasetLoader  # noqa: E402
from sarenv.core.lost_person import LostPersonLocationGenerator  # noqa: E402
from sarenv.core.victim_models import VICTIM_MODELS  # noqa: E402

OUTPUT_DIR = ROOT / "results" / "audit_block_C"
SCHEMA_VERSION = "block-C-v1"
BLOCK_B_CODE_COMMIT = "d2eb0adbd780bf76a4e7820ede07de090bf9f15d"
BLOCK_B_CLOSURE_COMMIT = "df9f0680a10725ab16664d32305f0a8343b7f886"
FIX_PROFILE = "revision_B_base"
BEHAVIOR_PROFILE = "revision_B_base"
BELIEF_MODEL = "evidence"
PRIMARY_ENDPOINTS = ["P_detect", "RMST"]
SECONDARY_ENDPOINTS = ["L", "revisit_fraction", "exposure_multiplicity", "planner_wall_time"]
MARGIN_PDET = 0.01
MARGIN_RMST_REL = 0.02
ALPHA = 0.05
DEFAULT_P_D = 0.8
DEFAULT_TAU = 10_000.0
DEFAULT_BUDGET = 200_000.0
DEFAULT_SIZE = "xlarge"
DEFAULT_NUM_DRONES = 5
DEFAULT_FOV_DEG = 45.0
DEFAULT_ALTITUDE = 80.0
DEFAULT_DRONE_SPEED = 5.0
DEFAULT_VICTIM_SPEED = 0.5
DEFAULT_DT = 1.0
NOMINAL_W = 0.5
SCHEDULING_COMPATIBLE_C1 = {
    "git_commit": "f0684f8459a1f1424f630778b2c606434d0bce08",
    "source_hash": "27784880a3418d09b46f5afe2a7042757968dd5494a542fdab85e6a06fc60fb2",
    "experiment_id": "c1-sensitivity_20260826_073133_703049_c676526b",
}
# C4-local factual hotfix (2026-08-27): the pizza_repartition arm built its
# alignment probe with planning_mode="static" without static_paths, rejected by
# SARSimulation (planning_mode='static' requires static_paths). C4 crashed
# before any arm ran. The fix is C4-only; C1-C3 were produced by the base
# implementation and remain valid. New C4 accepts the C3 campaign produced by
# the base identity via this closed registry (exact IDs + hashes only) and
# records the acceptance in its provenance, exactly like the closed C1
# scheduling exception. Nothing else is accepted.
C4_HOTFIX_LEGACY_PARENTS = {
    "base_git_commit": "7b52d9149fd569229bcf4dbdfab434edfabc2e03",
    "base_source_hash": "7821c330697bd81d6d7c78669e14fc4e785ea7c98d4305a8f8d4a52af044ffbd",
    "legacy_experiment_ids": {
        "c2-mismatch": "c2-mismatch_20260826_181638_227156_4f4ff330",
        "c3-mobility": "c3-mobility_20260827_043124_280564_ae41202c",
    },
}
C4_HOTFIX_IMPLEMENTATION_COMMIT = "2ce3a901fecb5ddde502d0a5eeffb8f48d73c5fd"
C4_HOTFIX_IMPLEMENTATION_SOURCE_HASH = "fab97b80b871eeeac68aa1d75a33173cb368895f3e96803690e3cbc0acb1147d"
NOMINAL_H_REF = 8_000.0
C1_DATASETS = [1, 7, 10]
C2_DATASETS = [1, 7, 10]
C3_DATASETS = [1, 7, 10]
C4_DATASETS = [1, 5, 10]
C5_DATASETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30]
C1_SEEDS = list(range(42, 47))
TEN_NOMINAL_SENSITIVITY_CONDITIONS = [
    {"condition_id": "w_0", "curve": "w", "w": 0.0, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
    {"condition_id": "w_025", "curve": "w", "w": 0.25, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
    {"condition_id": "nominal", "curve": "nominal", "w": NOMINAL_W, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
    {"condition_id": "w_075", "curve": "w", "w": 0.75, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
    {"condition_id": "w_1", "curve": "w", "w": 1.0, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
    {"condition_id": "pd_060", "curve": "p_d", "w": NOMINAL_W, "p_d": 0.60, "tau": DEFAULT_TAU},
    {"condition_id": "pd_095", "curve": "p_d", "w": NOMINAL_W, "p_d": 0.95, "tau": DEFAULT_TAU},
    {"condition_id": "tau_2000", "curve": "tau", "w": NOMINAL_W, "p_d": DEFAULT_P_D, "tau": 2_000.0},
    {"condition_id": "tau_32000", "curve": "tau", "w": NOMINAL_W, "p_d": DEFAULT_P_D, "tau": 32_000.0},
    {"condition_id": "tau_128000", "curve": "tau", "w": NOMINAL_W, "p_d": DEFAULT_P_D, "tau": 128_000.0},
]
C5_EXPECTED_DATASETS = C5_DATASETS
BASELINE_UNTRACKED = [
    "scripts/a2_posthoc.py",
    "scripts/p0_mask_gate.py",
    "scripts/p0_w0_trajectory.py",
]
class CampaignAlreadyComplete(Exception):
    def __init__(self, experiment_id: str):
        super().__init__(experiment_id)
        self.experiment_id = experiment_id


_STAGE_RAW_FILES = {
    "c0-smoke": "c0_smoke_runs.csv",
    "c1-sensitivity": "c1_sensitivity_runs.csv",
    "c2-mismatch": "c2_prior_mismatch_runs.csv",
    "c2-lost-person": "c2_lost_person_runs.csv",
    "c3-mobility": "c3_mobility_runs.csv",
    "c4-attrition": "c4_attrition_runs.csv",
    "c5-generalization": "c5_generalization_runs.csv",
}



def _json_default(value: Any):
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _audit_commit_sha() -> str:
    override = os.environ.get("SARENV_C_IMPL_VERSION")
    if override:
        return override
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


_CRITICAL_SOURCES = [
    "scripts/audit_block_c.py",
    "scripts/c6_stage.py",
    "scripts/c7_stage.py",
    "scripts/run_block_c.sh",
    "scripts/paired_stats.py",
    "sarenv/analytics/detection_metrics.py",
    "sarenv/analytics/evidence_belief.py",
    "sarenv/analytics/midflight.py",
    "sarenv/analytics/simulation.py",
    "sarenv/analytics/paths.py",
    "sarenv/audit.py",
    "sarenv/core/loading.py",
    "sarenv/core/lost_person.py",
    "sarenv/core/victim_models.py",
    "tests/test_paired_stats.py",
    "tests/test_audit_block_c.py",
    "tests/test_evidence_belief.py",
    "tests/test_detection_metrics.py",
    "tests/test_c7_midflight.py",
    "tests/test_c7_pizza_replan.py",
]


def _source_hash() -> str:
    payload = {}
    for rel in _CRITICAL_SOURCES:
        path = ROOT / rel
        payload[rel] = path.read_text() if path.exists() else ""
    return stable_hash(payload)


def _tracked_changes() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _untracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


_C0_IMPLEMENTATION_PATHS = {
    "scripts/audit_block_c.py",
    "scripts/paired_stats.py",
    "tests/test_paired_stats.py",
    "tests/test_audit_block_c.py",
}


def _classify_preexisting_path(rel: str) -> dict:
    if rel in BASELINE_UNTRACKED:
        classification = "approved_untracked_source"
        can_affect = False
        reason = "pre-registered Block A script; not imported by C"
    elif rel.startswith("results/audit_block_A/") or rel.startswith("results/audit_block_B/") or rel in {"block_b_scripts.zip", "scripts/run_block_b.sh"}:
        classification = "historical_artifact"
        can_affect = False
        reason = "frozen historical A/B artifact not imported by C"
    elif rel == "EVIDENCE_BELIEF_PLAN.md":
        classification = "documentation"
        can_affect = False
        reason = "historical design documentation not imported by C"
    elif rel in _C0_IMPLEMENTATION_PATHS:
        classification = "active_source/config"
        can_affect = True
        reason = "C0 implementation must be committed before scientific C campaigns"
    else:
        classification = "unknown"
        can_affect = True
        reason = "unclassified untracked path"
    return {
        "path": rel,
        "classification": classification,
        "allowed_to_change": False,
        "can_affect_C_execution": can_affect,
        "reason": reason,
    }


def _capture_baseline_untracked(path: Path) -> dict:
    if (OUTPUT_DIR / "c0_manifest.json").exists():
        raise SystemExit("c0 preexisting baseline is frozen after c0_impl_commit")
    entries = []
    for rel in _untracked_paths():
        if rel == "results/audit_block_C" or rel.startswith("results/audit_block_C/"):
            continue
        entry = _classify_preexisting_path(rel)
        entry["sha256"] = _file_sha256(ROOT / rel)
        entries.append(entry)
    accepted = [entry for entry in entries if entry["classification"] in {
        "approved_untracked_source", "historical_artifact", "documentation",
    }]
    rejected = [entry for entry in entries if entry not in accepted and entry["path"] not in _C0_IMPLEMENTATION_PATHS]
    pending_c0 = [entry for entry in entries if entry["path"] in _C0_IMPLEMENTATION_PATHS]
    allowed = {entry["path"]: entry["sha256"] for entry in accepted}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy": "preexisting_untracked_hashed_baseline",
        "created_before_scientific_campaigns": True,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "allowed": allowed,
        "paths": entries,
        "pending_c0_sources": pending_c0,
        "rejected": rejected,
        "note": (
            "The preregistered three-file untracked allowlist was expanded before "
            "any C scientific campaign after inventory/classification found "
            "pre-existing historical artifacts. Active C source/config remains "
            "commit-bound and is never accepted as a baseline artifact."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return payload


def _load_baseline_untracked() -> dict:
    path = OUTPUT_DIR / "c0_preexisting_untracked.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("allowed", {})


def _require_clean_tree(*, require_baseline: bool = True, precommit_c0: bool = False) -> None:
    """Reject changed provenance, with one explicit pre-commit C0 smoke seam."""
    if os.environ.get("SARENV_C_NO_GATE") == "1":
        return
    allowed = _load_baseline_untracked() if require_baseline else {}
    baseline_path = OUTPUT_DIR / "c0_preexisting_untracked.json"
    if require_baseline and not baseline_path.exists():
        raise SystemExit("c0_preexisting_untracked.json is missing; run c0-preflight")
    if require_baseline:
        data = json.loads(baseline_path.read_text())
        if data.get("rejected"):
            raise SystemExit("preflight rejected untracked paths: " + ", ".join(item["path"] for item in data["rejected"]))
        for rel, expected_hash in allowed.items():
            file_path = ROOT / rel
            if not file_path.exists() or _file_sha256(file_path) != expected_hash:
                raise SystemExit(f"preexisting baseline changed or disappeared: {rel}")
    tracked = _tracked_changes()
    if tracked:
        if not precommit_c0:
            raise SystemExit("Tracked working-tree changes; git_commit cannot identify the implementation")
        changed_paths = {line[3:].strip() for line in tracked}
        prefix = ROOT.name + "/"
        changed_paths = {path[len(prefix):] if path.startswith(prefix) else path for path in changed_paths}
        unexpected_tracked = changed_paths - _C0_IMPLEMENTATION_PATHS
        if unexpected_tracked:
            raise SystemExit("Unexpected tracked changes during C0 smoke: " + ", ".join(sorted(unexpected_tracked)))
    unexpected = []
    for rel in _untracked_paths():
        if rel == "results/audit_block_C" or rel.startswith("results/audit_block_C/"):
            continue
        if rel in allowed and (ROOT / rel).exists() and _file_sha256(ROOT / rel) == allowed[rel]:
            continue
        if precommit_c0 and rel in _C0_IMPLEMENTATION_PATHS:
            continue
        unexpected.append(rel)
    if unexpected:
        raise SystemExit("Unexpected untracked files outside results/audit_block_C: " + ", ".join(unexpected))


def _base_config(args, stage: str) -> dict:
    commit = _audit_commit_sha()
    c0_manifest = OUTPUT_DIR / "c0_manifest.json"
    c0_impl = "pending"
    if c0_manifest.exists():
        c0_impl = json.loads(c0_manifest.read_text()).get("c0_impl_commit", "pending")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "implementation_version": f"revision_C_base@{commit[:12]}",
        "git_commit": commit,
        "source_hash": _source_hash(),
        "block_B_code_commit": BLOCK_B_CODE_COMMIT,
        "block_B_closure_commit": BLOCK_B_CLOSURE_COMMIT,
        "fix_profile": FIX_PROFILE,
        "behavior_profile": BEHAVIOR_PROFILE,
        "belief_model": BELIEF_MODEL,
        "c0_impl_commit": c0_impl,
        "primary_endpoints": PRIMARY_ENDPOINTS,
        "secondary_endpoints": SECONDARY_ENDPOINTS,
        "MARGIN_PDET": MARGIN_PDET,
        "MARGIN_RMST_REL": MARGIN_RMST_REL,
        "alpha": ALPHA,
        "ci_method": "Student-t paired mean CI",
        "bootstrap": {"cluster": True, "unit": "dataset_mean", "seed": 20260826, "iterations": 10000},
        "datasets": sorted(args.datasets) if args.datasets else [],
        "seeds": [],
        "planners": ["online_static_3step", "dynamic_3step"],
        "w": args.w,
        "p_d": args.p_d,
        "tau": args.tau,
        "budget": args.budget,
        "label": "exploratory",
        "_campaign_id": getattr(args, "campaign_id", None),
        "_rerun": bool(getattr(args, "rerun", False)),
        "size": args.size,
        "num_drones": args.num_drones,
        "fov_deg": args.fov_deg,
        "altitude": args.altitude,
        "drone_speed": args.drone_speed,
        "victim_speed": args.victim_speed,
        "dt": args.dt,
        "snapshot_interval": args.snapshot_interval,
        "horizon_policy": "T_common=max(T_i); H_common=T_common+1; path-ended cells keep survival",
        "seed_policy": {
            "planning_seed": "literal seed for initial positions and SARSimulation",
            "victim_seed": "SeedSequence child",
            "sensor_seed": "SeedSequence child",
            "failure_seed": "independent SeedSequence child",
            "condition_key_excludes_planner": True,
        },
        "condition_key": {"stage": stage},
        "classification_rule": "classify_paired_endpoints",
        "tuning_datasets": [],
        "holdout_datasets": [],
        "confirmation_datasets": [],
        "label": "exploratory",
    }


def _write_stage_complete(outdir: Path, config: dict) -> None:
    marker = outdir / "stage_complete.json"
    if marker.exists():
        return
    marker.write_text(json.dumps({
        "completed": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stage": config.get("stage"),
        "experiment_id": config.get("experiment_id"),
        "config_hash": config.get("config_hash"),
        "git_commit": config.get("git_commit"),
        "source_hash": config.get("source_hash"),
    }, indent=2, default=_json_default) + "\n")


def _campaign_complete(outdir: Path, config: dict) -> bool:
    marker = outdir / "stage_complete.json"
    if marker.exists():
        try:
            data = json.loads(marker.read_text())
            return bool(
                data.get("completed")
                and data.get("config_hash") == config.get("config_hash")
                and data.get("source_hash") == config.get("source_hash")
            )
        except (OSError, json.JSONDecodeError):
            return False
    raw_name = _STAGE_RAW_FILES.get(config.get("stage"))
    if not raw_name:
        return False
    raw = outdir / raw_name
    if not raw.exists():
        return False
    if config.get("stage") == "c0-smoke":
        inferred = (outdir / "c0_smoke_report.md").exists() and (outdir / "c0_smoke_paired_contrasts.csv").exists()
    else:
        expected = config.get("expected_rows")
        try:
            inferred = expected is not None and len(pd.read_csv(raw)) == int(expected)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            inferred = False
        gate_name = {
            "c1-sensitivity": "c1_gate.md",
            "c2-mismatch": "c2_gate.md",
            "c2-lost-person": "c2_lost_person_gate.md",
            "c3-mobility": "c3_gate.md",
            "c4-attrition": "c4_gate.md",
            "c5-generalization": "c5_gate.md",
        }.get(config.get("stage"))
        if gate_name:
            inferred = inferred and (outdir / gate_name).exists()
    if inferred:
        _write_stage_complete(outdir, config)
    return inferred


def _make_campaign(stage: str, base_config: dict, jobs: int, *, allow_resume: bool = False):
    frozen_base = {key: value for key, value in base_config.items() if not key.startswith("_")}
    config_hash = stable_hash(frozen_base)
    requested_id = base_config.get("_campaign_id")
    rerun = bool(base_config.get("_rerun", False))
    if requested_id and not rerun:
        requested_dir = OUTPUT_DIR / requested_id
        if requested_dir.exists():
            existing = json.loads((requested_dir / "config.json").read_text())
            if existing.get("stage") != stage or existing.get("config_hash") != config_hash:
                raise SystemExit(f"campaign id does not match frozen config: {requested_id}")
            if _campaign_complete(requested_dir, existing):
                raise CampaignAlreadyComplete(requested_id)
            if allow_resume and any(requested_dir.glob("ckpt_*")):
                # Durable per-job shards exist (possibly empty): continue the
                # same campaign instead of failing. The original config (and
                # its timestamp/experiment identity) is kept.
                return existing, requested_dir
            raise SystemExit(
                f"campaign exists but is incomplete: {requested_id}; use --rerun for a new campaign"
            )
    if not rerun and OUTPUT_DIR.exists():
        for candidate in sorted(OUTPUT_DIR.iterdir()):
            if not candidate.is_dir() or not (candidate / "config.json").exists():
                continue
            try:
                existing = json.loads((candidate / "config.json").read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if existing.get("stage") == stage and existing.get("config_hash") == config_hash:
                if _campaign_complete(candidate, existing):
                    raise CampaignAlreadyComplete(existing["experiment_id"])
                if allow_resume and any(candidate.glob("ckpt_*")):
                    return existing, candidate
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    experiment_id = requested_id if requested_id and rerun is False else f"{stage}_{timestamp}_{config_hash[:8]}"
    config = {
        **frozen_base,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "experiment_id": experiment_id,
        "jobs": int(jobs),
        "config_hash": config_hash,
        "rerun_policy": "completed exact-config campaigns are skipped; --rerun creates a new campaign",
    }
    outdir = OUTPUT_DIR / experiment_id
    outdir.mkdir(parents=False, exist_ok=False)
    (outdir / "config.json").write_text(json.dumps(config, indent=2, default=_json_default) + "\n")
    return config, outdir


_PROVENANCE_COLUMNS = [
    "implementation_version", "git_commit", "source_hash", "experiment_id", "config_hash",
    "dataset", "planning_seed", "victim_seed", "sensor_seed", "distribution_seed", "failure_seed",
    "belief_model", "fix_profile", "behavior_profile", "planning_mode", "w", "p_d", "tau",
    "simulation_victim_model", "evaluation_victim_model", "victim_sampling_model",
    "simulation_num_victims", "evaluation_num_victims", "num_drones", "budget", "size",
    "fov_deg", "altitude", "drone_speed", "victim_speed", "dt", "condition_key",
    "condition_key_hash", "T", "H", "T_common", "H_common",
]


def _none_if_missing(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "none"
    return value


def _with_provenance(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = frame.copy()
    for key in _PROVENANCE_COLUMNS:
        if key in frame.columns:
            continue
        value = config.get(key, "none")
        if key == "condition_key":
            value = _json_text(config.get(key, {}))
        if key == "condition_key_hash":
            value = stable_hash(config.get("condition_key", {}))
        frame[key] = _none_if_missing(value)
    if "condition_key" in frame.columns:
        frame["condition_key"] = frame["condition_key"].map(
            lambda x: _json_text(x) if isinstance(x, dict) else _none_if_missing(x)
        )
    if "condition_key_hash" not in frame.columns:
        frame["condition_key_hash"] = frame["condition_key"].map(stable_hash)
    for key in _PROVENANCE_COLUMNS:
        if key not in frame.columns:
            frame[key] = "none"
        frame[key] = frame[key].map(_none_if_missing)
    return frame


def _validate_campaign_frame(
    frame: pd.DataFrame,
    *,
    pair_keys: list[str] | None = None,
    expected_arms: list[str] | None = None,
    arm_col: str = "planning_mode",
    expected_rows: int | None = None,
) -> dict:
    missing = [column for column in _PROVENANCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"raw schema missing provenance columns: {missing}")
    if frame.isna().any().any():
        cols = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"raw schema contains nulls: {cols}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"row count {len(frame)} != expected {expected_rows}")
    for endpoint in ("P_detect", "RMST", "L", "T_common", "H_common"):
        if endpoint in frame.columns and not np.isfinite(frame[endpoint].to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite endpoint column: {endpoint}")
    if "P_detect" in frame.columns and not frame["P_detect"].between(0.0, 1.0).all():
        raise ValueError("P_detect outside [0,1]")
    result = {"valid": True, "rows": int(len(frame)), "missing": missing}
    if pair_keys is not None and expected_arms is not None:
        pairing = validate_paired_keys(frame, pair_keys, arm_col, expected_arms)
        if not pairing["valid"]:
            raise ValueError(f"invalid pairing: {pairing}")
        result["pairing"] = pairing
    return result


def _load_item(dataset_id: int, size: str):
    item = DatasetLoader(f"sarenv_dataset/{dataset_id}").load_environment(size)
    if item is None:
        raise ValueError(f"dataset {dataset_id} could not load at size={size}")
    return item


def _initial_positions(dataset_item, num_drones: int, seed: int) -> list[tuple[float, float]]:
    cx = (dataset_item.bounds[0] + dataset_item.bounds[2]) / 2.0
    cy = (dataset_item.bounds[1] + dataset_item.bounds[3]) / 2.0
    minx, miny, maxx, maxy = dataset_item.bounds
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(num_drones):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = 100.0 * np.sqrt(rng.uniform())
        positions.append((
            float(np.clip(cx + radius * np.cos(angle), minx, maxx)),
            float(np.clip(cy + radius * np.sin(angle), miny, maxy)),
        ))
    return positions


def _seed_bundle(dataset_id: int, planning_seed: int, condition_key: dict) -> dict:
    """Return independent deterministic streams for one paired condition."""
    condition_hash_int = int(stable_hash(condition_key)[:8], 16)
    sequence = np.random.SeedSequence([
        int(planning_seed) % (2**32), int(dataset_id) % (2**32), condition_hash_int,
    ])
    children = sequence.spawn(4)
    values = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return {
        "planning_seed": int(planning_seed),
        "victim_seed": values[0],
        "sensor_seed": values[1],
        "failure_seed": values[2],
        "distribution_seed": values[3],
        "condition_key": condition_key,
        "condition_key_hash": stable_hash(condition_key),
    }


def _build_sim(
    item,
    *,
    planning_mode: str,
    w: float,
    p_d: float,
    tau: float,
    budget: float,
    num_drones: int,
    planning_seed: int,
    implementation_version: str,
    config: dict | None = None,
    victim_speed: float = DEFAULT_VICTIM_SPEED,
    static_paths: list[LineString] | None = None,
    policy_seed: int | None = None,
    drone_identity: list[int] | None = None,
):
    config = config or {}
    return SARSimulation(
        dataset_item=item,
        num_drones=num_drones,
        num_victims=0,
        fov_deg=float(config.get("fov_deg", DEFAULT_FOV_DEG)),
        altitude=float(config.get("altitude", DEFAULT_ALTITUDE)),
        detection_probability=p_d,
        decay_tau=tau,
        victim_speed=float(config.get("victim_speed", victim_speed)),
        victim_model="random_walk",
        budget=budget,
        init_strategy="random",
        init_radius=100.0,
        drone_speed=float(config.get("drone_speed", DEFAULT_DRONE_SPEED)),
        planning_mode=planning_mode,
        revisit_weight=float(w),
        static_paths=static_paths,
        seed=planning_seed,
        dataset_id=None,
        implementation_version=implementation_version,
        fix_profile=FIX_PROFILE,
        belief_model=BELIEF_MODEL,
        policy_seed=policy_seed,
        drone_identity=drone_identity,
    )


def _run_from_positions(sim: SARSimulation, positions: list[tuple[float, float]], dt: float):
    sim.setup(initial_positions=list(positions), victim_positions=[])
    return sim.run_from_state(dt=dt, snapshot_interval=50, heatmap_interval=500)


def _normalise_prior(prior: np.ndarray) -> np.ndarray:
    p = np.asarray(prior, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("prior must be a finite non-negative two-dimensional array")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("prior has no positive mass")
    return p / total


def _path_exposure_metrics(sim: SARSimulation, paths: list[list[tuple[float, float]]], prior: np.ndarray, p_d: float, T_common: int):
    p = _normalise_prior(prior)
    counts = exposure_counts(paths, sim._get_visible_cells_world, sim.heatmap_shape)
    cell, ts, offsets = exposure_events(paths, sim._get_visible_cells_world, sim.heatmap_shape)
    P = expected_detection(p.ravel(), counts.ravel(), p_d)
    R = expected_rmst(p.ravel(), cell, ts, offsets, p_d, int(T_common))
    observed = set()
    for path in paths:
        for wx, wy in path:
            observed.update(sim._get_visible_cells_world(wx, wy))
    L = float(sum(p[r, c] for r, c in observed))
    total = int(counts.sum())
    multiplicity = float(total / len(observed)) if observed else 0.0
    revisit = float((total - len(observed)) / total) if total else 0.0
    return {
        "P_detect": float(P),
        "RMST": float(R),
        "L": L,
        "exposure_multiplicity": multiplicity,
        "revisit_fraction": revisit,
        "cell_exposure_counts": counts,
        "observed_cells": len(observed),
    }


def _run_pair(item, cfg: dict, *, dataset_id: int, planning_seed: int, w: float, p_d: float, tau: float, budget: float, num_drones: int, condition_key: dict, positions: list[tuple[float, float]] | None = None):
    if positions is None:
        positions = _initial_positions(item, num_drones, planning_seed)
    sims = {}
    for mode in ("online_static_3step", "dynamic_3step"):
        sim = _build_sim(
            item, planning_mode=mode, w=w, p_d=p_d, tau=tau, budget=budget,
            num_drones=num_drones, planning_seed=planning_seed,
            implementation_version=cfg["implementation_version"], config=cfg,
        )
        sim.dataset_id = dataset_id
        result = _run_from_positions(sim, positions, cfg.get("dt", DEFAULT_DT))
        sims[mode] = (sim, result)
    T_common = max(max(len(path) for path in sim.drone_paths) - 1 for sim, _ in sims.values())
    rows = []
    for mode, (sim, result) in sims.items():
        metrics = _path_exposure_metrics(sim, sim.drone_paths, item.heatmap, p_d, T_common)
        row = {
            "dataset": dataset_id, "planning_seed": planning_seed, "planning_mode": mode,
            "w": w, "p_d": p_d, "tau": tau,
            **{key: value for key, value in metrics.items() if key != "cell_exposure_counts"},
            "T": max(len(path) for path in sim.drone_paths) - 1,
            "H": max(len(path) for path in sim.drone_paths),
            "T_common": T_common, "H_common": T_common + 1,
            "actual_distance_total": float(result.total_distance),
            "planner_wall_time": float(result.planner_wall_time),
            "number_of_replans": int(result.number_of_replans),
            "initial_positions": _json_text(positions),
            "initial_positions_hash": stable_hash(positions),
            "trajectory_hash": stable_hash(sim.drone_paths),
            "simulation_victim_model": "none",
            "evaluation_victim_model": "stationary",
            "victim_sampling_model": "matched_prior_exact",
            "evaluation_num_victims": 1,
            "simulation_num_victims": 0,
            "condition_key": _json_text(condition_key),
            "condition_key_hash": stable_hash(condition_key),
        }
        rows.append(row)
    return pd.DataFrame(rows), sims, positions


def _paired_contrasts(frame: pd.DataFrame, *, keys: list[str], control: str, method: str) -> pd.DataFrame:
    status = validate_paired_keys(frame, keys, "planning_mode", [control, method])
    if not status["valid"]:
        raise ValueError(f"invalid paired frame: {status}")
    rows = []
    for key_values, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        arms = {row.planning_mode: row for row in group.itertuples(index=False)}
        left, right = arms[control], arms[method]
        h_common = max(float(left.H_common), float(right.H_common))
        row = dict(zip(keys, key_values))
        row.update({
            "control": control, "method": method,
            "dPdet": float(right.P_detect - left.P_detect),
            "dRMST_rel": float((left.RMST - right.RMST) / h_common),
            "dL": float(right.L - left.L),
            "H_common": h_common,
            "exposure_multiplicity_control": float(left.exposure_multiplicity),
            "exposure_multiplicity_method": float(right.exposure_multiplicity),
            "revisit_fraction_control": float(left.revisit_fraction),
            "revisit_fraction_method": float(right.revisit_fraction),
            "replans_control": float(getattr(left, "number_of_replans", np.nan)),
            "replans_method": float(getattr(right, "number_of_replans", np.nan)),
            "planner_wall_time_control": float(getattr(left, "planner_wall_time", np.nan)),
            "planner_wall_time_method": float(getattr(right, "planner_wall_time", np.nan)),
            "initial_positions_hash": str(left.initial_positions_hash),
            "planning_seed_common": int(left.planning_seed),
        })
        for metadata_col in ("curve", "mobility_model", "condition_id", "alpha", "w", "k"):
            if metadata_col in group.columns and metadata_col not in row:
                values = group[metadata_col].drop_duplicates().tolist()
                if len(values) == 1:
                    row[metadata_col] = values[0]
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_table(contrasts: pd.DataFrame, group_cols: list[str], *, pdet_margin: float = MARGIN_PDET, rmst_margin: float = MARGIN_RMST_REL) -> pd.DataFrame:
    rows = []
    for group_values, group in contrasts.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        pdet = summarize_paired(group["dPdet"].to_numpy(), pdet_margin)
        rmst = summarize_paired(group["dRMST_rel"].to_numpy(), rmst_margin)
        row = dict(zip(group_cols, group_values))
        for prefix, summary in (("dPdet", pdet), ("dRMST_rel", rmst)):
            for key in ("n", "mean", "sd", "se", "ci_lo", "ci_hi", "favorable_count", "n_total"):
                row[f"{prefix}_{key}"] = summary.get(key)
            row[f"{prefix}_tost_p_lower"] = summary["tost"]["p_lower"]
            row[f"{prefix}_tost_p_upper"] = summary["tost"]["p_upper"]
            row[f"{prefix}_tost_p_max"] = summary["tost"]["p_max"]
            row[f"{prefix}_equivalent"] = summary["tost"]["equivalent"]
        row["state"] = classify_paired_endpoints(pdet, rmst)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_frame(frame: pd.DataFrame, outdir: Path, filename: str):
    frame.to_csv(outdir / filename, index=False)

def _progress(iterable, args, description: str, total: int | None = None):
    return tqdm(
        iterable,
        desc=description,
        total=total,
        unit="job",
        disable=bool(getattr(args, "no_progress", False)),
        dynamic_ncols=True,
    )

def _parallel_jobs(args, jobs: list[tuple], worker, description: str) -> list:
    """Run independent statistical-unit jobs and report completions, not dispatches."""
    if args.jobs > 1:
        completed = Parallel(
            n_jobs=args.jobs,
            backend="loky",
            return_as="generator_unordered",
        )(delayed(worker)(*job) for job in jobs)
    else:
        completed = (worker(*job) for job in jobs)
    return list(_progress(completed, args, description, total=len(jobs)))


def _job_slug(job: tuple) -> str:
    return "_".join(str(value).replace("/", "-") for value in job)


def _pack_payload(frames: list) -> list:
    """JSON-safe packing of worker payloads (DataFrames survive intact)."""
    packed = []
    for frame in frames:
        if isinstance(frame, pd.DataFrame):
            packed.append({
                "__dataframe__": True,
                "columns": [str(column) for column in frame.columns],
                "rows": frame.astype(object).where(pd.notna(frame), None).to_dict("records"),
            })
        else:
            packed.append(frame)
    return packed


def _unpack_payload(payload: list) -> list:
    frames = []
    for item in payload:
        if isinstance(item, dict) and item.get("__dataframe__"):
            frames.append(pd.DataFrame(item["rows"], columns=item["columns"]))
        else:
            frames.append(item)
    return frames


def _parallel_jobs_ckpt(args, jobs: list[tuple], worker, description: str, outdir, tag: str, key_cols: list[str]) -> list:
    """``_parallel_jobs`` with durable per-job checkpoint shards.

    Every finished job is written atomically (temp + ``os.replace``) to
    ``outdir/ckpt_<tag>/job_<slug>.json``. On resume the completed shards are
    loaded and only the missing jobs run, so a crash/reboot loses at most the
    in-flight trial. Fail-closed: a corrupt/duplicate/mismatched shard aborts
    (never silently mixes rows).
    """
    shard_dir = Path(outdir) / f"ckpt_{tag}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    done: dict[tuple, list] = {}
    for shard in sorted(shard_dir.glob("job_*.json")):
        try:
            data = json.loads(shard.read_text())
            frames = _unpack_payload(data["payload"])
            key = tuple(data["key"])
            first = frames[0]
            if not isinstance(first, pd.DataFrame) or not all(
                column in first.columns for column in key_cols
            ):
                raise ValueError("shard payload missing key columns")
            recorded = [first[column].iloc[0] for column in key_cols]
            if not all(a == b for a, b in zip(recorded, key)):
                raise ValueError(
                    f"shard key {key!r} does not match recorded rows {recorded!r}"
                )
            if key in done:
                raise ValueError(f"duplicate checkpoint shard for job {key!r}")
            done[key] = frames
        except Exception as exc:
            raise SystemExit(f"corrupt checkpoint shard {shard}: {exc!r}") from exc

    remaining = [job for job in jobs if tuple(job) not in done]
    new: dict[tuple, list] = {}
    if remaining:
        # Each completion is persisted immediately (shard per job), so a crash
        # loses at most the in-flight trial. The worker returns its own job key
        # (jolib's unordered generator yields results without their inputs).
        def ckpt_worker(*job):
            return job, worker(*job)

        if args.jobs > 1:
            completed = Parallel(
                n_jobs=args.jobs,
                backend="loky",
                return_as="generator_unordered",
            )(delayed(ckpt_worker)(*job) for job in remaining)
        else:
            completed = (ckpt_worker(*job) for job in remaining)
        for job, frames in _progress(completed, args, description, total=len(remaining)):
            key = tuple(job)
            new[key] = frames
            slug = _job_slug(job)
            tmp = shard_dir / f"job_{slug}.json.tmp"
            tmp.write_text(
                json.dumps(
                    {"key": list(job), "payload": _pack_payload(frames)},
                    default=_json_default,
                )
                + "\n"
            )
            os.replace(tmp, shard_dir / f"job_{slug}.json")

    ordered: list[list] = []
    for job in jobs:
        key = tuple(job)
        if key in done:
            ordered.append(done[key])
        elif key in new:
            ordered.append(new[key])
        else:  # pragma: no cover - structural invariant
            raise SystemExit(f"checkpoint bookkeeping lost job {job!r}")
    return ordered


def _design_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_identity": "revision_C_base; behavior remains revision_B_base",
        "hypotheses": {
            "H-C1": "the operational effect does not depend on a narrow w/p_d/tau point",
            "H-C2": "dynamic-control degrades measurably under Q != P",
            "H-C3": "P_detect/RMST sign is maintained across mobility models",
            "H-C4": "fair initial attrition separates replanning from belief",
            "H-C5": "the effect reproduces by dataset without universalising the mean",
        },
        "primary_endpoints": PRIMARY_ENDPOINTS,
        "secondary_endpoints": SECONDARY_ENDPOINTS,
        "margins": {"P_detect": MARGIN_PDET, "RMST_relative": MARGIN_RMST_REL},
        "nominal": {"w": NOMINAL_W, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
        "c1_h_ref_seconds": NOMINAL_H_REF,
        "c1_tau_over_h_ref": [0.25, 1.25, 4.0, 16.0],
        "c1_tau_seconds": [2_000.0, 10_000.0, 32_000.0, 128_000.0],
        "c1_p_d": [0.60, 0.80, 0.95],
        "c1_w": [0.0, 0.25, 0.5, 0.75, 1.0],
        "c1_unique_conditions": TEN_NOMINAL_SENSITIVITY_CONDITIONS,
        "c1b": "not_run_by_design; no post-C1 tuning",
        "datasets": {
            "C1-C3": [1, 7, 10], "C4": [1, 5, 10], "C5-Q1": C5_DATASETS,
        },
        "seeds": {"pilot": C1_SEEDS, "nominal_confirmation": list(range(42, 52))},
        "commit_chronology": ["Block B closure", "C0 implementation commit", "C1-C5 campaigns", "C closure commit"],
    }


def run_c0_preflight(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = OUTPUT_DIR / "c0_preexisting_untracked.json"
    baseline = _capture_baseline_untracked(baseline_path)
    design = _design_payload()
    (OUTPUT_DIR / "c0_design.json").write_text(json.dumps(design, indent=2, default=_json_default) + "\n")
    tracked = _tracked_changes()
    rejected = baseline["rejected"]
    pending = baseline["pending_c0_sources"]
    lines = [
        f"HEAD={_audit_commit_sha()}",
        f"B_CLOSURE={BLOCK_B_CLOSURE_COMMIT}",
        f"B_CODE_ANCESTOR={BLOCK_B_CODE_COMMIT}",
        f"source_hash={_source_hash()}",
        "tracked_changes=" + json.dumps(tracked),
        "inventory_count=" + str(len(baseline["paths"])),
        "allowed_baseline_count=" + str(len(baseline["allowed"])),
        "pending_c0_sources=" + json.dumps(pending, sort_keys=True),
        "rejected_inventory=" + json.dumps(rejected, sort_keys=True),
        "status=" + ("PASS" if not rejected else "INVALID"),
        (
            "note=The preregistered three-file untracked allowlist was expanded "
            "before any C scientific campaign after classifying pre-existing "
            "historical artifacts/docs by exact hash. Active C source/config is "
            "not hidden in the baseline and must be committed in c0_impl_commit."
        ),
    ]
    (OUTPUT_DIR / "c0_preflight.txt").write_text("\n".join(lines) + "\n")
    if rejected:
        raise SystemExit("C0 preflight INVALID: unknown or active non-C untracked paths: " + ", ".join(item["path"] for item in rejected))
    _require_clean_tree(require_baseline=True, precommit_c0=True)
    return baseline


def run_c0_smoke(args):
    config = _base_config(args, "c0-smoke")
    config.update({
        "datasets": [1], "seeds": [args.base_seed], "size": "small", "budget": args.budget,
        "label": "exploratory", "simulation_victim_model": "none",
        "evaluation_victim_model": "stationary", "victim_sampling_model": "matched_prior_exact",
        "condition_key": {"stage": "c0-smoke", "w": NOMINAL_W, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
    })
    config, outdir = _make_campaign("c0-smoke", config, args.jobs)
    item = _load_item(1, "small")
    condition = config["condition_key"]
    bundle = _seed_bundle(1, args.base_seed, condition)
    runs, _, positions = _run_pair(
        item, config, dataset_id=1, planning_seed=args.base_seed, w=NOMINAL_W,
        p_d=DEFAULT_P_D, tau=DEFAULT_TAU, budget=args.budget, num_drones=args.num_drones,
        condition_key=condition,
    )
    runs["victim_seed"] = bundle["victim_seed"]
    runs["sensor_seed"] = bundle["sensor_seed"]
    runs["failure_seed"] = "none"
    runs["distribution_seed"] = "none"
    runs = _with_provenance(runs, config)
    _validate_campaign_frame(runs, pair_keys=["dataset", "planning_seed"], expected_arms=["online_static_3step", "dynamic_3step"])
    contrasts = _paired_contrasts(runs, keys=["dataset", "planning_seed"], control="online_static_3step", method="dynamic_3step")
    _write_frame(runs, outdir, "c0_smoke_runs.csv")
    _write_frame(contrasts, outdir, "c0_smoke_paired_contrasts.csv")
    report = {
        "campaign_id": config["experiment_id"], "rows": len(runs), "pairs": len(contrasts),
        "initial_positions_equal": runs["initial_positions_hash"].nunique() == 1,
        "planning_seed_common": runs["planning_seed"].nunique() == 1,
        "victim_seed_distinct_from_sensor": int(runs["victim_seed"].iloc[0]) != int(runs["sensor_seed"].iloc[0]),
        "victim_seed": int(runs["victim_seed"].iloc[0]),
        "sensor_seed": int(runs["sensor_seed"].iloc[0]),
        "endpoints_finite": bool(np.isfinite(runs[["P_detect", "RMST", "L"]].to_numpy(dtype=float)).all()),
        "P_detect_range": [float(runs.P_detect.min()), float(runs.P_detect.max())],
        "T_common": sorted(runs.T_common.unique().tolist()), "H_common": sorted(runs.H_common.unique().tolist()),
        "source_hash": config["source_hash"], "config_hash": config["config_hash"],
        "positions": positions,
    }
    (outdir / "c0_smoke_report.md").write_text("# C0 smoke\n\n```json\n" + json.dumps(report, indent=2, default=_json_default) + "\n```\n")
    if not all(report[key] for key in ("initial_positions_equal", "planning_seed_common", "victim_seed_distinct_from_sensor", "endpoints_finite")):
        raise SystemExit("C0 smoke invariants failed")
    print(f"campaign {config['experiment_id']} -> {outdir}")
    return config["experiment_id"]


def _b3_nominal_status() -> dict[int, str]:
    manifest = ROOT / "results/audit_block_B/block_B_manifest_final.json"
    if not manifest.exists():
        return {}
    data = json.loads(manifest.read_text())
    if "B3" not in data.get("stages", {}):
        return {}
    status_path = ROOT / "results/audit_block_B" / data["stages"]["B3"]["experiment_id"] / "b3_gate_status.csv"
    if status_path.exists():
        frame = pd.read_csv(status_path)
        frame = frame[frame["w"] == 0.5]
        return {int(row.dataset): str(row.verdict) for row in frame.itertuples()}
    return {1: "GREEN", 7: "GREEN", 10: "GREEN"} if "GREEN" in data["stages"]["B3"].get("verdict_w05", "") else {}


def run_c1_sensitivity(args):
    _require_c0_ready()
    config = _base_config(args, "c1-sensitivity")
    datasets = sorted(args.datasets or C1_DATASETS)
    seeds = list(args.seeds_list or C1_SEEDS)
    full_design = bool(
        datasets == sorted(C1_DATASETS)
        and seeds == C1_SEEDS
        and args.size == DEFAULT_SIZE
        and args.budget == DEFAULT_BUDGET
        and args.num_drones == DEFAULT_NUM_DRONES
        and args.drone_speed == DEFAULT_DRONE_SPEED
        and args.fov_deg == DEFAULT_FOV_DEG
        and args.altitude == DEFAULT_ALTITUDE
        and args.victim_speed == DEFAULT_VICTIM_SPEED
        and args.dt == DEFAULT_DT
    )
    config.update({
        "datasets": datasets, "seeds": seeds, "confirmation_datasets": datasets,
        "label": "sensitivity" if full_design else "functional-only",
        "scientific_valid": full_design,
        "planners": ["online_static_3step", "dynamic_3step"],
        "simulation_victim_model": "none", "evaluation_victim_model": "stationary",
        "victim_sampling_model": "matched_prior_exact", "conditions": TEN_NOMINAL_SENSITIVITY_CONDITIONS,
        "condition_key": {"stage": "c1-sensitivity", "evaluation": "stationary matched-prior"},
        "H_ref": float(args.budget / (args.num_drones * args.drone_speed)), "nominal_condition_id": "nominal",
        "expected_rows": len(TEN_NOMINAL_SENSITIVITY_CONDITIONS) * len(datasets) * len(seeds) * 2,
    })
    config, outdir = _make_campaign("c1-sensitivity", config, args.jobs)
    jobs = [(dataset, seed, condition) for dataset in datasets for seed in seeds for condition in TEN_NOMINAL_SENSITIVITY_CONDITIONS]

    def one(dataset, seed, condition):
        item = _load_item(dataset, args.size)
        key = {
            "stage": "c1-sensitivity",
            "condition_id": condition["condition_id"],
            "evaluation": "stationary",
        }
        bundle = _seed_bundle(dataset, seed, key)
        frame, _, _ = _run_pair(
            item,
            config,
            dataset_id=dataset,
            planning_seed=seed,
            w=condition["w"],
            p_d=condition["p_d"],
            tau=condition["tau"],
            budget=args.budget,
            num_drones=args.num_drones,
            condition_key=key,
        )
        frame["condition_id"] = condition["condition_id"]
        frame["curve"] = condition["curve"]
        frame["victim_seed"] = bundle["victim_seed"]
        frame["sensor_seed"] = bundle["sensor_seed"]
        frame["failure_seed"] = "none"
        frame["distribution_seed"] = "none"
        return frame

    frames = _parallel_jobs(args, jobs, one, "C1 sensitivity")
    runs = _with_provenance(pd.concat(frames, ignore_index=True), config)
    _validate_campaign_frame(runs, pair_keys=["dataset", "planning_seed", "condition_id"], expected_arms=["online_static_3step", "dynamic_3step"], expected_rows=config["expected_rows"])
    contrasts = _paired_contrasts(runs, keys=["dataset", "planning_seed", "condition_id"], control="online_static_3step", method="dynamic_3step")
    expected_pairs = len(datasets) * len(seeds) * len(TEN_NOMINAL_SENSITIVITY_CONDITIONS)
    if len(contrasts) != expected_pairs:
        raise ValueError(f"C1 pair count {len(contrasts)} != {expected_pairs}")
    summary = _summary_table(contrasts, ["dataset", "condition_id", "curve"])
    response = contrasts.groupby(["dataset", "condition_id"], as_index=False).agg(
        mean_dPdet=("dPdet", "mean"), mean_dRMST_rel=("dRMST_rel", "mean"), mean_dL=("dL", "mean"), n=("dPdet", "size"),
    )
    _write_frame(runs, outdir, "c1_sensitivity_runs.csv")
    _write_frame(contrasts, outdir, "c1_paired_contrasts.csv")
    _write_frame(summary, outdir, "c1_sensitivity_summary.csv")
    _write_frame(response, outdir, "c1_response_surfaces.csv")
    b3 = _b3_nominal_status()
    nominal = summary[summary["condition_id"] == "nominal"]
    nonnominal = summary[summary["condition_id"] != "nominal"]
    sensitivity_region = nonnominal[nonnominal["condition_id"] != "w_0"]
    nonadverse = float((sensitivity_region["state"] != "WORSENING").mean()) if len(sensitivity_region) else float("nan")
    both_positive = float(((sensitivity_region["dPdet_mean"] > 0) & (sensitivity_region["dRMST_rel_mean"] > 0)).mean()) if len(sensitivity_region) else float("nan")
    practical_improvement = float((sensitivity_region["state"] == "IMPROVEMENT").mean()) if len(sensitivity_region) else float("nan")
    b3_green = sum(value in {"GREEN", "OK", "INTERMEDIATE"} for value in b3.values())
    adverse_half = any(
        float((sensitivity_region[sensitivity_region["curve"] == curve]["state"] == "WORSENING").mean()) >= 0.5
        for curve in sorted(sensitivity_region["curve"].dropna().unique())
    ) if len(sensitivity_region) else False
    if not full_design:
        gate = "FUNCTIONAL_ONLY"
    elif b3_green < 2:
        gate = "FRAGILE"
    elif adverse_half:
        gate = "FRAGILE"
    elif nonadverse >= 0.75 and both_positive >= 0.60 and practical_improvement >= 0.60:
        gate = "ROBUST"
    else:
        gate = "SENSITIVE"
    gate_text = {
        "gate": gate,
        "scientific_valid": full_design,
        "b3_nominal_status": b3,
        "w0_control_states": summary[summary["condition_id"] == "w_0"].set_index("dataset")["state"].to_dict(),
        "non_nominal_non_adverse_fraction_excluding_w0": nonadverse,
        "non_nominal_both_positive_fraction_excluding_w0": both_positive,
        "practical_improvement_fraction_excluding_w0": practical_improvement,
        "nominal_is_diagnostic_only": True,
        "classification": "classify_paired_endpoints",
        "parameter_selection": "none; w=.5,p_d=.8,tau=10000 remain frozen",
    }
    (outdir / "c1_gate.md").write_text("# C1 gate\n\n```json\n" + json.dumps(gate_text, indent=2, default=_json_default) + "\n```\n")
    return config["experiment_id"]


def valid_prior_support(sim: SARSimulation, prior: np.ndarray) -> np.ndarray:
    """Return the operational/searchable mask used to build U_valid."""
    p = np.asarray(prior)
    mask = np.asarray(sim._valid_domain_mask, dtype=bool)
    if p.ndim != 2 or mask.ndim != 2 or p.shape != mask.shape:
        raise ValueError(f"prior/mask must be two-dimensional and aligned: {p.shape}, {mask.shape}")
    if not mask.any():
        raise ValueError("operational/searchable domain is empty")
    return mask.copy()


def mix_uniform_target(prior: np.ndarray, valid_mask: np.ndarray, alpha: float) -> np.ndarray:
    """Return normalized Q_alpha=(1-alpha)P+alpha U_valid."""
    p = _normalise_prior(prior)
    mask = np.asarray(valid_mask, dtype=bool)
    if p.shape != mask.shape or p.ndim != 2 or not mask.any():
        raise ValueError("prior and valid_mask must be aligned non-empty 2-D arrays")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    uniform = np.zeros_like(p)
    uniform[mask] = 1.0 / float(mask.sum())
    q = (1.0 - alpha) * p + alpha * uniform
    return q / q.sum()


def target_distribution_from_lost_person(dataset_item, samples: int, distribution_seed: int, shape: tuple[int, int], bounds: tuple[float, float, float, float]):
    """Sample and bin one frozen lost-person target distribution."""
    generator = LostPersonLocationGenerator(dataset_item, seed=distribution_seed)
    points = generator.generate_locations(n=int(samples), percent_random_samples=0)
    counts = np.zeros(shape, dtype=np.float64)
    minx, miny, maxx, maxy = bounds
    dx = (maxx - minx) / shape[1]
    dy = (maxy - miny) / shape[0]
    valid = 0
    for point in points:
        col = int((point.x - minx) / dx)
        row = int((point.y - miny) / dy)
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            counts[row, col] += 1.0
            valid += 1
    if valid == 0:
        raise ValueError("lost-person target distribution has no valid samples")
    q = counts / counts.sum()
    diagnostics = {
        "distribution_seed": int(distribution_seed),
        "requested_samples": int(samples),
        "generated_samples": int(len(points)),
        "valid_samples": int(valid),
        "generated_fraction": float(len(points) / max(int(samples), 1)),
        "valid_fraction": float(valid / max(int(samples), 1)),
        "valid_fraction_of_generated": float(valid / max(len(points), 1)),
        "target_distribution_hash": stable_hash(q.tolist()),
    }
    return q, diagnostics


def _save_distribution_npz(outdir: Path, distributions: dict[int, np.ndarray], metadata: dict):
    np.savez_compressed(outdir / "c2_lost_person_distributions.npz", **{f"dataset_{key}": value for key, value in distributions.items()})
    (outdir / "c2_lost_person_distributions.json").write_text(json.dumps(metadata, indent=2, default=_json_default) + "\n")


def _read_gate_payload(path: Path) -> dict:
    text = path.read_text()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    else:
        text = text.split("\n\n", 1)[1]
    return json.loads(text.strip())


def _is_scheduling_compatible_c1(parent_id: str, parent_config: dict) -> bool:
    return bool(
        parent_id == SCHEDULING_COMPATIBLE_C1["experiment_id"]
        and parent_config.get("stage") == "c1-sensitivity"
        and parent_config.get("git_commit") == SCHEDULING_COMPATIBLE_C1["git_commit"]
        and parent_config.get("source_hash") == SCHEDULING_COMPATIBLE_C1["source_hash"]
        and parent_config.get("scientific_valid") is True
    )


def _is_c4_hotfix_legacy_parent(parent_id: str, parent_config: dict) -> bool:
    """Accept only the closed C2/C3 parents produced by the C4 hotfix base.

    The C4 probe fix is stage-local; C2/C3 were produced by the base identity
    and remain scientifically valid. Only the exact campaign IDs with the
    exact base git/source hashes are accepted; every other parent is
    rejected.
    """
    stage = parent_config.get("stage")
    expected_id = C4_HOTFIX_LEGACY_PARENTS["legacy_experiment_ids"].get(stage)
    return bool(
        expected_id is not None
        and parent_id == expected_id
        and parent_config.get("git_commit") == C4_HOTFIX_LEGACY_PARENTS["base_git_commit"]
        and parent_config.get("source_hash") == C4_HOTFIX_LEGACY_PARENTS["base_source_hash"]
        and parent_config.get("scientific_valid") is True
    )

def _is_c4_hotfix_implementation(config: dict) -> bool:
    """True if a campaign was produced by the C4 hotfix implementation.

    C4-C6 were executed with implementation commit 2ce3a90 / hash fab97b80.
    The only difference between that and the current HEAD is a validation
    constant restored for c-final; no scientific behaviour changed.
    """
    return bool(
        config.get("git_commit") == C4_HOTFIX_IMPLEMENTATION_COMMIT
        and config.get("source_hash") == C4_HOTFIX_IMPLEMENTATION_SOURCE_HASH
    )


def _require_parent_campaign(args, expected_stage: str, gate_filename: str) -> tuple[str, dict, dict]:
    parent_id = args.input_experiment_id
    if not parent_id:
        raise SystemExit(f"{expected_stage} requires --input-experiment-id from the preceding stage")
    parent_dir = OUTPUT_DIR / parent_id
    config_path = parent_dir / "config.json"
    gate_path = parent_dir / gate_filename
    if not config_path.exists() or not gate_path.exists():
        raise SystemExit(f"parent campaign is incomplete: {parent_id}")
    parent_config = json.loads(config_path.read_text())
    _require_c0_ready()
    current_commit = _audit_commit_sha()
    current_source_hash = _source_hash()
    exact_identity = bool(
        parent_config.get("git_commit") == current_commit
        and parent_config.get("source_hash") == current_source_hash
    )
    scheduling_compatible = bool(
        expected_stage == "c1-sensitivity"
        and _is_scheduling_compatible_c1(parent_id, parent_config)
    )
    hotfix_legacy = bool(
        expected_stage == "c3-mobility"
        and _is_c4_hotfix_legacy_parent(parent_id, parent_config)
    )
    hotfix_identity = bool(_is_c4_hotfix_implementation(parent_config))
    if not exact_identity and not scheduling_compatible and not hotfix_legacy and not hotfix_identity:
        raise SystemExit(
            f"parent campaign provenance differs from current C0 implementation: {parent_id}"
        )
    parent_config = dict(parent_config)
    parent_config["_compatibility_status"] = (
        "exact_identity"
        if exact_identity
        else "accepted_c4_hotfix_implementation"
        if hotfix_identity
        else "accepted_hotfix_legacy_parent"
        if hotfix_legacy
        else "accepted_scheduling_only_c1_parent"
    )
    if parent_config.get("stage") != expected_stage:
        raise SystemExit(
            f"expected parent stage {expected_stage!r}, got {parent_config.get('stage')!r}"
        )
    if (
        expected_stage == "c1-sensitivity"
        and parent_config.get("scientific_valid") is not True
    ):
        raise SystemExit(
            f"parent C1 campaign is not a full scientific design: {parent_id}"
        )
    if parent_config.get("scientific_valid") is False:
        raise SystemExit(f"parent campaign is functional-only: {parent_id}")
    gate = _read_gate_payload(gate_path)
    gate_value = gate.get("gate") or gate.get("scientific_gate")
    if gate_value in {"INVALID", "FUNCTIONAL_ONLY"}:
        raise SystemExit(
            f"parent campaign gate is not scientifically valid: {gate_value}"
        )
    return parent_id, parent_config, gate
def _require_c0_ready() -> dict:
    manifest_path = OUTPUT_DIR / "c0_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("C1 requires a frozen c0_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("c0_impl_commit") or manifest.get("c0_impl_commit") == "pending":
        raise SystemExit("c0_manifest.json does not identify a clean implementation commit")
    if manifest["c0_impl_commit"] != _audit_commit_sha():
        raise SystemExit("c0 implementation commit differs from current HEAD; freeze a new implementation commit")
    if manifest.get("source_hash") != _source_hash():
        raise SystemExit("c0 source_hash differs from current source; freeze a new implementation commit")
    return manifest


def _is_full_scientific_design(args, datasets: list[int], seeds: list[int], expected_datasets: list[int]) -> bool:
    return bool(
        datasets == sorted(expected_datasets)
        and seeds == list(range(42, 52))
        and args.size == DEFAULT_SIZE
        and args.budget == DEFAULT_BUDGET
        and args.w == NOMINAL_W
        and args.p_d == DEFAULT_P_D
        and args.tau == DEFAULT_TAU
        and args.num_drones == DEFAULT_NUM_DRONES
        and args.fov_deg == DEFAULT_FOV_DEG
        and args.altitude == DEFAULT_ALTITUDE
        and args.drone_speed == DEFAULT_DRONE_SPEED
        and args.victim_speed == DEFAULT_VICTIM_SPEED
        and args.dt == DEFAULT_DT
    )
def _require_nominal_parameters(args) -> None:
    if (
        not math.isclose(float(args.w), NOMINAL_W)
        or not math.isclose(float(args.p_d), DEFAULT_P_D)
        or not math.isclose(float(args.tau), DEFAULT_TAU)
    ):
        raise SystemExit("this Block C stage is frozen to w=.5, p_d=.8, and tau=10000")




def _c2_alpha_frame(rows: list[dict], alpha: float) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("C2 alpha condition produced no arm rows")
    frame["condition_id"] = f"alpha_{alpha:g}"
    return frame


def run_c2_mismatch(args):
    _require_nominal_parameters(args)
    parent_id, parent_config, parent_gate = _require_parent_campaign(args, "c1-sensitivity", "c1_gate.md")
    config = _base_config(args, "c2-mismatch")
    datasets = sorted(args.datasets or C2_DATASETS)
    seeds = list(args.seeds_list or range(42, 52))
    alphas = [0.0, 0.25, 0.5, 0.75]
    full_design = _is_full_scientific_design(args, datasets, seeds, C2_DATASETS)
    config.update({
        "parent_experiment_id": parent_id,
        "parent_gate": parent_gate.get("gate") or parent_gate.get("scientific_gate"),
        "parent_git_commit": parent_config.get("git_commit"),
        "parent_source_hash": parent_config.get("source_hash"),
        "parent_compatibility_status": parent_config.get("_compatibility_status"),
        "scientific_valid": full_design,
        "datasets": datasets, "seeds": seeds, "alphas": alphas,
        "label": "confirmatory" if full_design else "functional-only",
        "simulation_victim_model": "none", "evaluation_victim_model": "stationary",
        "victim_sampling_model": "none_reweighted_Q_alpha",
        "evaluation_distribution": "Q_alpha=(1-alpha)P+alpha*U_valid",
        "trajectory_reused": True,
        "parallel_unit": ["dataset", "planning_seed"],
        "condition_key": {"stage": "c2-mismatch", "w": NOMINAL_W, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
        "expected_rows": len(datasets) * len(seeds) * len(alphas) * 2,
    })
    config, outdir = _make_campaign("c2-mismatch", config, args.jobs)

    def one(dataset, seed):
        item = _load_item(dataset, args.size)
        probe = _build_sim(
            item, planning_mode="online_static_3step", w=NOMINAL_W,
            p_d=DEFAULT_P_D, tau=DEFAULT_TAU, budget=args.budget,
            num_drones=args.num_drones, planning_seed=42,
            implementation_version=config["implementation_version"], config=config,
        )
        prior = _normalise_prior(item.heatmap)
        mask = valid_prior_support(probe, prior)
        zero_prior = int(np.sum(mask & (prior == 0)))
        meta = {
            "valid_support_cells": int(mask.sum()),
            "zero_prior_cells_inside_domain": zero_prior,
        }
        condition = {
            "stage": "c2-mismatch", "w": NOMINAL_W,
            "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU,
        }
        base_frame, sims, _ = _run_pair(
            item, config, dataset_id=dataset, planning_seed=seed,
            w=NOMINAL_W, p_d=DEFAULT_P_D, tau=DEFAULT_TAU,
            budget=args.budget, num_drones=args.num_drones,
            condition_key=condition,
        )
        T_common = int(base_frame.T_common.max())
        alpha_frames = []
        for alpha in alphas:
            q = mix_uniform_target(prior, mask, alpha)
            arm_rows = []
            for mode in ("online_static_3step", "dynamic_3step"):
                sim, _ = sims[mode]
                metrics = _path_exposure_metrics(
                    sim, sim.drone_paths, q, DEFAULT_P_D, T_common,
                )
                source = base_frame[base_frame.planning_mode == mode].iloc[0].to_dict()
                arm_rows.append({
                    **source,
                    "alpha": alpha,
                    "P_detect": metrics["P_detect"],
                    "RMST": metrics["RMST"],
                    "L": _path_exposure_metrics(
                        sim, sim.drone_paths, prior, DEFAULT_P_D, T_common,
                    )["L"],
                    "exposure_multiplicity": metrics["exposure_multiplicity"],
                    "revisit_fraction": metrics["revisit_fraction"],
                    "trajectory_reused": True,
                    "target_distribution_hash": stable_hash(q.tolist()),
                    "valid_support_cells": int(mask.sum()),
                    "zero_prior_cells_inside_domain": zero_prior,
                    "evaluation_distribution": "Q_alpha",
                    "evaluation_prior_source": "planner=P; endpoint=Q_alpha",
                    "effective_w": float(config["w"]),
                    "effective_p_d": float(config["p_d"]),
                    "effective_tau": float(config["tau"]),
                    "alpha_condition_key": _json_text({"alpha": alpha, **condition}),
                })
            alpha_frames.append(_c2_alpha_frame(arm_rows, alpha))
        return dataset, pd.concat(alpha_frames, ignore_index=True), meta

    jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    completed = _parallel_jobs(args, jobs, one, "C2 mismatch")
    distribution_meta = {}
    run_frames = []
    for dataset, frame, meta in completed:
        distribution_meta[dataset] = meta
        run_frames.append(frame)

    runs = _with_provenance(pd.concat(run_frames, ignore_index=True), config)
    _validate_campaign_frame(
        runs,
        pair_keys=["dataset", "planning_seed", "condition_id"],
        expected_arms=["online_static_3step", "dynamic_3step"],
        expected_rows=config["expected_rows"],
    )
    contrasts = _paired_contrasts(
        runs,
        keys=["dataset", "planning_seed", "condition_id"],
        control="online_static_3step",
        method="dynamic_3step",
    )
    summary = _summary_table(contrasts, ["dataset", "condition_id"])
    curves = contrasts.groupby(
        ["dataset", "condition_id"], as_index=False,
    ).agg(
        mean_dPdet=("dPdet", "mean"),
        mean_dRMST_rel=("dRMST_rel", "mean"),
        mean_dL=("dL", "mean"),
        n=("dPdet", "size"),
    )
    _write_frame(runs, outdir, "c2_prior_mismatch_runs.csv")
    _write_frame(contrasts, outdir, "c2_paired_contrasts.csv")
    _write_frame(summary, outdir, "c2_mismatch_summary.csv")
    _write_frame(curves, outdir, "c2_degradation_curves.csv")
    (outdir / "c2_prior_support.json").write_text(
        json.dumps(distribution_meta, indent=2) + "\n",
    )

    statuses = {}
    for alpha in (0.25, 0.5):
        sub = summary[summary.condition_id == f"alpha_{alpha:g}"]
        statuses[alpha] = sub.set_index("dataset")["state"].to_dict()
    first_adverse = None
    for alpha in sorted(alphas):
        states = summary[summary.condition_id == f"alpha_{alpha:g}"]
        adverse = states[states.state == "WORSENING"]
        if not adverse.empty:
            first_adverse = {
                "alpha": alpha,
                "datasets": adverse.dataset.astype(int).tolist(),
            }
            break
    if any(
        any(state == "WORSENING" for state in values.values())
        for values in statuses.values()
    ):
        gate = "INVERTS"
    elif all(
        sum(state == "IMPROVEMENT" for state in values.values())
        >= math.ceil(2 * len(datasets) / 3)
        for values in statuses.values()
    ):
        gate = "ROBUST_TO_MODERATE_MISMATCH"
    else:
        gate = "DEGRADES"
    (outdir / "c2_gate.md").write_text(
        "# C2 gate\n\n"
        + json.dumps({
            "gate": gate,
            "states": statuses,
            "first_adverse_alpha": first_adverse,
            "classification": "classify_paired_endpoints",
            "L_is_diagnostic": True,
        }, indent=2, default=_json_default)
        + "\n"
    )
    return config["experiment_id"]

def run_c2_lost_person(args):
    """Run the preregistered C2 follow-up only after the alpha=.5 gate allows it."""
    _require_nominal_parameters(args)
    parent_id, parent_config, parent_gate = _require_parent_campaign(args, "c2-mismatch", "c2_gate.md")
    if parent_config.get("scientific_valid") is not True:
        raise SystemExit("c2-lost-person requires a full scientific c2-mismatch parent")
    alpha_states = parent_gate.get("states", {}).get("0.5", parent_gate.get("states", {}).get(0.5, {}))
    if sum(state != "WORSENING" for state in alpha_states.values()) < math.ceil(2 * len(alpha_states) / 3):
        raise SystemExit("c2-lost-person not authorised: alpha=.5 has too many WORSENING datasets")
    config = _base_config(args, "c2-lost-person")
    datasets = sorted(args.datasets or parent_config.get("datasets") or C2_DATASETS)
    seeds = list(args.seeds_list or parent_config.get("seeds") or range(42, 52))
    full_design = bool(
        parent_config.get("scientific_valid") is True
        and _is_full_scientific_design(args, datasets, seeds, C2_DATASETS)
    )
    config.update({
        "datasets": datasets, "seeds": seeds,
        "scientific_valid": full_design,
        "label": "confirmatory" if full_design else "functional-only",
        "M": 100_000, "parent_experiment_id": parent_id,
        "parent_gate": parent_gate.get("gate"),
        "simulation_victim_model": "none", "evaluation_victim_model": "stationary",
        "victim_sampling_model": "lost_person_feature_generator",
        "condition_key": {"stage": "c2-lost-person", "w": NOMINAL_W, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
        "expected_rows": len(datasets) * len(seeds) * 2,
    })
    config, outdir = _make_campaign("c2-lost-person", config, args.jobs)
    distributions = {}
    metadata = {}
    rows = []
    contrasts = []
    for dataset in _progress(datasets, args, "C2 lost-person", total=len(datasets)):
        item = _load_item(dataset, args.size)
        probe = _build_sim(item, planning_mode="online_static_3step", w=NOMINAL_W, p_d=DEFAULT_P_D, tau=DEFAULT_TAU, budget=args.budget, num_drones=args.num_drones, planning_seed=42, implementation_version=config["implementation_version"], config=config)
        q, diagnostics = target_distribution_from_lost_person(item, config["M"], _seed_bundle(dataset, args.base_seed, config["condition_key"])["distribution_seed"], probe.heatmap_shape, probe.bounds)
        if diagnostics["valid_fraction"] < 0.99:
            raise SystemExit(f"lost-person target invalid for dataset {dataset}: {diagnostics}")
        distributions[dataset] = q
        metadata[dataset] = diagnostics
        for seed in seeds:
            condition = config["condition_key"]
            frame, sims, _ = _run_pair(item, config, dataset_id=dataset, planning_seed=seed, w=NOMINAL_W, p_d=DEFAULT_P_D, tau=DEFAULT_TAU, budget=args.budget, num_drones=args.num_drones, condition_key=condition)
            T_common = int(frame.T_common.max())
            arm_rows = []
            for mode in ("online_static_3step", "dynamic_3step"):
                sim, _ = sims[mode]
                q_metrics = _path_exposure_metrics(sim, sim.drone_paths, q, DEFAULT_P_D, T_common)
                p_metrics = _path_exposure_metrics(sim, sim.drone_paths, item.heatmap, DEFAULT_P_D, T_common)
                source = frame[frame.planning_mode == mode].iloc[0].to_dict()
                arm_rows.append({
                    **source, "P_detect": q_metrics["P_detect"], "RMST": q_metrics["RMST"], "L": p_metrics["L"],
                    "exposure_multiplicity": q_metrics["exposure_multiplicity"], "revisit_fraction": q_metrics["revisit_fraction"],
                    "condition_id": "lost_person", "trajectory_reused": True,
                    "target_distribution_hash": diagnostics["target_distribution_hash"],
                    "distribution_seed": diagnostics["distribution_seed"],
                    "target_valid_samples": diagnostics["valid_samples"],
                })
            run_frame = pd.DataFrame(arm_rows)
            rows.append(run_frame)
            contrasts.append(_paired_contrasts(run_frame, keys=["dataset", "planning_seed", "condition_id"], control="online_static_3step", method="dynamic_3step"))
    runs = _with_provenance(pd.concat(rows, ignore_index=True), config)
    _validate_campaign_frame(runs, pair_keys=["dataset", "planning_seed", "condition_id"], expected_arms=["online_static_3step", "dynamic_3step"], expected_rows=config["expected_rows"])
    paired = pd.concat(contrasts, ignore_index=True)
    summary = _summary_table(paired, ["dataset", "condition_id"])
    _save_distribution_npz(outdir, distributions, metadata)
    _write_frame(runs, outdir, "c2_lost_person_runs.csv")
    _write_frame(paired, outdir, "c2_lost_person_paired_contrasts.csv")
    _write_frame(summary, outdir, "c2_lost_person_summary.csv")
    (outdir / "c2_lost_person_gate.md").write_text("# C2 lost-person gate\n\n" + json.dumps({"parent": args.input_experiment_id, "classification": "classify_paired_endpoints", "states": summary.set_index("dataset").state.to_dict()}, indent=2, default=_json_default) + "\n")
    return config["experiment_id"]


def _mobility_overlay(
    sim: SARSimulation,
    paths: list[list[tuple[float, float]]],
    model_name: str,
    victim_seed: int,
    sensor_seed: int,
    p_d: float,
    T_common: int,
    victim_speed: float = DEFAULT_VICTIM_SPEED,
    dt: float = DEFAULT_DT,
):
    prior = _normalise_prior(sim.dataset.heatmap)
    cell = stationary_victim_cell(prior.ravel(), victim_seed)
    row, col = divmod(int(cell), sim.heatmap_shape[1])
    position = Point(*sim._grid_to_world(row, col))
    model = VICTIM_MODELS[model_name](speed=victim_speed, seed=victim_seed)
    victim_positions = [position]
    for _ in range(T_common):
        position = model.move(position, dt, sim.dataset.features, sim.bounds)
        victim_positions.append(position)
    q = 1.0 - p_d
    survival = 1.0
    rmst = 0.0
    uniform = np.random.default_rng(sensor_seed).random((T_common + 1, len(paths)))
    detected_t = None
    for timestep in range(T_common + 1):
        vx, vy = victim_positions[timestep].x, victim_positions[timestep].y
        hazard_count = 0
        for drone_id, path in enumerate(paths):
            if timestep < len(path) and np.hypot(path[timestep][0] - vx, path[timestep][1] - vy) <= sim.detection_radius:
                hazard_count += 1
        hazard = 1.0 - q**hazard_count
        survival *= 1.0 - hazard
        rmst += survival
        if detected_t is None:
            for drone_id, path in enumerate(paths):
                if timestep < len(path) and np.hypot(path[timestep][0] - vx, path[timestep][1] - vy) <= sim.detection_radius and uniform[timestep, drone_id] < p_d:
                    detected_t = timestep
                    break
    realized = 1.0 if detected_t is not None else 0.0
    return {
        "P_detect": float(1.0 - survival), "RMST": float(rmst),
        "P_detect_realized": realized, "T_detect_realized": detected_t if detected_t is not None else T_common + 1,
        "victim_trajectory_hash": stable_hash([(point.x, point.y) for point in victim_positions]),
        "sensor_uniforms_hash": stable_hash(uniform.tolist()), "victim_start_cell": int(cell),
        "exposure_multiplicity": float(sum(1 for path in paths for t in range(min(len(path), T_common + 1)) if np.hypot(path[t][0] - victim_positions[t].x, path[t][1] - victim_positions[t].y) <= sim.detection_radius)),
    }


def run_c3_mobility(args):
    _require_nominal_parameters(args)
    parent_id, parent_config, parent_gate = _require_parent_campaign(
        args, "c2-mismatch", "c2_gate.md",
    )
    config = _base_config(args, "c3-mobility")
    datasets = sorted(args.datasets or C3_DATASETS)
    seeds = list(args.seeds_list or range(42, 52))
    models = ["random_walk", "route_following", "lost_person"]
    full_design = _is_full_scientific_design(
        args, datasets, seeds, C3_DATASETS,
    )
    config.update({
        "parent_experiment_id": parent_id,
        "parent_git_commit": parent_config.get("git_commit"),
        "parent_source_hash": parent_config.get("source_hash"),
        "parent_compatibility_status": parent_config.get("_compatibility_status"),
        "parent_gate": parent_gate.get("gate"),
        "scientific_valid": full_design,
        "datasets": datasets,
        "seeds": seeds,
        "mobility_models": models,
        "label": "confirmatory" if full_design else "functional-only",
        "simulation_victim_model": "none",
        "evaluation_victim_model": models,
        "victim_sampling_model": "stationary_victim_cell(P,victim_seed)",
        "paths_reused_by_model": True,
        "effective_victim_speed": float(args.victim_speed),
        "effective_dt": float(args.dt),
        "parallel_unit": ["dataset", "planning_seed"],
        "condition_key": {
            "stage": "c3-mobility", "w": NOMINAL_W,
            "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU,
        },
        "expected_rows": len(datasets) * len(models) * len(seeds) * 2,
    })
    config, outdir = _make_campaign("c3-mobility", config, args.jobs)

    def one(dataset, seed):
        item = _load_item(dataset, args.size)
        condition = {
            "stage": "c3-mobility", "w": NOMINAL_W,
            "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU,
        }
        bundle = _seed_bundle(dataset, seed, condition)
        frame, sims, _ = _run_pair(
            item, config, dataset_id=dataset, planning_seed=seed,
            w=NOMINAL_W, p_d=DEFAULT_P_D, tau=DEFAULT_TAU,
            budget=args.budget, num_drones=args.num_drones,
            condition_key=condition,
        )
        T_common = int(frame.T_common.max())
        rows = []
        for model in models:
            for mode in ("online_static_3step", "dynamic_3step"):
                sim, _ = sims[mode]
                metrics = _mobility_overlay(
                    sim,
                    sim.drone_paths,
                    model,
                    bundle["victim_seed"],
                    bundle["sensor_seed"],
                    float(config["p_d"]),
                    T_common,
                    float(config["victim_speed"]),
                    float(config["dt"]),
                )
                source = frame[frame.planning_mode == mode].iloc[0].to_dict()
                rows.append({
                    **source,
                    "mobility_model": model,
                    "P_detect": metrics["P_detect"],
                    "RMST": metrics["RMST"],
                    "L": source["L"],
                    "P_detect_realized": metrics["P_detect_realized"],
                    "T_detect_realized": metrics["T_detect_realized"],
                    "victim_seed": bundle["victim_seed"],
                    "sensor_seed": bundle["sensor_seed"],
                    "victim_trajectory_hash": metrics["victim_trajectory_hash"],
                    "sensor_uniforms_hash": metrics["sensor_uniforms_hash"],
                    "victim_start_cell": metrics["victim_start_cell"],
                    "exposure_multiplicity": metrics["exposure_multiplicity"],
                    "effective_victim_speed": float(config["victim_speed"]),
                    "effective_dt": float(config["dt"]),
                    "effective_p_d": float(config["p_d"]),
                    "condition_id": model,
                    "evaluation_victim_model": model,
                    "victim_sampling_model": "stationary_victim_cell(P,victim_seed)",
                })
        return pd.DataFrame(rows)

    jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    run_frames = _parallel_jobs(args, jobs, one, "C3 mobility")
    runs = _with_provenance(pd.concat(run_frames, ignore_index=True), config)
    _validate_campaign_frame(
        runs,
        pair_keys=["dataset", "mobility_model", "planning_seed"],
        expected_arms=["online_static_3step", "dynamic_3step"],
        expected_rows=config["expected_rows"],
    )
    contrasts = _paired_contrasts(
        runs,
        keys=["dataset", "mobility_model", "planning_seed"],
        control="online_static_3step",
        method="dynamic_3step",
    )
    summary = _summary_table(contrasts, ["dataset", "mobility_model"])
    _write_frame(runs, outdir, "c3_mobility_runs.csv")
    _write_frame(contrasts, outdir, "c3_paired_contrasts.csv")
    _write_frame(summary, outdir, "c3_mobility_summary.csv")

    model_status = {}
    for model, group in summary.groupby("mobility_model"):
        non_worsening = (group.state != "WORSENING").all()
        model_status[model] = {
            "pass": bool(
                non_worsening
                and (group.state == "IMPROVEMENT").sum()
                >= math.ceil(2 * len(datasets) / 3)
            ),
            "states": group.set_index("dataset").state.to_dict(),
        }
    passing = sum(value["pass"] for value in model_status.values())
    any_bad = any(
        (summary[summary.mobility_model == model].state == "WORSENING").sum()
        >= math.ceil(2 * len(datasets) / 3)
        for model in models
    )
    trajectory_ok = bool(
        runs.groupby(
            ["dataset", "mobility_model", "planning_seed"],
        )["victim_trajectory_hash"].nunique().max() == 1
    )
    sensor_crn_ok = bool(
        runs.groupby(
            ["dataset", "mobility_model", "planning_seed"],
        )["sensor_uniforms_hash"].nunique().max() == 1
    )
    if not trajectory_ok or not sensor_crn_ok:
        gate = "INVALID"
    elif any_bad:
        gate = "FAIL"
    elif passing >= math.ceil(2 * len(models) / 3):
        gate = "PASS"
    else:
        gate = "BOUNDARY"
    (outdir / "c3_gate.md").write_text(
        "# C3 gate\n\n"
        + json.dumps({
            "gate": gate,
            "model_status": model_status,
            "trajectory_hashes_reproducible": trajectory_ok,
            "sensor_crn_reproducible": sensor_crn_ok,
            "classification": "classify_paired_endpoints",
        }, indent=2, default=_json_default)
        + "\n"
    )
    return config["experiment_id"]

def run_c3_controls(args):
    """Non-confirmatory w=0/1 controls with one path pair per w and seed."""
    _require_nominal_parameters(args)
    parent_id, parent_config, parent_gate = _require_parent_campaign(args, "c2-mismatch", "c2_gate.md")
    config = _base_config(args, "c3-controls")
    datasets = sorted(args.datasets or C3_DATASETS)
    seeds = list(args.seeds_list or range(42, 45))
    models = ["random_walk", "route_following", "lost_person"]
    weights = [0.0, 1.0]
    config.update({
        "parent_experiment_id": parent_id,
        "parent_gate": parent_gate.get("gate"),
        "datasets": datasets, "seeds": seeds, "mobility_models": models, "w": weights,
        "label": "sensitivity", "simulation_victim_model": "none",
        "evaluation_victim_model": models, "paths_reused_by_model": True,
        "effective_victim_speed": float(args.victim_speed), "effective_dt": float(args.dt),
        "condition_key": {"stage": "c3-controls", "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
        "expected_rows": len(datasets) * len(seeds) * len(weights) * len(models) * 2,
    })
    config, outdir = _make_campaign("c3-controls", config, args.jobs)
    rows = []
    for dataset in _progress(datasets, args, "C3 controls", total=len(datasets)):
        item = _load_item(dataset, args.size)
        for seed in seeds:
            for w in weights:
                condition = {"stage": "c3-controls", "w": w, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU}
                bundle = _seed_bundle(dataset, seed, condition)
                frame, sims, _ = _run_pair(item, config, dataset_id=dataset, planning_seed=seed, w=w, p_d=DEFAULT_P_D, tau=DEFAULT_TAU, budget=args.budget, num_drones=args.num_drones, condition_key=condition)
                T_common = int(frame.T_common.max())
                for model in models:
                    for mode in ("online_static_3step", "dynamic_3step"):
                        sim, _ = sims[mode]
                        metrics = _mobility_overlay(
                            sim, sim.drone_paths, model, bundle["victim_seed"], bundle["sensor_seed"],
                            float(config["p_d"]), T_common, float(config["victim_speed"]), float(config["dt"]),
                        )
                        source = frame[frame.planning_mode == mode].iloc[0].to_dict()
                        rows.append({
                            **source, "w": w, "mobility_model": model, "condition_id": f"{model}_w{w:g}",
                            "P_detect": metrics["P_detect"], "RMST": metrics["RMST"],
                            "P_detect_realized": metrics["P_detect_realized"], "T_detect_realized": metrics["T_detect_realized"],
                            "victim_seed": bundle["victim_seed"], "sensor_seed": bundle["sensor_seed"],
                            "victim_trajectory_hash": metrics["victim_trajectory_hash"], "sensor_uniforms_hash": metrics["sensor_uniforms_hash"],
                            "effective_victim_speed": float(config["victim_speed"]), "effective_dt": float(config["dt"]),
                            "effective_p_d": float(config["p_d"]),
                            "evaluation_victim_model": model,
                        })
    runs = _with_provenance(pd.DataFrame(rows), config)
    _validate_campaign_frame(runs, pair_keys=["dataset", "mobility_model", "planning_seed", "w"], expected_arms=["online_static_3step", "dynamic_3step"], expected_rows=config["expected_rows"])
    contrasts = _paired_contrasts(runs, keys=["dataset", "mobility_model", "planning_seed", "w"], control="online_static_3step", method="dynamic_3step")
    _write_frame(runs, outdir, "c3_controls_runs.csv")
    _write_frame(contrasts, outdir, "c3_controls_paired_contrasts.csv")
    _write_frame(_summary_table(contrasts, ["dataset", "mobility_model", "w"]), outdir, "c3_controls_summary.csv")
    return config["experiment_id"]


def _assign_pizza_paths(survivor_positions: list[tuple[float, float]], pizza_paths: list[LineString]) -> tuple[list[LineString], dict[int, int]]:
    """Assign paths to survivor IDs by deterministic minimum transit cost."""
    if len(pizza_paths) != len(survivor_positions):
        raise ValueError("pizza path count must equal survivor count")
    first_points = [tuple(path.coords[0]) for path in pizza_paths if not path.is_empty and len(path.coords)]
    common = np.asarray(first_points[0] if first_points else (0.0, 0.0), dtype=float)
    targets = []
    for path in pizza_paths:
        target = None
        for point in path.coords:
            if np.hypot(point[0] - common[0], point[1] - common[1]) > 1e-9:
                target = tuple(point)
                break
        targets.append(target or tuple(common))
    costs = np.asarray([[np.hypot(position[0] - target[0], position[1] - target[1]) for target in targets] for position in survivor_positions], dtype=float)
    best = None
    for assignment in itertools.permutations(range(len(pizza_paths))):
        cost = float(sum(costs[drone_id, path_id] for drone_id, path_id in enumerate(assignment)))
        candidate = (cost, tuple(assignment))
        if best is None or candidate < best:
            best = candidate
    _, assignment = best
    mapping = {int(drone_id): int(path_id) for drone_id, path_id in enumerate(assignment)}
    ordered = [pizza_paths[mapping[drone_id]] for drone_id in range(len(survivor_positions))]
    return ordered, mapping


def _termination_reason(sim: SARSimulation, budget_total: float) -> str:
    tolerance = 1e-9 * max(1.0, float(budget_total))
    if sim.total_distance >= float(budget_total) - tolerance:
        return "budget_exhausted"
    if sim._base_mode in ("static", "static_nstep", "pizza_replan", "pizza_fixed") and sim.static_paths:
        exhausted = all(
            progress >= path.length - 1e-9
            for i, (progress, path) in enumerate(zip(sim._path_progress, sim.static_paths))
            if sim._is_active(i)
        )
        if exhausted:
            return "path_exhausted"
    return "completed"


def _aligned_positions(sim: SARSimulation, positions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [sim._grid_to_world(*sim._world_to_grid(x, y)) for x, y in positions]


def _pizza_paths(item, positions: list[tuple[float, float]], num_drones: int) -> tuple[list[LineString], dict[int, int]]:
    cx = (item.bounds[0] + item.bounds[2]) / 2.0
    cy = (item.bounds[1] + item.bounds[3]) / 2.0
    paths = generate_pizza_zigzag_path(
        center_x=cx, center_y=cy, max_radius=item.radius_km * 1000.0, num_drones=num_drones,
        fov_deg=DEFAULT_FOV_DEG, altitude=DEFAULT_ALTITUDE, overlap=0.0,
        path_point_spacing_m=10.0, border_gap_m=0.0,
    )
    ordered, assignment = _assign_pizza_paths(positions, paths)
    transit_paths = []
    for position, path in zip(positions, ordered):
        coords = list(path.coords) if not path.is_empty else []
        if not coords:
            coords = [position, position]
        elif np.hypot(coords[0][0] - position[0], coords[0][1] - position[1]) > 1e-9:
            coords.insert(0, position)
        elif len(coords) == 1:
            coords.append(coords[0])
        transit_paths.append(LineString(coords))
    return transit_paths, assignment


def _run_attrition_arm(item, config, dataset, seed, k, mode, initial_positions, failure_ids, failure_seed, args):
    active_ids = [index for index in range(DEFAULT_NUM_DRONES) if index not in failure_ids[:k]]
    survivor_positions = [initial_positions[index] for index in active_ids]
    budget_total = len(active_ids) * 40_000.0
    static_assignment = "none"
    assignment_hash = "none"
    # Probe and executing simulator share exactly the same relevant arguments
    # (item, w, p_d, tau, budget, fleet, seed, version). The probe is used only
    # to grid-align survivor positions; its planning mode is irrelevant here and
    # must NOT require static_paths: a "static" probe without them is rejected
    # by SARSimulation (planning_mode='static' requires static_paths).
    common_kwargs = dict(
        item=item,
        w=NOMINAL_W,
        p_d=DEFAULT_P_D,
        tau=DEFAULT_TAU,
        budget=budget_total,
        num_drones=len(active_ids),
        planning_seed=seed,
        implementation_version=config["implementation_version"],
        config=config,
    )
    probe = _build_sim(
        planning_mode="online_static_3step",
        **common_kwargs,
    )
    executed_positions = _aligned_positions(probe, survivor_positions)
    if mode == "pizza_repartition":
        paths, assignment = _pizza_paths(item, executed_positions, len(active_ids))
        static_assignment = _json_text({str(active_ids[key]): value for key, value in assignment.items()})
        assignment_hash = stable_hash(static_assignment)
        sim = _build_sim(
            planning_mode="static",
            static_paths=paths,
            **common_kwargs,
        )
    else:
        sim = _build_sim(
            planning_mode=("dynamic_3step" if mode == "dynamic_evidence_3step" else mode),
            **common_kwargs,
        )
    result = _run_from_positions(sim, executed_positions, args.dt)
    T = max(len(path) for path in sim.drone_paths) - 1
    metrics = _path_exposure_metrics(sim, sim.drone_paths, item.heatmap, DEFAULT_P_D, T)
    max_step = max(
        (
            np.hypot(path[index + 1][0] - path[index][0], path[index + 1][1] - path[index][1])
            for path in sim.drone_paths
            for index in range(len(path) - 1)
        ),
        default=0.0,
    )
    return {
        "dataset": dataset, "planning_seed": seed, "k": k, "planning_mode": mode,
        "P_detect": metrics["P_detect"], "RMST": metrics["RMST"], "L": metrics["L"],
        "NRMST": metrics["RMST"] / (T + 1), "T": T, "H": T + 1, "T_common": T, "H_common": T + 1,
        "actual_distance_total": float(result.total_distance), "budget_total": budget_total,
        "budget_utilization": float(result.total_distance / budget_total) if budget_total else float("nan"),
        "termination_reason": _termination_reason(sim, budget_total),
        "exposure_multiplicity": metrics["exposure_multiplicity"], "revisit_fraction": metrics["revisit_fraction"],
        "planner_wall_time": float(result.planner_wall_time), "number_of_replans": int(result.number_of_replans),
        "active_fleet": len(active_ids), "failed_drone_ids": _json_text(failure_ids[:k]),
        "failure_seed": int(failure_seed), "victim_seed": "none", "sensor_seed": "none", "distribution_seed": "none",
        "survivor_ids": _json_text(active_ids), "survivor_positions": _json_text(survivor_positions),
        "initial_positions": _json_text(executed_positions),
        "initial_positions_hash": stable_hash(executed_positions),
        "pizza_assignment": static_assignment, "pizza_assignment_hash": assignment_hash,
        "max_step_distance": float(max_step),
        "retention_Pdet": "none", "retention_invalid_reason": "none",
        "degradation_NRMST": "none",
        "simulation_victim_model": "none", "evaluation_victim_model": "stationary",
        "victim_sampling_model": "matched_prior_exact", "evaluation_num_victims": 1, "simulation_num_victims": 0,
        "_sim": sim,
    }


def _c4_degradation_contrasts(runs: pd.DataFrame, comparator: str) -> pd.DataFrame:
    rows = []
    for (dataset, seed, k), group in runs[runs.k > 0].groupby(["dataset", "planning_seed", "k"], sort=True):
        by_mode = {row.planning_mode: row for row in group.itertuples(index=False)}
        dyn = by_mode["dynamic_evidence_3step"]
        comp = by_mode[comparator]
        if comparator == "online_static_3step":
            prefix = "Dyn_OS"
        else:
            prefix = "Dyn_Pizza"
        dyn_ret = float(dyn.retention_Pdet) if dyn.retention_Pdet != "none" else float("nan")
        comp_ret = float(comp.retention_Pdet) if comp.retention_Pdet != "none" else float("nan")
        d_ret = dyn_ret - comp_ret if np.isfinite(dyn_ret) and np.isfinite(comp_ret) else float("nan")
        d_deg = comp.degradation_NRMST - dyn.degradation_NRMST
        rows.append({
            "dataset": dataset, "planning_seed": seed, "k": k, "comparator": comparator,
            "dRetention_Pdet_" + prefix: d_ret,
            "dDegradation_NRMST_" + prefix: d_deg,
            "dPdet": dyn.P_detect - comp.P_detect,
            "dRMST_rel": (comp.RMST - dyn.RMST) / max(dyn.H_common, comp.H_common),
        })
    return pd.DataFrame(rows)

def _c4_pairwise_degradation_contrasts(runs: pd.DataFrame, first: str, second: str) -> pd.DataFrame:
    """Compare two attrition arms using retention/degradation endpoints."""
    rows = []
    for (dataset, seed, k), group in runs[runs.k > 0].groupby(["dataset", "planning_seed", "k"], sort=True):
        by_mode = {row.planning_mode: row for row in group.itertuples(index=False)}
        left = by_mode[first]
        right = by_mode[second]
        left_ret = float(left.retention_Pdet) if left.retention_Pdet != "none" else float("nan")
        right_ret = float(right.retention_Pdet) if right.retention_Pdet != "none" else float("nan")
        rows.append({
            "dataset": dataset,
            "planning_seed": seed,
            "k": k,
            "comparator": f"{first}_vs_{second}",
            "dRetention_Pdet": left_ret - right_ret if np.isfinite(left_ret) and np.isfinite(right_ret) else float("nan"),
            "dDegradation_NRMST": right.degradation_NRMST - left.degradation_NRMST,
            "dPdet": left.P_detect - right.P_detect,
            "dRMST_rel": (right.RMST - left.RMST) / max(left.H_common, right.H_common),
        })
    return pd.DataFrame(rows)

def _c4_degradation_finiteness(contrasts: pd.DataFrame) -> dict[str, bool]:
    result = {}
    for comparator, columns in {
        "online_static_3step": ("dRetention_Pdet_Dyn_OS", "dDegradation_NRMST_Dyn_OS"),
        "pizza_repartition": ("dRetention_Pdet_Dyn_Pizza", "dDegradation_NRMST_Dyn_Pizza"),
    }.items():
        subset = contrasts[contrasts["comparator"] == comparator]
        result[comparator] = bool(np.isfinite(subset[list(columns)].to_numpy(dtype=float)).all())
    return result


def run_c4_attrition(args):
    _require_nominal_parameters(args)
    parent_id, parent_config, parent_gate = _require_parent_campaign(args, "c3-mobility", "c3_gate.md")
    config = _base_config(args, "c4-attrition")
    datasets = sorted(args.datasets or C4_DATASETS)
    seeds = list(args.seeds_list or range(42, 52))
    modes = ["dynamic_evidence_3step", "online_static_3step", "pizza_repartition"]
    full_design = _is_full_scientific_design(
        args, datasets, seeds, C4_DATASETS,
    )
    config.update({
        "parent_experiment_id": parent_id,
        "parent_git_commit": parent_config.get("git_commit"),
        "parent_source_hash": parent_config.get("source_hash"),
        "parent_compatibility_status": parent_config.get("_compatibility_status"),
        "parent_gate": parent_gate.get("gate"),
        "scientific_valid": full_design,
        "datasets": datasets, "seeds": seeds, "attrition_levels": [0, 1, 2, 3], "fleet_nominal": 5,
        "nominal_budget_per_active_uav": 40_000.0, "planners": modes,
        "label": "confirmatory" if full_design else "functional-only",
        "simulation_victim_model": "none", "evaluation_victim_model": "stationary", "victim_sampling_model": "matched_prior_exact",
        "effective_w": float(args.w), "effective_p_d": float(args.p_d), "effective_tau": float(args.tau),
        "parallel_unit": ["dataset", "planning_seed"],
        "condition_key": {"stage": "c4-attrition", "w": NOMINAL_W, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
        "expected_rows": len(datasets) * len(seeds) * 4 * 3,
    })
    config, outdir = _make_campaign("c4-attrition", config, args.jobs)
    def one(dataset, seed):
        item = _load_item(dataset, args.size)
        initial = _initial_positions(item, DEFAULT_NUM_DRONES, seed)
        condition = {
            "stage": "c4-attrition",
            "dataset": dataset,
            "planning_seed": seed,
        }
        failure_seed = _seed_bundle(dataset, seed, condition)["failure_seed"]
        failure_ids = list(
            np.random.default_rng(failure_seed).permutation(DEFAULT_NUM_DRONES),
        )
        mission_rows = []
        for k in range(4):
            mission_rows.extend(
                _run_attrition_arm(
                    item, config, dataset, seed, k, mode, initial,
                    failure_ids, failure_seed, args,
                )
                for mode in modes
            )
        mission_T_common = max(
            max(len(path) for path in row["_sim"].drone_paths) - 1
            for row in mission_rows
        )
        for row in mission_rows:
            sim = row.pop("_sim")
            metrics = _path_exposure_metrics(
                sim,
                sim.drone_paths,
                item.heatmap,
                DEFAULT_P_D,
                mission_T_common,
            )
            row.update({
                "P_detect": metrics["P_detect"],
                "RMST": metrics["RMST"],
                "L": metrics["L"],
                "NRMST": metrics["RMST"] / (mission_T_common + 1),
                "T_common": mission_T_common,
                "H_common": mission_T_common + 1,
                "exposure_multiplicity": metrics["exposure_multiplicity"],
                "revisit_fraction": metrics["revisit_fraction"],
            })
        by_mode = {
            (row["k"], row["planning_mode"]): row for row in mission_rows
        }
        for mode in modes:
            baseline = by_mode[(0, mode)]
            for row in (
                item for item in mission_rows
                if item["planning_mode"] == mode
            ):
                if baseline["P_detect"] > 0.0:
                    row["retention_Pdet"] = (
                        row["P_detect"] / baseline["P_detect"]
                    )
                    row["retention_invalid_reason"] = "none"
                else:
                    row["retention_Pdet"] = "none"
                    row["retention_invalid_reason"] = "baseline_P_detect_zero"
                row["degradation_NRMST"] = (
                    row["NRMST"] - baseline["NRMST"]
                )
        return pd.DataFrame(mission_rows)

    jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    run_frames = _parallel_jobs(args, jobs, one, "C4 attrition")
    runs = _with_provenance(pd.concat(run_frames, ignore_index=True), config)
    _validate_campaign_frame(runs, expected_rows=config["expected_rows"])
    contrasts = []
    for comparator in ("online_static_3step", "pizza_repartition"):
        part = _c4_degradation_contrasts(runs, comparator)
        if not part.empty:
            contrasts.append(part)
    if not contrasts:
        raise ValueError("C4 produced no degradation contrasts")
    contrasts = pd.concat(contrasts, ignore_index=True)
    # Rename the comparison columns into the six stable fields requested by C4.
    for source, target in (("dRetention_Pdet_Dyn_OS", "dRetention_Pdet_Dyn_OS"), ("dDegradation_NRMST_Dyn_OS", "dDegradation_NRMST_Dyn_OS"), ("dRetention_Pdet_Dyn_Pizza", "dRetention_Pdet_Dyn_Pizza"), ("dDegradation_NRMST_Dyn_Pizza", "dDegradation_NRMST_Dyn_Pizza")):
        if source not in contrasts:
            contrasts[source] = np.nan
    summary_rows = []
    for (dataset, k, comparator), group in contrasts.groupby(["dataset", "k", "comparator"], sort=True):
        if comparator == "online_static_3step":
            p_col, r_col = "dRetention_Pdet_Dyn_OS", "dDegradation_NRMST_Dyn_OS"
        else:
            p_col, r_col = "dRetention_Pdet_Dyn_Pizza", "dDegradation_NRMST_Dyn_Pizza"
        p = summarize_paired(group[p_col], MARGIN_PDET)
        r = summarize_paired(group[r_col], MARGIN_RMST_REL)
        summary_rows.append({"dataset": dataset, "k": k, "comparator": comparator, "n": len(group), "dRetention_mean": p["mean"], "dRetention_ci_lo": p["ci_lo"], "dRetention_ci_hi": p["ci_hi"], "dDegradation_mean": r["mean"], "dDegradation_ci_lo": r["ci_lo"], "dDegradation_ci_hi": r["ci_hi"], "state": classify_paired_endpoints(p, r)})
    summary = pd.DataFrame(summary_rows)
    online_pizza_contrasts = _c4_pairwise_degradation_contrasts(
        runs, "online_static_3step", "pizza_repartition"
    )
    online_pizza_summary_rows = []
    for (dataset, k), group in online_pizza_contrasts.groupby(["dataset", "k"], sort=True):
        p = summarize_paired(group["dRetention_Pdet"], MARGIN_PDET)
        r = summarize_paired(group["dDegradation_NRMST"], MARGIN_RMST_REL)
        online_pizza_summary_rows.append({
            "dataset": dataset,
            "k": k,
            "n": len(group),
            "dRetention_mean": p["mean"],
            "dDegradation_mean": r["mean"],
            "state": classify_paired_endpoints(p, r),
        })
    online_pizza_summary = pd.DataFrame(online_pizza_summary_rows)
    _write_frame(online_pizza_contrasts, outdir, "c4_online_pizza_contrasts.csv")
    _write_frame(online_pizza_summary, outdir, "c4_online_pizza_summary.csv")
    _write_frame(runs, outdir, "c4_attrition_runs.csv")
    _write_frame(contrasts, outdir, "c4_paired_contrasts.csv")
    _write_frame(summary, outdir, "c4_attrition_summary.csv")
    retention_columns = [
        "dataset", "planning_seed", "k", "planning_mode", "P_detect", "RMST",
        "NRMST", "retention_Pdet", "retention_invalid_reason",
        "degradation_NRMST", "T", "H", "T_common", "H_common",
        "budget_total", "actual_distance_total", "budget_utilization",
        "termination_reason", "active_fleet", "failed_drone_ids",
        "pizza_assignment_hash",
    ]
    _write_frame(runs[retention_columns], outdir, "c4_retention.csv")
    fairness_rows = []
    for (dataset, seed, k), group in runs.groupby(["dataset", "planning_seed", "k"], sort=True):
        fairness_rows.append({
            "dataset": dataset,
            "planning_seed": seed,
            "k": k,
            "failure_identity_equal": group["failed_drone_ids"].nunique() == 1,
            "survivor_positions_equal": group["initial_positions_hash"].nunique() == 1,
            "budget_total_equal": group["budget_total"].nunique() == 1,
            "active_fleet_equal": group["active_fleet"].nunique() == 1,
            "temporal_resolution_equal": group["H_common"].nunique() == 1,
            "max_step_distance": float(group["max_step_distance"].max()),
            "step_bound": float(config["drone_speed"] * args.dt),
            "step_bound_ok": bool((group["max_step_distance"] <= config["drone_speed"] * args.dt + 1e-6).all()),
            "invalid_path_exhaustion": bool(((group["termination_reason"] == "path_exhausted") & (group["budget_utilization"] < 0.99)).any()),
        })
    fairness = pd.DataFrame(fairness_rows)
    _write_frame(fairness, outdir, "c4_fairness_checks.csv")
    comparator_finite = _c4_degradation_finiteness(contrasts)
    degradation_finite = bool(all(comparator_finite.values()))
    online_pizza_finite = bool(np.isfinite(online_pizza_contrasts[["dRetention_Pdet", "dDegradation_NRMST"]].to_numpy(dtype=float)).all())
    fairness_valid = bool(
        fairness[[
            "failure_identity_equal", "survivor_positions_equal", "budget_total_equal",
            "active_fleet_equal", "temporal_resolution_equal", "step_bound_ok",
        ]].to_numpy(dtype=bool).all()
        and not fairness["invalid_path_exhaustion"].any()
        and degradation_finite
        and online_pizza_finite
    )
    (outdir / "c4_fairness_checks.md").write_text(
        "# C4 fairness checks\n\n"
        + json.dumps({
            "valid": fairness_valid,
            "degradation_finite": degradation_finite,
            "comparator_finite": comparator_finite,
            "online_pizza_finite": online_pizza_finite,
            "checks": fairness.to_dict(orient="records"),
        }, indent=2, default=_json_default)
        + "\n"
    )
    gate_checks = {}
    for comparator in ("online_static_3step", "pizza_repartition"):
        sub = summary[summary.comparator == comparator]
        states = sub[sub.k > 0].groupby("k")["state"].apply(list).to_dict()
        gate_checks[comparator] = states
    no_worsening_all = all("WORSENING" not in states for values in gate_checks.values() for states in values.values())
    dynamic_pizza_equiv = all(sum(state == "EQUIVALENT" for state in states) >= math.ceil(2 * len(datasets) / 3) for states in gate_checks.get("pizza_repartition", {}).values())
    online_pizza_states = online_pizza_summary[online_pizza_summary.k > 0].groupby("k")["state"].apply(list).to_dict()
    dynamic_online_equiv = all(
        sum(state == "EQUIVALENT" for state in gate_checks.get("online_static_3step", {}).get(k, [])) >= math.ceil(2 * len(datasets) / 3)
        for k in (1, 2, 3)
    )
    online_pizza_improved = all(
        sum(state == "IMPROVEMENT" for state in online_pizza_states.get(k, [])) >= math.ceil(2 * len(datasets) / 3)
        for k in (1, 2, 3)
    )
    online_main = dynamic_online_equiv and online_pizza_improved
    dynamic_online_nonworsening = all(
        sum(state in {"IMPROVEMENT", "EQUIVALENT"} for state in gate_checks.get("online_static_3step", {}).get(k, [])) >= math.ceil(2 * len(datasets) / 3)
        for k in (1, 2, 3)
    )
    dynamic_pizza_nonworsening = all(
        sum(state in {"IMPROVEMENT", "EQUIVALENT"} for state in gate_checks.get("pizza_repartition", {}).get(k, [])) >= math.ceil(2 * len(datasets) / 3)
        for k in (1, 2, 3)
    )
    if not fairness_valid:
        gate = "INVALID"
    elif dynamic_pizza_equiv:
        gate = "PIZZA_GAP_CLOSED"
    elif online_main:
        gate = "ONLINE_REPLANNING_MAIN"
    elif no_worsening_all and dynamic_online_nonworsening and dynamic_pizza_nonworsening:
        gate = "GRACEFUL_REPLANNING"
    else:
        gate = "NO_DIFFERENTIAL_ADVANTAGE"
    (outdir / "c4_gate.md").write_text(
        "# C4 gate\n\n"
        + json.dumps({
            "gate": gate,
            "checks": gate_checks,
            "online_pizza_states": online_pizza_states,
            "dynamic_online_equivalent": dynamic_online_equiv,
            "online_pizza_improved": online_pizza_improved,
            "no_worsening_all": no_worsening_all,
            "fairness_valid": fairness_valid,
            "absolute_endpoints_secondary": True,
            "wording_prohibition": "do not call this fault tolerance",
        }, indent=2, default=_json_default)
        + "\n"
    )
    return config["experiment_id"]


def _classify_c5_gate(
    *,
    scientific_valid: bool,
    global_valid: bool,
    non_adverse_count: int,
    practically_positive_count: int,
    lower_ok: bool,
    global_means,
    global_classifications,
) -> dict:
    means = np.asarray(global_means, dtype=float)
    classifications = list(global_classifications)
    global_mean_positive = bool(np.any(means > 0.0))
    global_not_worse = bool(all(state != "WORSENED" for state in classifications))
    if not scientific_valid:
        gate = "FUNCTIONAL_ONLY"
    elif global_valid and non_adverse_count >= 14 and practically_positive_count >= 12 and lower_ok:
        gate = "GENERALIZES"
    elif global_valid and global_mean_positive and global_not_worse:
        gate = "HETEROGENEOUS"
    else:
        gate = "LIMITED"
    return {
        "gate": gate,
        "global_mean_positive": global_mean_positive,
        "global_not_worse": global_not_worse,
    }


def run_c5_generalization(args, exploratory_60: bool = False):
    stage_name = "c5-generalization-60-exploratory" if exploratory_60 else "c5-generalization"
    _require_nominal_parameters(args)
    parent_stage = "c5-generalization" if exploratory_60 else "c4-attrition"
    parent_gate_file = "c5_gate.md" if exploratory_60 else "c4_gate.md"
    parent_id, parent_config, parent_gate = _require_parent_campaign(args, parent_stage, parent_gate_file)
    config = _base_config(args, stage_name)
    datasets = sorted(args.datasets or (list(range(1, 61)) if exploratory_60 else C5_DATASETS))
    seeds = list(args.seeds_list or range(42, 52))
    full_design = bool(
        not exploratory_60
        and datasets == sorted(C5_DATASETS)
        and seeds == list(range(42, 52))
        and args.size == DEFAULT_SIZE
        and args.budget == DEFAULT_BUDGET
        and args.num_drones == DEFAULT_NUM_DRONES
        and args.drone_speed == DEFAULT_DRONE_SPEED
        and args.fov_deg == DEFAULT_FOV_DEG
        and args.altitude == DEFAULT_ALTITUDE
        and args.victim_speed == DEFAULT_VICTIM_SPEED
        and args.dt == DEFAULT_DT
    )
    config.update({
        "parent_experiment_id": parent_id,
        "parent_git_commit": parent_config.get("git_commit"),
        "parent_source_hash": parent_config.get("source_hash"),
        "parent_compatibility_status": parent_config.get("_compatibility_status"),
        "parent_gate": parent_gate.get("gate"),
        "scientific_valid": full_design,
        "datasets": datasets, "seeds": seeds, "confirmation_datasets": [] if exploratory_60 else datasets,
        "label": "exploratory" if exploratory_60 or not full_design else "confirmatory",
        "size": args.size, "planners": ["online_static_3step", "dynamic_3step"],
        "simulation_victim_model": "none", "evaluation_victim_model": "stationary", "victim_sampling_model": "matched_prior_exact",
        "parallel_unit": ["dataset", "planning_seed"],
        "condition_key": {"stage": stage_name, "w": NOMINAL_W, "p_d": DEFAULT_P_D, "tau": DEFAULT_TAU},
        "expected_rows": len(datasets) * len(seeds) * 2,
        "inclusion_rule": "IDs 1..60 descriptive extension; no retuning" if exploratory_60 else "historical Q1 list; no result-based exclusion",
    })
    config, outdir = _make_campaign(stage_name, config, args.jobs)
    def one(dataset, seed):
        item = _load_item(dataset, args.size)
        condition = {
            "stage": stage_name,
            "w": NOMINAL_W,
            "p_d": DEFAULT_P_D,
            "tau": DEFAULT_TAU,
        }
        frame, _, _ = _run_pair(
            item,
            config,
            dataset_id=dataset,
            planning_seed=seed,
            w=NOMINAL_W,
            p_d=DEFAULT_P_D,
            tau=DEFAULT_TAU,
            budget=args.budget,
            num_drones=args.num_drones,
            condition_key=condition,
        )
        p = _normalise_prior(item.heatmap)
        frame["environment_type"] = item.environment_type
        frame["environment_climate"] = item.environment_climate
        frame["prior_support"] = int(np.count_nonzero(p))
        positive = p > 0
        frame["prior_entropy"] = float(
            -np.sum(p[positive] * np.log(p[positive])),
        )
        return frame

    jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    frames = _parallel_jobs(args, jobs, one, "C5 generalization")
    runs = _with_provenance(pd.concat(frames, ignore_index=True), config)
    _validate_campaign_frame(runs, pair_keys=["dataset", "planning_seed"], expected_arms=["online_static_3step", "dynamic_3step"], expected_rows=config["expected_rows"])
    contrasts = _paired_contrasts(runs, keys=["dataset", "planning_seed"], control="online_static_3step", method="dynamic_3step")
    dataset_effects = _summary_table(contrasts, ["dataset"])
    dataset_effects["non_adverse"] = dataset_effects.state != "WORSENING"
    dataset_effects["practically_positive"] = dataset_effects.state == "IMPROVEMENT"
    metadata = runs.groupby("dataset", as_index=False).agg(
        environment_type=("environment_type", "first"),
        environment_climate=("environment_climate", "first"),
        prior_support=("prior_support", "first"),
        prior_entropy=("prior_entropy", "first"),
    )
    dataset_effects = dataset_effects.merge(metadata, on="dataset", how="left", validate="one_to_one")
    global_rows = []
    for column, margin in (("dPdet", MARGIN_PDET), ("dRMST_rel", MARGIN_RMST_REL)):
        means = contrasts.groupby("dataset", as_index=False)[column].mean().rename(columns={column: "dataset_mean"})
        boot = cluster_bootstrap_ci(means, "dataset_mean", "dataset", seed=20260826, iterations=10_000)
        dataset_values = means["dataset_mean"].to_numpy(dtype=float)
        cluster_tost = tost_equivalence(dataset_values, margin)
        cluster_summary = {
            "mean": boot["mean"],
            "sd": float(np.std(dataset_values, ddof=1)) if len(dataset_values) >= 2 else float("nan"),
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "margin": margin,
            "tost": cluster_tost,
        }
        global_rows.append({
            "endpoint": column,
            "mean": boot["mean"],
            "sd": cluster_summary["sd"],
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "cluster_ci_lo": boot["ci_lo"],
            "cluster_ci_hi": boot["ci_hi"],
            "margin": margin,
            "classification": classify_endpoint(cluster_summary, margin),
            "n_datasets": boot["n_clusters"],
            "n_pairs": int(len(contrasts)),
        })
    global_summary = pd.DataFrame(global_rows)
    non_adverse_count = int(dataset_effects.non_adverse.sum())
    positive_count = int(dataset_effects.practically_positive.sum())
    global_valid = len(dataset_effects) == len(datasets) and len(contrasts) == len(datasets) * len(seeds)
    lower_ok = bool((global_summary.cluster_ci_lo >= -global_summary.margin).all())
    gate_info = _classify_c5_gate(
        scientific_valid=bool(config["scientific_valid"]),
        global_valid=global_valid,
        non_adverse_count=non_adverse_count,
        practically_positive_count=positive_count,
        lower_ok=lower_ok,
        global_means=global_summary["mean"],
        global_classifications=global_summary["classification"],
    )
    gate = gate_info["gate"]
    _write_frame(runs, outdir, "c5_generalization_runs.csv")
    _write_frame(contrasts, outdir, "c5_paired_contrasts.csv")
    _write_frame(dataset_effects, outdir, "c5_dataset_effects.csv")
    _write_frame(global_summary, outdir, "c5_global_summary.csv")
    (outdir / "c5_gate.md").write_text(
        "# C5 gate\n\n"
        + json.dumps({
            "gate": gate,
            **gate_info,
            "dataset_count": len(dataset_effects),
            "non_adverse": non_adverse_count,
            "practically_positive": positive_count,
            "global_cluster_ci_lower_ok": lower_ok,
            "missing_or_duplicate_invalid": not global_valid,
            "bootstrap_unit": "dataset_mean",
            "inclusion_rule": config["inclusion_rule"],
        }, indent=2, default=_json_default)
        + "\n"
    )
    return config["experiment_id"]


def _read_campaign(args, filename: str) -> tuple[dict, pd.DataFrame]:
    if not args.input_experiment_id:
        raise SystemExit("summary stages require --input-experiment-id")
    outdir = OUTPUT_DIR / args.input_experiment_id
    config = json.loads((outdir / "config.json").read_text())
    return config, pd.read_csv(outdir / filename)


def run_summary(args, stage: str):
    mapping = {
        "c1-summary": ("c1_paired_contrasts.csv", "c1_summary_recomputed.csv", ["dataset", "condition_id", "curve"]),
        "c2-summary": ("c2_paired_contrasts.csv", "c2_summary_recomputed.csv", ["dataset", "condition_id"]),
        "c3-summary": ("c3_paired_contrasts.csv", "c3_summary_recomputed.csv", ["dataset", "mobility_model"]),
        "c4-summary": ("c4_paired_contrasts.csv", "c4_summary_recomputed.csv", ["dataset", "k", "comparator"]),
        "c5-summary": ("c5_paired_contrasts.csv", "c5_summary_recomputed.csv", ["dataset"]),
    }
    source, output, groups = mapping[stage]
    config, contrasts = _read_campaign(args, source)
    if stage == "c4-summary":
        return _summary_table(contrasts.rename(columns={"dRetention_Pdet_Dyn_OS": "dPdet", "dDegradation_NRMST_Dyn_OS": "dRMST_rel"}), groups)
    summary = _summary_table(contrasts, groups)
    summary.to_csv(OUTPUT_DIR / args.input_experiment_id / output, index=False)
    return args.input_experiment_id


def run_c6_report(args):
    parent_id, parent_config, parent_gate = _require_parent_campaign(args, "c5-generalization", "c5_gate.md")
    config = _base_config(args, "c6-report")
    config.update({
        "parent_experiment_id": parent_id,
        "parent_gate": parent_gate.get("gate"),
        "label": "descriptive", "condition_key": {"stage": "c6-report"}, "datasets": [],
    })
    config, outdir = _make_campaign("c6-report", config, args.jobs)
    matrix = """# C6 baseline matrix\n\n| Baseline | Information class | Inputs | Training/ground truth | Multi-UAV | Replanning | Reproducibility |\n|---|---|---|---|---|---|---|\n| `pizza_repartition` | geometry-only | bounds, FOV, altitude | none | yes | fixed path | local source |\n| `online_static_3step` | fixed-prior online | heatmap, bounds, FOV | none | yes | online fixed prior | local source |\n| `dynamic_3step` | evidence-aware | heatmap, bounds, FOV, EvidenceBelief | none | yes | online posterior | local source |\n\nExternal map/feature methods remain related work unless their inputs and reproducibility are executable in SAREnv.\n"""
    notes = """# C6 reproducibility notes\n\nBlock B remains frozen at `revision_B_base`; C uses the same `EvidenceBelief`, sensing order, multiplicity, and simulator API. C campaigns use separate planning, victim, sensor, and failure streams and long arm-level CSVs with validated pairing.\n"""
    related = """# C6 related work update\n\nComparisons are separated by information available: geometry-only, fixed-prior online, evidence-aware, and externally map/feature-aware. Non-reproducible external methods are conceptual context, not new causal numbers.\n"""
    for filename, text in (("c6_baseline_matrix.md", matrix), ("c6_reproducibility_notes.md", notes), ("c6_related_work_update.md", related)):
        (outdir / filename).write_text(text)
        root_path = OUTPUT_DIR / filename
        if not root_path.exists():
            root_path.write_text(text)
    return config["experiment_id"]


def _campaign_configs(ids: list[str]) -> dict:
    campaigns = {}
    for experiment_id in ids:
        path = OUTPUT_DIR / experiment_id / "config.json"
        if not path.exists():
            raise SystemExit(f"campaign config not found: {experiment_id}")
        campaigns[experiment_id] = json.loads(path.read_text())
    return campaigns

def _validate_closure_campaigns(ids: list[str], campaigns: dict[str, dict]) -> tuple[dict, dict, dict]:
    required = [
        "c1-sensitivity", "c2-mismatch", "c3-mobility",
        "c4-attrition", "c5-generalization", "c6-random", "c6-editorial",
    ]
    # c6-report was the pre-C6 historical report-only stage; it is accepted as
    # a legacy artifact but is not part of the C6 closure.
    legacy_optional = ["c6-report"]
    if len(ids) != len(set(ids)):
        raise SystemExit("c-final requires unique campaign IDs")
    by_stage = {config.get("stage"): experiment_id for experiment_id, config in campaigns.items()}
    unexpected = sorted(set(by_stage) - set(required) - set(legacy_optional))
    if unexpected:
        raise SystemExit("c-final received unexpected stages: " + ", ".join(unexpected))
    missing = [stage for stage in required if stage not in by_stage]
    if missing:
        raise SystemExit("c-final missing required stages: " + ", ".join(missing))
    c0_manifest = OUTPUT_DIR / "c0_manifest.json"
    if not c0_manifest.exists():
        raise SystemExit("c-final requires c0_manifest.json")
    c0_identity = json.loads(c0_manifest.read_text())
    current_commit = _audit_commit_sha()
    current_source_hash = _source_hash()

    # Two separated identities: C1-C5 stay frozen at the pre-C6 closure; C6 is
    # the new implementation. The frozen pre-C6 manifest is still on disk here
    # (run_c_final archives it only after successful validation).
    pre_c6_path = OUTPUT_DIR / "block_C_manifest_final.json"
    if not pre_c6_path.exists():
        raise SystemExit("c-final requires the frozen pre-C6 block_C_manifest_final.json")
    pre_c6 = json.loads(pre_c6_path.read_text())
    frozen_commit = pre_c6.get("implementation_commit")
    frozen_source_hash = pre_c6.get("source_hash")
    frozen_hashes = pre_c6.get("campaign_config_hashes", {})
    if not frozen_commit or not frozen_source_hash:
        raise SystemExit("c-final: frozen pre-C6 manifest is missing implementation identity")

    # C0 provenance is validated against C0's own frozen identity (the committed
    # c0_manifest.json), never forced equal to the pre-C6 closure or to the
    # current C6 identity. In this repository they coincide, which is recorded
    # as a fact and verified, but the closure does not require equality.
    c0_impl = c0_identity.get("c0_impl_commit")
    c0_source = c0_identity.get("source_hash")
    if not c0_impl or not c0_source:
        raise SystemExit("c-final: c0_manifest.json is missing its frozen C0 identity")
    # Transient C0 identities recorded by the frozen campaigns at runtime (the
    # C1 scheduling base and the C4-hotfix implementation) are admitted through
    # the same closed registries used by the rest of the provenance checks.
    _c0_allowed = {
        c0_impl,
        frozen_commit,
        C4_HOTFIX_IMPLEMENTATION_COMMIT,
        C4_HOTFIX_LEGACY_PARENTS.get("base_git_commit"),
        SCHEDULING_COMPATIBLE_C1["git_commit"],
    }
    _c0_allowed.discard(None)

    for stage, experiment_id in by_stage.items():
        config = campaigns[experiment_id]
        if stage in ("c6-random", "c6-editorial"):
            # C6 identity: produced by the CURRENT implementation. Its parents
            # point at the FROZEN campaigns (checked in the chain section).
            exact_identity = bool(
                config.get("git_commit") == current_commit
                and config.get("source_hash") == current_source_hash
            )
            if not exact_identity:
                raise SystemExit(f"c-final C6 provenance mismatch in stage {stage}")
            continue
        # C1-C5 frozen identity: exact against the pre-C6 closure, plus the
        # closed registries (C1 scheduling exception, C4 hotfix base).
        frozen_identity = bool(
            config.get("git_commit") == frozen_commit
            and config.get("source_hash") == frozen_source_hash
        )
        scheduling_compatible = bool(
            stage == "c1-sensitivity"
            and _is_scheduling_compatible_c1(experiment_id, config)
        )
        hotfix_legacy = bool(_is_c4_hotfix_legacy_parent(experiment_id, config))
        hotfix_identity = bool(_is_c4_hotfix_implementation(config))
        if not frozen_identity and not scheduling_compatible and not hotfix_legacy and not hotfix_identity:
            raise SystemExit(f"c-final provenance mismatch in stage {stage}")
        recorded_c0 = config.get("c0_impl_commit")
        if recorded_c0 not in _c0_allowed:
            raise SystemExit(
                f"c-final C0 identity mismatch in frozen stage {stage}: "
                f"{recorded_c0!r} not in the closed C0 identity registry"
            )
        recorded_hash = frozen_hashes.get(experiment_id)
        if recorded_hash is not None and config.get("config_hash") != recorded_hash:
            raise SystemExit(f"c-final config-hash mismatch for {stage}: {config.get('config_hash')} != {recorded_hash}")
    for stage in required:
        if stage == "c6-editorial":
            continue
        config = campaigns[by_stage[stage]]
        if config.get("scientific_valid") is False or config.get("label") == "functional-only":
            raise SystemExit(f"c-final rejects non-scientific campaign: {stage}")
    chain = [
        ("c2-mismatch", "c1-sensitivity"),
        ("c3-mobility", "c2-mismatch"),
        ("c4-attrition", "c3-mobility"),
        ("c5-generalization", "c4-attrition"),
    ]
    for child_stage, parent_stage in chain:
        child = campaigns[by_stage[child_stage]]
        expected_parent = by_stage[parent_stage]
        if child.get("parent_experiment_id") != expected_parent:
            raise SystemExit(
                f"c-final parent chain mismatch: {child_stage} -> {child.get('parent_experiment_id')!r}; "
                f"expected {expected_parent!r}"
            )
    c6 = campaigns[by_stage["c6-random"]]
    c5_cfg = campaigns[by_stage["c5-generalization"]]
    c4_cfg = campaigns[by_stage["c4-attrition"]]
    if c6.get("parent_experiment_id") != by_stage["c5-generalization"]:
        raise SystemExit("c-final C6 stationary parent mismatch")
    if (
        c6.get("parent_git_commit") != c5_cfg.get("git_commit")
        or c6.get("parent_source_hash") != c5_cfg.get("source_hash")
    ):
        raise SystemExit("c-final C6 stationary parent provenance differs from frozen C5")
    if c6.get("attrition_parent_experiment_id") != by_stage["c4-attrition"]:
        raise SystemExit("c-final C6 attrition parent mismatch")
    if (
        c6.get("attrition_parent_git_commit") != c4_cfg.get("git_commit")
        or c6.get("attrition_parent_source_hash") != c4_cfg.get("source_hash")
    ):
        raise SystemExit("c-final C6 attrition parent provenance differs from frozen C4")
    c6_editorial = campaigns[by_stage["c6-editorial"]]
    if c6_editorial.get("parent_experiment_id") != by_stage["c6-random"]:
        raise SystemExit("c-final C6-editorial parent mismatch")
    if (
        c6_editorial.get("parent_git_commit") != c6.get("git_commit")
        or c6_editorial.get("parent_source_hash") != c6.get("source_hash")
    ):
        raise SystemExit("c-final C6-editorial parent provenance differs from c6-random")
    gate_files = {
        "c1-sensitivity": "c1_gate.md",
        "c2-mismatch": "c2_gate.md",
        "c3-mobility": "c3_gate.md",
        "c4-attrition": "c4_gate.md",
        "c6-random": "c6_gate.md",
    }
    gates = {}
    gates["_c0_identity"] = {
        "c0_impl_commit": c0_impl,
        "matches_pre_c6_closure_identity": bool(
            c0_impl == frozen_commit and c0_source == frozen_source_hash
        ),
        "note": "C0 provenance is validated against the frozen c0_manifest.json identity; the pre-C6 closure records the same identity in this repository, recorded here as a fact, not as a required equality",
    }
    for stage, filename in gate_files.items():
        gate = _read_gate_payload(OUTPUT_DIR / by_stage[stage] / filename)
        if stage == "c6-random":
            gate_value = gate.get("C6_GREEN", gate.get("C6_RANDOM_GREEN"))
            if not gate.get("C6_GREEN", False):
                raise SystemExit(
                    f"c-final refuses non-green C6: C6_GREEN={gate.get('C6_GREEN')} "
                    f"C6_RANDOM_GREEN={gate.get('C6_RANDOM_GREEN')} "
                    f"C6_BASELINES_GREEN={gate.get('C6_BASELINES_GREEN')} "
                    f"C6_EDITORIAL_GREEN={gate.get('C6_EDITORIAL_GREEN')}"
                )
        else:
            gate_value = gate.get("gate") or gate.get("scientific_gate")
        if stage != "c6-random" and gate_value in {"INVALID", "FUNCTIONAL_ONLY", "FAIL"}:
            raise SystemExit(f"c-final refuses failed gate {stage}: {gate_value}")
        gates[stage] = dict(gate)
        if stage != "c6-random":
            gates[stage]["gate"] = gate_value
    raw_files = {
        "c1-sensitivity": "c1_sensitivity_runs.csv",
        "c2-mismatch": "c2_prior_mismatch_runs.csv",
        "c3-mobility": "c3_mobility_runs.csv",
        "c4-attrition": "c4_attrition_runs.csv",
        "c5-generalization": "c5_generalization_runs.csv",
    }
    row_counts = {}
    seeds = {}
    for stage, filename in raw_files.items():
        config = campaigns[by_stage[stage]]
        path = OUTPUT_DIR / by_stage[stage] / filename
        if not path.exists():
            raise SystemExit(f"c-final missing raw artifact: {path}")
        observed = int(len(pd.read_csv(path)))
        expected = config.get("expected_rows")
        if expected is not None and observed != int(expected):
            raise SystemExit(f"c-final row-count mismatch for {stage}: {observed} != {expected}")
        row_counts[stage] = {"expected": expected, "observed": observed}
        seeds[stage] = config.get("seeds", [])
    c6_config = campaigns[by_stage["c6-random"]]
    for raw_name, counter in (
        ("c6_random_stationary_runs.csv", "expected_stationary_rows"),
        ("c6_random_attrition_runs.csv", "expected_attrition_rows"),
    ):
        path = OUTPUT_DIR / by_stage["c6-random"] / raw_name
        if not path.exists():
            raise SystemExit(f"c-final missing C6 raw artifact: {path}")
        observed = int(len(pd.read_csv(path)))
        expected = c6_config.get(counter)
        if observed != int(expected):
            raise SystemExit(f"c-final row-count mismatch for c6-random {raw_name}: {observed} != {expected}")
        row_counts[f"c6-{raw_name[:14]}"] = {"expected": expected, "observed": observed}
    return row_counts, seeds, gates


def run_c_final(args):
    _require_clean_tree(require_baseline=True)
    ids = list(args.campaign_ids or [])
    campaigns = _campaign_configs(ids)
    row_counts, seeds, gates = _validate_closure_campaigns(ids, campaigns)
    implementation_commit = _audit_commit_sha()
    manifest = {
        "block": "C", "schema_version": SCHEMA_VERSION,
        "implementation_commit": implementation_commit,
        "closure_parent_commit": implementation_commit,
        "freeze_ref": "block_C_final",
        "revision_C_base": f"revision_C_base@{implementation_commit[:12]}",
        "behavior_profile": BEHAVIOR_PROFILE, "fix_profile": FIX_PROFILE,
        "block_B_code_commit": BLOCK_B_CODE_COMMIT, "block_B_closure_commit": BLOCK_B_CLOSURE_COMMIT,
        "source_hash": _source_hash(), "campaign_ids": ids,
        "campaign_config_hashes": {key: value.get("config_hash") for key, value in campaigns.items()},
        "stages": {value.get("stage", key): {"experiment_id": key, "status": "executed"} for key, value in campaigns.items()},
        "row_counts": row_counts, "seeds": seeds,
        "endpoints": {"primary": PRIMARY_ENDPOINTS, "secondary": SECONDARY_ENDPOINTS},
        "gates": gates, "artifacts": {"local": True, "server": False},
        "not_executed": {"c1b": "not_run_by_design", "c5-generalization-60": "optional extension not automatic"},
    }
    manifest_path = OUTPUT_DIR / "block_C_manifest_final.json"
    archived = False
    if manifest_path.exists():
        archive_path = OUTPUT_DIR / "block_C_manifest_final.pre_c6.json"
        archive_path.write_text(manifest_path.read_text())
        archived = True
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n")
    closure = OUTPUT_DIR / "block_C_closure.md"
    c6_gate = gates.get("c6-random", {})
    closure.write_text(
        "# Block C closure\n\n"
        "C retains only claims supported by the operational endpoints, paired "
        "confidence intervals, common horizons, and fair controls. Block B "
        "remains frozen; `revision_B_base` and `EvidenceBelief` were not altered. "
        "A moving-target or mismatch direction with a CI crossing zero is "
        "sensitivity, not a robustness claim.\n\n"
        "## C6 (closing block)\n\n"
        "C6 adds the uninformed+memoryless+uncoordinated exploration floor "
        "`random_uniform_n8_uncoordinated` (per-UAV RNG streams keyed by persistent "
        "uav identity; no map/belief/history access; target deconfliction OFF) and "
        "integrates it with the frozen C1-C5 results under exact trial keys. "
        "The safe terminology for fleet reduction is: robustness to reduced "
        "initial fleet size, graceful degradation, online replanning robustness. "
        "`fault tolerance`, `positive synergy`, `evidence-driven fleet resilience`, "
        "and `moving-victim robustness` remain prohibited.\n\n"
        "Scientific output separates three facts: the evidence effect "
        "(dynamic_evidence - online_static), informed online planning vs naive "
        "floor (online_static - Random), and attrition architecture "
        "(Random, pizza_repartition, online planners). "
        f"C6_GREEN={'true' if c6_gate.get('C6_GREEN') else 'false'} (Random's "
        "numeric result is not part of the gate).\n"
    )
    checks = {
        "manifest_exists": manifest_path.exists(), "closure_exists": closure.exists(),
        "source_hash": manifest["source_hash"], "implementation_commit": implementation_commit,
        "archived_previous_manifest": archived, "provenance": True, "pairing": True,
        "common_horizon_policy": True, "c6_green": bool(c6_gate.get("C6_GREEN")),
    }
    (OUTPUT_DIR / "c_final_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    return "block_C_final"


def _add_common_arguments(parser):
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", type=int, default=None)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--w", type=float, default=NOMINAL_W)
    parser.add_argument("--p-d", dest="p_d", type=float, default=DEFAULT_P_D)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--num-drones", type=int, default=DEFAULT_NUM_DRONES)
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG)
    parser.add_argument("--altitude", type=float, default=DEFAULT_ALTITUDE)
    parser.add_argument("--drone-speed", type=float, default=DEFAULT_DRONE_SPEED)
    parser.add_argument("--victim-speed", type=float, default=DEFAULT_VICTIM_SPEED)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--snapshot-interval", type=int, default=50)
    parser.add_argument("--input-experiment-id", default=None)
    parser.add_argument("--campaign-ids", nargs="*", default=None)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--attrition-parent-id", default=None)
    parser.add_argument("--attrition-datasets", nargs="+", type=int, default=None)
    parser.add_argument("--attrition-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--failure-fraction", type=float, default=0.5)
    parser.add_argument("--failure-fractions", nargs="+", type=float, default=None)
    parser.add_argument("--failure-k", type=int, default=1)
    parser.add_argument(
        "--no-shadow-diagnostics",
        action="store_true",
        help="skip the exact-tree shadow diagnostics (trace fields only; "
             "planner decisions, trajectories and metrics are unchanged)",
    )


STAGES = {
    "c0-preflight": run_c0_preflight,
    "c0-smoke": run_c0_smoke,
    "c1-sensitivity": run_c1_sensitivity,
    "c1-summary": lambda args: run_summary(args, "c1-summary"),
    "c1b-confirmation": lambda args: (_ for _ in ()).throw(SystemExit("C1b is not run by design")),
    "c1b-summary": lambda args: (_ for _ in ()).throw(SystemExit("C1b is not run by design")),
    "c2-mismatch": run_c2_mismatch,
    "c2-summary": lambda args: run_summary(args, "c2-summary"),
    "c2-lost-person": run_c2_lost_person,
    "c2-lost-summary": lambda args: run_summary(args, "c2-summary"),
    "c3-mobility": run_c3_mobility,
    "c3-summary": lambda args: run_summary(args, "c3-summary"),
    "c3-controls": run_c3_controls,
    "c4-attrition": run_c4_attrition,
    "c4-summary": lambda args: run_summary(args, "c4-summary"),
    "c5-generalization": run_c5_generalization,
    "c5-summary": lambda args: run_summary(args, "c5-summary"),
    "c5-generalization-60": lambda args: run_c5_generalization(args, exploratory_60=True),
    "c6-report": run_c6_report,
    "c6-random": lambda args: __import__("c6_stage", fromlist=["run_c6_random"]).run_c6_random(args),
    "c6-editorial": lambda args: __import__("c6_editorial", fromlist=["run_c6_editorial"]).run_c6_editorial(args),
    "c7-midflight": lambda args: __import__("c7_stage", fromlist=["run_c7_midflight"]).run_c7_midflight(args),
    "c7-state-matched": lambda args: __import__("c7_stage", fromlist=["run_c7_state_matched"]).run_c7_state_matched(args),
    "c7-timing": lambda args: __import__("c7_stage", fromlist=["run_c7_timing"]).run_c7_timing(args),
    "c7-severity": lambda args: __import__("c7_stage", fromlist=["run_c7_severity"]).run_c7_severity(args),
    "c7-editorial": lambda args: __import__("c7_stage", fromlist=["run_c7_editorial"]).run_c7_editorial(args),
    "c-final": run_c_final,
}


def main():
    parser = argparse.ArgumentParser(description="Block C audit orchestrator")
    parser.add_argument("stage", choices=sorted(STAGES))
    _add_common_arguments(parser)
    args = parser.parse_args()
    args.shadow_diagnostics = not getattr(args, "no_shadow_diagnostics", False)
    args.seeds_list = args.seeds
    if args.stage == "c0-smoke":
        _require_clean_tree(require_baseline=True, precommit_c0=True)
    elif args.stage != "c0-preflight":
        _require_clean_tree(require_baseline=True)
    if args.stage == "c0-preflight":
        STAGES[args.stage](args)
        return
    try:
        result = STAGES[args.stage](args)
    except CampaignAlreadyComplete as exc:
        print(f"campaign {exc.experiment_id} already complete; skipping")
        return
    if result:
        if args.stage in {
            "c0-smoke", "c1-sensitivity", "c2-mismatch", "c2-lost-person",
            "c3-mobility", "c3-controls", "c4-attrition", "c5-generalization",
            "c5-generalization-60", "c6-report", "c6-random", "c6-editorial",
            "c7-midflight", "c7-state-matched", "c7-timing", "c7-severity",
            "c7-editorial",
        }:
            config_path = OUTPUT_DIR / result / "config.json"
            if config_path.exists():
                _write_stage_complete(OUTPUT_DIR / result, json.loads(config_path.read_text()))
        print(f"campaign {result} complete")


if __name__ == "__main__":
    main()

