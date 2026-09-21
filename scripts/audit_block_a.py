#!/usr/bin/env python3
"""Prepare and run the reproducibility audit described in Block A.

This script intentionally keeps the published ``static`` modes untouched. It
uses the same ``StudyConfig``/``run_single_job`` pipeline as the existing
study runner, so paired seeds, initial positions, victims, and parameters are
shared across modes.

No stage is run implicitly. Examples:

  python scripts/audit_block_a.py online --runs 10
  python scripts/audit_block_a.py tree --runs 30
  python scripts/audit_block_a.py differential --runs 20
  python scripts/audit_block_a.py table --csv results/comprehensive_study/study_main_*.csv
  python scripts/audit_block_a.py provenance
  python scripts/audit_block_a.py report
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "audit_block_A"
DEFAULT_DATASETS = [1, 10, 5, 7, 9, 12]
ONLINE_MODES = [
    "online_static",
    "online_static_3step",
    "dynamic",
    "dynamic_3step",
]
TREE_MODES = [
    "dynamic_2step",
    "tree_2step",
    "dynamic_3step",
    "tree_3step",
    "dynamic_4step",
    "tree_4step",
]
DIFFERENTIAL_PROFILES = [
    "original",
    "t0",
    "sensing_order",
    "target_reservation",
    "bellman_no_center",
    "union_observations_per_timestep",
]


def _load_comprehensive_module():
    path = ROOT / "scripts" / "05_comprehensive_study.py"
    spec = importlib.util.spec_from_file_location("sarenv_comprehensive_study", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load study runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _base_config(
    module,
    dataset: int,
    mode: str,
    seed: int,
    profile: str = "original",
    budget: float = 200_000.0,
    revisit_weight: float = 0.5,
    num_drones: int = 5,
):
    implementation_version = f"audit_A@{_audit_commit_sha()}"
    return module.StudyConfig(
        study_name="audit_block_A",
        dataset_id=dataset,
        size="xlarge",
        num_drones=num_drones,
        num_victims=5,
        budget=budget,
        fov_deg=45.0,
        altitude=80.0,
        detection_prob=0.8,
        decay_tau=10_000.0,
        victim_speed=0.5,
        victim_model="random_walk",
        drone_speed=5.0,
        init_strategy="random",
        init_radius=100.0,
        revisit_weight=revisit_weight,
        planning_mode=mode,
        dt=1.0,
        trial_seed=seed,
        trial_num=seed,
        implementation_version=implementation_version,
        fix_profile=profile,
    )


def _run_configs(configs, output_name: str, jobs: int = 1) -> pd.DataFrame:
    module = _load_comprehensive_module()
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    if jobs == 1:
        rows = [module.run_single_job(config) for config in configs]
    else:
        rows = Parallel(n_jobs=jobs, backend="loky")(
            delayed(module.run_single_job)(config) for config in configs
        )
    frame = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / output_name, index=False)
    return frame


def _online_tag(args) -> str:
    tag = f"w{args.w}_{args.fix_profile}"
    if args.modes:
        tag += "_" + "_".join(sorted(args.modes))
    return tag


def run_online(args):
    module = _load_comprehensive_module()
    modes = args.modes or ONLINE_MODES
    tag = _online_tag(args)
    raw_name = f"a1_online_controls_{tag}.csv"
    contrasts_name = f"a1_paired_contrasts_{tag}.csv"
    levels_name = f"a1_horizon_effects_{tag}.csv"
    configs = [
        _base_config(
            module,
            dataset,
            mode,
            args.base_seed + run,
            profile=args.fix_profile,
            budget=args.budget,
            revisit_weight=args.w,
        )
        for dataset in args.datasets
        for run in range(args.runs)
        for mode in modes
    ]
    frame = _run_configs(configs, raw_name, jobs=args.jobs)
    pivot = frame.pivot_table(
        index=["dataset", "seed"], columns="planning_mode", values="L", aggfunc="first"
    ).reset_index()
    if {"dynamic", "online_static"}.issubset(pivot.columns):
        pivot["D1"] = pivot["dynamic"] - pivot["online_static"]
    if {"dynamic_3step", "online_static_3step"}.issubset(pivot.columns):
        pivot["D3"] = pivot["dynamic_3step"] - pivot["online_static_3step"]
    if {"D1", "D3"}.issubset(pivot.columns):
        pivot["I"] = pivot["D3"] - pivot["D1"]
    pivot.to_csv(OUTPUT_DIR / contrasts_name, index=False)

    levels = (
        frame.groupby(["dataset", "planning_mode"])["L"]
        .mean()
        .unstack()
        .reindex(columns=ONLINE_MODES)
    )
    levels.columns = ["L_OS1", "L_OS3", "L_D1", "L_D3"]
    if {"L_OS1", "L_OS3"}.issubset(levels.columns):
        levels["H_S"] = levels["L_OS3"] - levels["L_OS1"]
    if {"L_D1", "L_D3"}.issubset(levels.columns):
        levels["H_D"] = levels["L_D3"] - levels["L_D1"]
    if {"L_D1", "L_OS1"}.issubset(levels.columns):
        levels["D1"] = levels["L_D1"] - levels["L_OS1"]
    if {"L_D3", "L_OS3"}.issubset(levels.columns):
        levels["D3"] = levels["L_D3"] - levels["L_OS3"]
    if {"D3", "D1"}.issubset(levels.columns):
        levels["I"] = levels["D3"] - levels["D1"]
    levels.to_csv(OUTPUT_DIR / levels_name)


def run_a1_summary(args):
    """Regenerate A1 contrasts and per-dataset summary from existing raw CSV."""
    paths = sorted(Path().glob(args.csv))
    if not paths:
        raise FileNotFoundError(f"No CSV matched {args.csv}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    pivot = frame.pivot_table(
        index=["dataset", "seed"], columns="planning_mode", values="L", aggfunc="first"
    ).reset_index()
    if {"dynamic", "online_static"}.issubset(pivot.columns):
        pivot["D1"] = pivot["dynamic"] - pivot["online_static"]
    if {"dynamic_3step", "online_static_3step"}.issubset(pivot.columns):
        pivot["D3"] = pivot["dynamic_3step"] - pivot["online_static_3step"]
    if {"D1", "D3"}.issubset(pivot.columns):
        pivot["I"] = pivot["D3"] - pivot["D1"]
    contrast_cols = [c for c in ("D1", "D3", "I") if c in pivot.columns]
    summary = pivot.groupby("dataset").agg(
        **{f"mean_{c}": (c, "mean") for c in contrast_cols},
        **{f"seeds_{c}_pos": (c, lambda s: int((s > 0).sum())) for c in contrast_cols},
        n=("seed", "size"),
    ).round(6)
    tag = Path(paths[0]).stem.replace("a1_online_controls_", "a1_summary_")
    summary.to_csv(OUTPUT_DIR / f"{tag}.csv")
    print(f"=== A1 summary from {len(frame)} runs ===")
    print(summary.to_string())
    if "I" in pivot:
        print(f"\nGLOBAL: D1={pivot['D1'].mean():+.5f}  D3={pivot['D3'].mean():+.5f}  "
              f"I={pivot['I'].mean():+.5f}  (n pairs={len(pivot)})")


def _trace_metrics(value: str) -> dict:
    try:
        trace = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {"action_agreement": float("nan"), "exact_regret": float("nan"), "n_decisions": 0}
    agreements = [item["action_agreement"] for item in trace if "action_agreement" in item]
    regrets = [item["exact_regret"] for item in trace if "exact_regret" in item]
    rewards = [item["exact_reward"] for item in trace if "exact_reward" in item]
    return {
        "action_agreement": float(np.mean(agreements)) if agreements else float("nan"),
        "exact_regret": float(np.mean(regrets)) if regrets else float("nan"),
        "n_decisions": len(agreements),
        "tree_exact_reward": float(np.mean(rewards)) if rewards else float("nan"),
    }

def _aligned_action_disagreement(left: str, right: str) -> tuple[float, int]:
    def parse(value):
        try:
            return {
                (item["replan_id"], item["drone"]): (
                    item["target_row"],
                    item["target_col"],
                )
                for item in json.loads(value)
            }
        except (TypeError, json.JSONDecodeError, KeyError):
            return {}

    left_actions = parse(left)
    right_actions = parse(right)
    keys = sorted(left_actions.keys() & right_actions.keys())
    if not keys:
        return float("nan"), 0
    changed = sum(left_actions[key] != right_actions[key] for key in keys)
    return changed / len(keys), len(keys)


def _decision_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand per-run planning traces into one row per Bellman decision."""
    rows = []
    for _, run_row in frame.iterrows():
        try:
            trace = json.loads(run_row["planning_trace_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        for item in trace:
            if "exact_optimal_reward" not in item:
                continue
            optimal = float(item["exact_optimal_reward"])
            bellman_reward = float(item["exact_bellman_action_reward"])
            regret = optimal - bellman_reward
            relative = regret / optimal if optimal > 0 else float("nan")
            rows.append({
                "planning_mode": run_row["planning_mode"],
                "seed": run_row["seed"],
                "action_agreement": bool(item.get("action_agreement")),
                "agreement_ties": bool(np.isclose(bellman_reward, optimal)),
                "exact_regret": regret,
                "relative_regret": relative,
                "exact_optimal_reward": optimal,
                "exact_bellman_action_reward": bellman_reward,
            })
    return pd.DataFrame(rows)


def _summarize_tree(frame: pd.DataFrame, output_name: str = "a2_tree_summary.csv"):
    decisions = _decision_frame(frame)
    rows = []
    for horizon in (2, 3, 4):
        mode = f"dynamic_{horizon}step"
        sub = decisions[decisions["planning_mode"] == mode]
        if sub.empty:
            continue
        regrets = sub["exact_regret"].to_numpy(dtype=float)
        relative = sub["relative_regret"].dropna().to_numpy(dtype=float)
        rows.append({
            "horizon": horizon,
            "action_agreement": float(sub["action_agreement"].mean()),
            "agreement_ties": float(sub["agreement_ties"].mean()),
            "mean_exact_regret": float(regrets.mean()),
            "regret_p90": float(np.quantile(regrets, 0.90)),
            "regret_p95": float(np.quantile(regrets, 0.95)),
            "regret_p99": float(np.quantile(regrets, 0.99)),
            "regret_max": float(regrets.max()),
            "mean_relative_regret": float(np.nanmean(relative)) if relative.size else float("nan"),
            "relative_p90": float(np.quantile(relative, 0.90)) if relative.size else float("nan"),
            "relative_p95": float(np.quantile(relative, 0.95)) if relative.size else float("nan"),
            "relative_p99": float(np.quantile(relative, 0.99)) if relative.size else float("nan"),
            "n_decisions": int(len(sub)),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / output_name, index=False)


def run_tree(args):
    module = _load_comprehensive_module()
    configs = [
        _base_config(module, 1, mode, args.base_seed + run, "n8", args.budget)
        for run in range(args.runs)
        for mode in TREE_MODES
    ]
    frame = _run_configs(configs, "a2_tree_bellman.csv", jobs=args.jobs)
    trace_metrics = frame["planning_trace_json"].map(_trace_metrics).apply(pd.Series)
    frame = pd.concat([frame, trace_metrics], axis=1)
    frame.to_csv(OUTPUT_DIR / "a2_tree_bellman.csv", index=False)
    _summarize_tree(frame)


def run_tree_summary(args):
    """Regenerate the extended summary from an existing a2 CSV without re-running."""
    paths = sorted(Path().glob(args.csv))
    if not paths:
        raise FileNotFoundError(f"No CSV matched {args.csv}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    _summarize_tree(frame)


def run_tree_timing(args):
    """Planner-only timing: Bellman value-map cost vs fleet-dependent selection.

    Runs dynamic_Nstep (N=2,3,4) with the shadow tree enabled under the n8
    profile, so ``planner_wall_time`` (already shadow-free) isolates the shared
    value-map construction + per-drone selection, while ``shadow_wall_time``
    records the diagnostic tree. Varying num_drones tests whether the value map
    is fleet-independent.
    """
    module = _load_comprehensive_module()
    horizons = args.horizons or [2, 3, 4]
    fleets = args.fleets or [1, 5, 10]
    configs = []
    for horizon in horizons:
        mode = f"dynamic_{horizon}step"
        for fleet in fleets:
            for run in range(args.runs):
                configs.append(_base_config(
                    module,
                    1,
                    mode,
                    args.base_seed + run,
                    "n8",
                    args.budget,
                    num_drones=fleet,
                ))
    frame = _run_configs(configs, "a2_timing.csv", jobs=args.jobs)
    frame = frame.rename(columns={"num_drones": "fleet"})
    frame["horizon"] = frame["planning_mode"].str.extract(r"(\d+)step").astype(int)
    agg_spec = {
        "n": ("seed", "size"),
        "planner_wall_time_mean": ("planner_wall_time", "mean"),
        "planner_wall_time_sd": ("planner_wall_time", "std"),
        "wall_time_mean": ("wall_time", "mean"),
        "replans_mean": ("number_of_replans", "mean"),
    }
    if "shadow_wall_time" in frame.columns:
        agg_spec["shadow_wall_time_mean"] = ("shadow_wall_time", "mean")
    summary = frame.groupby(["horizon", "fleet"]).agg(**agg_spec).round(4)
    summary["planner_ms_per_replan"] = (
        1000.0 * summary["planner_wall_time_mean"] / summary["replans_mean"]
    )
    summary.to_csv(OUTPUT_DIR / "a2_timing_summary.csv")
    print("=== A2 planner-only timing (planner_wall_time excludes shadow) ===")
    print(summary.to_string())
    if "shadow_wall_time_mean" not in summary.columns:
        print("\n[shadow_wall_time no disponible en este CSV: run previo al fix de emision]")
    print("\nCost per replan scales sublinearly with fleet if the value map dominates.")


def run_timing_summary(args):
    """Regenerate the timing summary from an existing a2_timing CSV."""
    paths = sorted(Path().glob(args.csv))
    if not paths:
        raise FileNotFoundError(f"No CSV matched {args.csv}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.rename(columns={"num_drones": "fleet"})
    frame["horizon"] = frame["planning_mode"].str.extract(r"(\d+)step").astype(int)
    agg_spec = {
        "n": ("seed", "size"),
        "planner_wall_time_mean": ("planner_wall_time", "mean"),
        "planner_wall_time_sd": ("planner_wall_time", "std"),
        "wall_time_mean": ("wall_time", "mean"),
        "replans_mean": ("number_of_replans", "mean"),
    }
    if "shadow_wall_time" in frame.columns:
        agg_spec["shadow_wall_time_mean"] = ("shadow_wall_time", "mean")
    summary = frame.groupby(["horizon", "fleet"]).agg(**agg_spec).round(4)
    summary["planner_ms_per_replan"] = (
        1000.0 * summary["planner_wall_time_mean"] / summary["replans_mean"]
    )
    summary.to_csv(OUTPUT_DIR / "a2_timing_summary.csv")
    print(summary.to_string())


def run_differential(args):
    module = _load_comprehensive_module()
    profiles = args.profiles or DIFFERENTIAL_PROFILES
    configs = [
        _base_config(
            module, dataset, mode, args.base_seed + run, profile, args.budget
        )
        for dataset in args.datasets
        for run in range(args.runs)
        for mode in ("dynamic", "dynamic_3step")
        for profile in profiles
    ]
    frame = _run_configs(configs, "a3_differential_fixes.csv", jobs=args.jobs)
    keys = ["dataset", "seed", "planning_mode"]
    original = frame[frame["fix_profile"] == "original"].set_index(keys)
    rows = []
    for _, row in frame[frame["fix_profile"] != "original"].iterrows():
        key = (row["dataset"], row["seed"], row["planning_mode"])
        baseline = original.loc[key]
        disagreement, comparable = _aligned_action_disagreement(
            row["target_trace_json"],
            baseline["target_trace_json"],
        )
        rows.append({
            **{name: row[name] for name in ("dataset", "seed", "planning_mode", "fix_profile")},
            "delta_L": row["L"] - baseline["L"],
            "delta_cells_observed": row["cells_observed"] - baseline["cells_observed"],
            "delta_victims_found": row["victims_found"] - baseline["victims_found"],
            "initial_action_vector_changed": float(
                row["first_action_sequence_hash"] != baseline["first_action_sequence_hash"]
            ),
            "action_disagreement_rate": disagreement,
            "comparable_action_decisions": comparable,
        })
    tag = f"ds{len(args.datasets)}_r{args.runs}_{'_'.join(profiles)}"
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"a3_differential_effects_{tag}.csv", index=False)


def run_differential_summary(args):
    """Summarize per-fix effects from an existing differential effects CSV."""
    paths = sorted(Path().glob(args.csv))
    if not paths:
        raise FileNotFoundError(f"No CSV matched {args.csv}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summary = frame.groupby(["fix_profile", "planning_mode"]).agg(
        n=("delta_L", "size"),
        mean_delta_L=("delta_L", "mean"),
        median_abs_delta_L=("delta_L", lambda s: s.abs().median()),
        max_abs_delta_L=("delta_L", lambda s: s.abs().max()),
        **{"pct_changed_L": ("delta_L", lambda s: (s.abs() > 1e-6).mean())},
        mean_delta_cells=("delta_cells_observed", "mean"),
        mean_action_disagreement=("action_disagreement_rate", "mean"),
        comparable_decisions=("comparable_action_decisions", "sum"),
    ).round(6)
    tag = Path(paths[0]).stem.replace("a3_differential_effects_", "a3_summary_")
    summary.to_csv(OUTPUT_DIR / f"{tag}.csv")
    print(f"=== A3 differential summary ({len(frame)} pairs) ===")
    print(summary.to_string())


def _prepare_table_frame(frame: pd.DataFrame, table_modes: list[str]) -> pd.DataFrame:
    value_col = "L" if "L" in frame else "likelihood"
    frame = frame.rename(columns={value_col: "L"}).copy()
    if "seed" not in frame:
        if "trial" not in frame:
            raise ValueError("Table I reproduction requires seed or trial")
        frame["seed"] = frame["trial"]
    mode_map = {
        "static": (0, 0),
        "dynamic": (1, 0),
        "static_3step": (0, 1),
        "dynamic_3step": (1, 1),
    }
    unknown = set(table_modes) - set(mode_map)
    if unknown:
        raise ValueError(f"Unsupported Table I modes: {sorted(unknown)}")
    frame = frame[frame["planning_mode"].isin(table_modes)].copy()
    frame["dynamism"] = frame["planning_mode"].map(lambda mode: mode_map[mode][0])
    frame["horizon"] = frame["planning_mode"].map(lambda mode: mode_map[mode][1])
    return frame


def _bootstrap_interaction(frame: pd.DataFrame, iterations: int, seed: int) -> dict:
    required = {"dataset", "seed", "dynamism", "horizon", "L"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Interaction bootstrap requires columns: {sorted(missing)}")
    pivot = frame.pivot_table(
        index=["dataset", "seed"],
        columns=["dynamism", "horizon"],
        values="L",
        aggfunc="first",
    )
    required_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if not required_cells.issubset(set(pivot.columns)):
        raise ValueError("Table I factorial cells are incomplete")
    effects = (
        pivot[(1, 1)]
        - pivot[(0, 1)]
        - pivot[(1, 0)]
        + pivot[(0, 0)]
    ).dropna()
    by_dataset = effects.groupby(level="dataset")
    dataset_ids = np.array(sorted(by_dataset.groups))
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations)
    for iteration in range(iterations):
        sampled_datasets = rng.choice(dataset_ids, size=len(dataset_ids), replace=True)
        dataset_means = []
        for dataset in sampled_datasets:
            seed_values = by_dataset.get_group(dataset).to_numpy()
            sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
            dataset_means.append(float(np.mean(sampled_seeds)))
        samples[iteration] = float(np.mean(dataset_means))
    return {
        "interaction_mean": float(effects.mean()),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "n_datasets": int(len(dataset_ids)),
        "n_dataset_seed_cells": int(len(effects)),
        "resampling": "datasets_then_seeds_within_dataset",
    }


def _fit_lmm(frame: pd.DataFrame, args, reml: bool, suffix: str):
    import statsmodels.formula.api as smf

    metadata = {
        "formula": args.lmm_formula,
        "re_formula": args.re_formula,
        "group_col": args.group_col,
        "reml": reml,
        "method": args.method,
    }
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                args.lmm_formula,
                frame,
                groups=frame[args.group_col],
                re_formula=args.re_formula,
            )
            fitted = model.fit(reml=reml, method=args.method)
        summary_path = OUTPUT_DIR / f"a4_lmm_summary_{suffix}.txt"
        summary_path.write_text(fitted.summary().as_text())
        cov_re = fitted.cov_re
        cov_re.to_csv(OUTPUT_DIR / f"a4_lmm_cov_re_{suffix}.csv")
        values = np.asarray(cov_re, dtype=float)
        std = np.sqrt(np.clip(np.diag(values), 0.0, None))
        denominator = np.outer(std, std)
        correlation = np.divide(
            values,
            denominator,
            out=np.zeros_like(values),
            where=denominator > 0,
        )
        pd.DataFrame(correlation, index=cov_re.index, columns=cov_re.columns).to_csv(
            OUTPUT_DIR / f"a4_lmm_corr_re_{suffix}.csv"
        )
        eigenvalues = np.linalg.eigvalsh(values)
        metadata.update({
            "converged": bool(getattr(fitted, "converged", False)),
            "singular": bool(np.min(eigenvalues) <= 1e-10),
            "random_effect_sd": std.tolist(),
            "warnings": [str(item.message) for item in captured],
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "log_likelihood": float(fitted.llf),
        })
    except Exception as exc:  # pragma: no cover - depends on optimizer/data
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    (OUTPUT_DIR / f"a4_lmm_metadata_{suffix}.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True)
    )


def run_table(args):
    paths = sorted(Path().glob(args.csv))
    if not paths:
        raise FileNotFoundError(f"No CSV matched {args.csv}")
    raw_frame = pd.concat(
        [pd.read_csv(path).assign(source=str(path)) for path in paths],
        ignore_index=True,
    )
    frame = _prepare_table_frame(raw_frame, args.table_i_modes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sanity = {
        "n_rows_raw": len(raw_frame),
        "n_rows_table_i": len(frame),
        "expected_rows": args.expected_rows,
        "row_count_matches_expected": len(frame) == args.expected_rows,
        "n_datasets": frame["dataset"].nunique(),
        "n_seeds": frame["seed"].nunique(),
        "n_modes": frame["planning_mode"].nunique(),
        "modes": sorted(frame["planning_mode"].unique()),
        "duplicates_dataset_seed_mode": int(
            frame.duplicated(["dataset", "seed", "planning_mode"]).sum()
        ),
        "missing_values": int(frame.isna().sum().sum()),
        "likelihood_min": float(frame["L"].min()),
        "likelihood_max": float(frame["L"].max()),
    }
    (OUTPUT_DIR / "a4_sanity.json").write_text(json.dumps(sanity, indent=2))
    frame.to_csv(OUTPUT_DIR / "a4_table_i_frame.csv", index=False)
    frame.groupby(["dataset", "planning_mode"])["L"].agg(
        ["mean", "std", "count"]
    ).reset_index().to_csv(OUTPUT_DIR / "a4_descriptive_means.csv", index=False)
    try:
        result = _bootstrap_interaction(
            frame, args.bootstrap_iterations, args.base_seed
        )
    except ValueError as exc:
        result = {"error": str(exc)}
    (OUTPUT_DIR / "a4_hierarchical_bootstrap.json").write_text(
        json.dumps(result, indent=2)
    )

    if args.lmm_formula:
        try:
            import statsmodels  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Table audit LMM requires statsmodels") from exc
        _fit_lmm(frame, args, reml=False, suffix="ml")
        _fit_lmm(frame, args, reml=True, suffix="reml")


def run_provenance(args):
    values = ["1.47", "1.41", "1.19"]
    suffixes = {".csv", ".log", ".ipynb", ".tex", ".md", ".py"}
    value_pattern = re.compile(r"|".join(re.escape(value) for value in values))
    symbol_patterns = {
        "simulation_count": re.compile(r"victims_found_count"),
        "geometric_percentage": re.compile(r"percentage_found|Victims Found"),
        "summary_column": re.compile(r"victims_found|Mean_Victims_Found"),
        "aggregation": re.compile(r"groupby|mean|agg"),
    }
    rows = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes or ".git" in path.parts:
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        relative_path = str(path.relative_to(ROOT))
        for line_number, line in enumerate(lines, start=1):
            match = value_pattern.search(line)
            if match:
                rows.append({
                    "kind": "literal_value",
                    "value": match.group(0),
                    "path": relative_path,
                    "line": line_number,
                    "text": line.strip(),
                })
            for kind, symbol_pattern in symbol_patterns.items():
                if symbol_pattern.search(line):
                    rows.append({
                        "kind": kind,
                        "value": "",
                        "path": relative_path,
                        "line": line_number,
                        "text": line.strip(),
                    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "a5_victims_found_candidates.csv", index=False)
    (OUTPUT_DIR / "a5_pipeline_map.json").write_text(json.dumps({
        "temporal_stochastic_pipeline": [
            "published value",
            "summary CSV",
            "aggregation",
            "per-run victims_found",
            "SARSimulation.victims_found_count",
        ],
        "geometric_path_pipeline": [
            "published value",
            "figure/table generator",
            "PathEvaluator.victim_detection_metrics.percentage_found",
        ],
        "candidate_values": values,
        "candidate_kinds": sorted(symbol_patterns),
        "interpretation": "Candidates require manual chain verification; discovery is not provenance proof.",
    }, indent=2))


def run_report(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    a1_path = OUTPUT_DIR / "a1_paired_contrasts.csv"
    if a1_path.exists():
        a1_frame = pd.read_csv(a1_path)
        a1 = {
            "n_paired_runs": int(len(a1_frame)),
            "mean_D1": float(a1_frame["D1"].mean()),
            "mean_D3": float(a1_frame["D3"].mean()),
            "mean_I": float(a1_frame["I"].mean()),
        }
    else:
        a1 = {"status": "not executed"}
    a4_path = OUTPUT_DIR / "a4_sanity.json"
    a4 = json.loads(a4_path.read_text()) if a4_path.exists() else {"status": "not executed"}
    provenance_path = OUTPUT_DIR / "a5_victims_found_candidates.csv"
    provenance = pd.read_csv(provenance_path) if provenance_path.exists() else pd.DataFrame()
    report = [
        "# Block A audit decision report",
        "",
        "This report is generated from audit outputs; it does not infer results when a stage was not run.",
        "",
        "| Claim | Evidence | Status |",
        "|---|---|---|",
        f"| Dynamic belief × horizon interaction | {a1} | {'pending' if 'status' in a1 else 'review'} |",
        "| Bellman ≈ exact tree | a2_tree_summary.csv | pending unless the file exists and is inspected |",
        "| Isolated fixes | a3_differential_effects.csv | pending unless the file exists and is inspected |",
        f"| Table I sanity | {a4} | {'pending' if 'status' in a4 else 'review'} |",
        f"| victims_found provenance | {len(provenance)} source candidates | provenance chain required |",
        "",
        "## Required gates",
        "",
        "* A1 uses paired seed contrasts D1, D3, and I.",
        "* A2 compares first-action agreement and exact FOV-union reward regret.",
        "* A3 compares every fix against the original profile, never only a cumulative stack.",
        "* A4 preserves the original model specification before any replacement inference.",
        "* A5 keeps temporal stochastic detections separate from geometric PathEvaluator coverage.",
    ]
    (OUTPUT_DIR / "a_decision_report.md").write_text("\n".join(report) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["online", "a1-summary", "tree", "tree-summary", "tree-timing", "timing-summary",
                 "differential", "differential-summary", "table", "provenance", "report"],
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", type=int, default=DEFAULT_DATASETS)
    parser.add_argument("--csv", default="results/comprehensive_study/study_main_*.csv")
    parser.add_argument("--lmm-formula", default="L ~ dynamism * horizon")
    parser.add_argument("--re-formula", default="~ dynamism + horizon")
    parser.add_argument("--table-i-modes", nargs="+", default=[
        "static", "dynamic", "static_3step", "dynamic_3step",
    ])
    parser.add_argument("--expected-rows", type=int, default=7_200)
    parser.add_argument("--group-col", default="dataset")
    parser.add_argument("--budget", type=float, default=200_000.0)
    parser.add_argument("--w", type=float, default=0.5, help="Revisit weight for the online-control factorial")
    parser.add_argument("--modes", nargs="+", default=None, help="Restrict the online stage to these planning modes")
    parser.add_argument("--fix-profile", default="original", help="Audit fix profile for simulation stages")
    parser.add_argument("--horizons", nargs="+", type=int, default=None, help="Horizons for tree-timing")
    parser.add_argument("--fleets", nargs="+", type=int, default=None, help="Fleet sizes for tree-timing")
    parser.add_argument("--profiles", nargs="+", default=None, help="Fix profiles for the differential stage")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes for simulation stages")
    parser.add_argument("--method", default="nm")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser.parse_args()


def main():
    args = parse_args()
    dispatch = {
        "online": run_online,
        "a1-summary": run_a1_summary,
        "tree": run_tree,
        "tree-summary": run_tree_summary,
        "tree-timing": run_tree_timing,
        "timing-summary": run_timing_summary,
        "differential": run_differential,
        "differential-summary": run_differential_summary,
        "table": run_table,
        "provenance": run_provenance,
        "report": run_report,
    }
    dispatch[args.stage](args)


if __name__ == "__main__":
    main()
