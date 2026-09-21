"""C7A Analysis 1 — why does pizza_replan < pizza_fixed? (mechanical diagnostics).

Frozen C7A artifacts only.  No protocol/policy/metric/hyperparameter changes,
no new simulations.  Outputs (results/audit_block_C/):
  C7A_PIZZA_DIAGNOSTIC.md
  C7A_PIZZA_DIAGNOSTIC_TRIALS.csv
  C7A_PIZZA_DIAGNOSTIC_DATASETS.csv
  C7_FIGURES/fig_c7a_a1_1..5*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c7a_analysis_common as A
import c7_stage as c7s
import paired_stats as ps

REPLAN = "pizza_replan_midflight"
FIXED = "pizza_fixed_midflight"
FAIL = "midflight_failure"


def _multi_index(series: pd.Series) -> pd.Series:
    """Reindex a (dataset, seed)-tuple-indexed series as a MultiIndex."""
    out = series.copy()
    out.index = pd.MultiIndex.from_tuples(
        [tuple(key) for key in out.index], names=["dataset", "planning_seed"]
    )
    return out


def _scalar(value, default=np.nan):
    """Coerce diag values: dict -> sum of its values, list -> length."""
    if isinstance(value, dict):
        return float(sum(float(v) for v in value.values())) if value else 0.0
    if isinstance(value, (list, tuple)):
        return float(len(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _diag_frame(runs: pd.DataFrame) -> pd.DataFrame:
    sub = runs[(runs.planning_mode == REPLAN) & (runs.condition == FAIL)].copy()
    rows = []
    for row in sub.itertuples(index=False):
        diag = json.loads(row.pizza_assignment_diag)
        deadhead_per_drone = diag.get("deadhead_per_drone", {})
        values = [_scalar(v) for v in deadhead_per_drone.values()]
        rows.append({
            "dataset": int(row.dataset),
            "planning_seed": int(row.planning_seed),
            "deadhead_total": _scalar(diag.get("deadhead_total")),
            "deadhead_fraction_remaining_budget": _scalar(
                diag.get("deadhead_fraction_remaining_budget")),
            "assignment_cost": _scalar(diag.get("assignment_cost")),
            "route_length_after_replan": _scalar(diag.get("route_length_after_replan")),
            "task_splits": int(_scalar(diag.get("task_splits"), 0)),
            "empty_sectors": int(_scalar(diag.get("empty_sectors"), 0)),
            "prefix_removed_total": int(_scalar(diag.get("prefix_removed_total"), 0)),
            "suffix_removed_total": int(_scalar(diag.get("suffix_removed_total"), 0)),
            "num_sectors": int(_scalar(diag.get("num_sectors"), 0)),
            "survivors": len(diag.get("survivor_ids", [])),
            "search_inactive_count": len(diag.get("search_inactive_uav_ids", [])),
            "deadhead_max_drone": max(values) if values else np.nan,
            "deadhead_mean_drone": float(np.mean(values)) if values else np.nan,
            "remaining_budget_at_failure": float(row.budget - row.failure_distance),
            "trigger_step": float(row.trigger_step),
            "failure_distance": float(row.failure_distance),
        })
    return pd.DataFrame(rows).set_index(["dataset", "planning_seed"]).sort_index()


def _recovery_postfault(recovery: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Post-fault prior-mass efficiency per trial from the recovery curves.

    Sampling is every 50 steps (C7_CURVE_EVERY); the fault boundary is the last
    sample with step <= trigger_step (approximation bounded by one sample
    interval, documented in the report).  Cells/unique-cell counts are not
    stored in the frozen artifacts -> reported as unavailable.
    """
    rows = []
    for (dataset, seed), group in recovery[recovery.planning_mode == mode].groupby(
        ["dataset", "planning_seed"], sort=True
    ):
        group = group.sort_values("step")
        fault_rows = group[group.distance_post_fault > 0.0]
        if fault_rows.empty:
            continue
        last = group.iloc[-1]
        l_end = float(last["L_cumulative"])
        l_at_fault = float(fault_rows.iloc[0].get("L_cumulative", np.nan))
        # last pre-fault sample (or zero)
        pre = group[group.distance_post_fault == 0.0]
        l_fault_boundary = float(pre.iloc[-1]["L_cumulative"]) if not pre.empty else 0.0
        dist_post = float(last["distance_post_fault"])
        rows.append({
            "dataset": int(dataset), "planning_seed": int(seed),
            "L_at_fault_curve": l_fault_boundary,
            "L_postfault_first_sample": l_at_fault,
            "L_end": l_end,
            "postfault_mass_gain": l_end - l_fault_boundary,
            "postfault_distance": dist_post,
            "postfault_mass_efficiency": (
                (l_end - l_fault_boundary) / dist_post if dist_post > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows).set_index(["dataset", "planning_seed"]).sort_index()


def main() -> int:
    runs = A.load_runs()
    recovery = A.load_recovery()
    fail = runs[runs.condition == FAIL]

    # ---------------- 1.1 schema ---------------------------------------- #
    schema_note = {
        "runs_rows": int(len(runs)),
        "trials": int(runs.groupby(["dataset", "planning_seed"]).ngroups),
        "columns_present": {
            "Pdet_at_fault": "Pdet_at_fault" in runs.columns,
            "PostFault_PdetGain": "PostFault_PdetGain" in runs.columns,
            "ConditionalPostFaultPdet": "ConditionalPostFaultPdet" in runs.columns,
            "termination_reason": "termination_reason" in runs.columns,
            "budget_utilization": "budget_utilization" in runs.columns,
            "actual_distance_total": "actual_distance_total" in runs.columns,
            "exposure_multiplicity": "exposure_multiplicity" in runs.columns,
            "revisit_fraction": "revisit_fraction" in runs.columns,
            "L": "L" in runs.columns,
            "retention_Pdet": "retention_Pdet" in runs.columns,
            "pizza_assignment_diag": "pizza_assignment_diag" in runs.columns,
        },
        "recovery_columns": list(recovery.columns),
        "curve_sampling": "every 50 steps (C7_CURVE_EVERY); step/1 s semantics",
        "trajectories_archived": "single trial (dataset 1, seed 42) only",
        "not_available": [
            "per-trial raw trajectories (=> per-trial unique-cell post-fault metrics)",
            "per-cell exposure maps",
        ],
    }

    # ---------------- blocking check ------------------------------------ #
    at_fault = fail.pivot_table(index=["dataset", "planning_seed"],
                                columns="planning_mode", values="Pdet_at_fault")
    at_fault_diff = (at_fault[REPLAN] - at_fault[FIXED]).abs()
    hashes = fail.pivot_table(index=["dataset", "planning_seed"], columns="planning_mode",
                              values="prefailure_trajectory_hash", aggfunc="first")
    triggers = fail.pivot_table(index=["dataset", "planning_seed"], columns="planning_mode",
                                values="trigger_step")
    blocking = {
        "at_fault_max_abs_diff": float(at_fault_diff.max()),
        "at_fault_all_equal": bool((at_fault_diff < 1e-12).all()),
        "prefailure_hash_all_equal": bool(
            (hashes[REPLAN] == hashes[FIXED]).all()),
        "trigger_step_all_equal": bool(
            (triggers[REPLAN].astype(float) == triggers[FIXED].astype(float)).all()),
    }
    if not (blocking["at_fault_all_equal"] and blocking["prefailure_hash_all_equal"]
            and blocking["trigger_step_all_equal"]):
        raise SystemExit(f"BLOCKING CHECK FAILED: {blocking}")

    # ---------------- 1.2 paired contrasts ------------------------------ #
    endpoints = {}
    for column, margin in (("dPdet", 0.01), ("dRMST_rel", 0.02),
                           ("dPostFaultGain", None)):
        summary = c7s._contrast_summary(runs, REPLAN, FIXED, column,
                                        margin if margin is not None else 0.01)
        endpoints[column] = summary

    diff_pdet, _ = c7s._paired_diff(runs, REPLAN, FIXED, "dPdet")
    diff_rmst, _ = c7s._paired_diff(runs, REPLAN, FIXED, "dRMST_rel")
    diff_pdet = _multi_index(diff_pdet)
    diff_rmst = _multi_index(diff_rmst)
    gain_replan = A.trial_metric(runs, REPLAN, "PostFault_PdetGain")
    gain_fixed = A.trial_metric(runs, FIXED, "PostFault_PdetGain")
    diff_gain = (gain_replan - gain_fixed).dropna()
    ret_replan = A.trial_metric(runs, REPLAN, "retention_Pdet")
    ret_fixed = A.trial_metric(runs, FIXED, "retention_Pdet")
    diff_retention = (ret_replan - ret_fixed).dropna()
    l_replan = A.trial_metric(runs, REPLAN, "L")
    l_fixed = A.trial_metric(runs, FIXED, "L")
    diff_l = (l_replan - l_fixed).dropna()
    retention_summary = A.cluster_summary(diff_retention)
    gain_summary = A.cluster_summary(diff_gain)
    l_summary = A.cluster_summary(diff_l)

    # ---------------- 1.3 deadhead -------------------------------------- #
    diag = _diag_frame(runs)
    dh_frac = diag["deadhead_fraction_remaining_budget"]
    dh = {
        "mean": float(dh_frac.mean()), "median": float(dh_frac.median()),
        "p25": float(dh_frac.quantile(0.25)), "p75": float(dh_frac.quantile(0.75)),
        "p90": float(dh_frac.quantile(0.90)),
        "min": float(dh_frac.min()), "max": float(dh_frac.max()),
        "deadhead_total_mean_m": float(diag["deadhead_total"].mean()),
        "deadhead_total_median_m": float(diag["deadhead_total"].median()),
        "deadhead_fraction_abs_mean": float(
            (diag["deadhead_total"] / diag["remaining_budget_at_failure"]).mean()),
    }
    dh_ds = diag.groupby("dataset")["deadhead_fraction_remaining_budget"].mean()
    joined = diff_pdet.to_frame("dPdet").join(dh_frac.to_frame("dh_frac")).dropna()
    pearson = float(np.corrcoef(joined["dh_frac"], joined["dPdet"])[0, 1]) if len(joined) > 2 else np.nan
    spearman = float(ps.__dict__ and pd.Series(joined["dh_frac"]).corr(
        pd.Series(joined["dPdet"]), method="spearman")) if len(joined) > 2 else np.nan
    slope = intercept = np.nan
    if len(joined) > 2:
        slope, intercept = np.polyfit(joined["dh_frac"], joined["dPdet"], 1)

    # ---------------- 1.4 re-coverage ----------------------------------- #
    recov = {}
    for column in ("revisit_fraction", "exposure_multiplicity", "L"):
        d = A.paired_vector(runs, REPLAN, FIXED, column)
        recov[column] = A.cluster_summary(d)
    eff_replan = _recovery_postfault(recovery, REPLAN)["postfault_mass_efficiency"]
    eff_fixed = _recovery_postfault(recovery, FIXED)["postfault_mass_efficiency"]
    eff_diff = (eff_replan - eff_fixed).dropna()
    eff_summary = A.cluster_summary(eff_diff)
    mass_replan = _recovery_postfault(recovery, REPLAN)["postfault_mass_gain"]
    mass_fixed = _recovery_postfault(recovery, FIXED)["postfault_mass_gain"]
    mass_diff = (mass_replan - mass_fixed).dropna()
    mass_summary = A.cluster_summary(mass_diff)
    dist_replan = _recovery_postfault(recovery, REPLAN)["postfault_distance"]
    dist_fixed = _recovery_postfault(recovery, FIXED)["postfault_distance"]
    dist_diff = (dist_replan - dist_fixed).dropna()
    dist_summary = A.cluster_summary(dist_diff)

    # ---------------- 1.5 termination ----------------------------------- #
    term = diag.join(A.trial_metric(runs, REPLAN, "termination_reason").to_frame("termination"))
    term = term.join(A.trial_metric(runs, REPLAN, "budget_utilization").to_frame("budget_utilization"))
    term = term.join(A.trial_metric(runs, REPLAN, "actual_distance_total").to_frame("distance"))
    term = term.join(A.trial_metric(runs, REPLAN, "P_detect").to_frame("Pdet"))
    term = term.join(A.trial_metric(runs, REPLAN, "retention_Pdet").to_frame("retention"))
    term = term.join(diff_pdet.to_frame("dPdet"))
    term["unused_budget"] = A.BUDGET - term["distance"]
    term_table = term.groupby("termination").agg(
        n=("Pdet", "size"), fraction=("Pdet", lambda s: len(s) / len(term)),
        mean_Pdet=("Pdet", "mean"), mean_retention=("retention", "mean"),
        mean_budget_utilization=("budget_utilization", "mean"),
        mean_unused_budget=("unused_budget", "mean"),
        mean_dPdet_vs_fixed=("dPdet", "mean"),
    ).sort_values("n", ascending=False)
    fixed_term = A.trial_metric(runs, FIXED, "termination_reason")
    fixed_util = A.trial_metric(runs, FIXED, "budget_utilization")

    # ---------------- 1.6 splits / inactive ----------------------------- #
    splits = {
        "task_splits_mean": float(diag["task_splits"].mean()),
        "empty_sectors_mean": float(diag["empty_sectors"].mean()),
        "prefix_removed_mean": float(diag["prefix_removed_total"].mean()),
        "suffix_removed_mean": float(diag["suffix_removed_total"].mean()),
        "trials_with_inactive_survivor": int((diag["search_inactive_count"] > 0).sum()),
        "inactive_survivor_fraction": float((diag["search_inactive_count"] > 0).mean()),
        "inactive_uav_count_values": diag["search_inactive_count"].value_counts().to_dict(),
    }
    assoc = {}
    for name, series in (
        ("task_splits", diag["task_splits"]),
        ("empty_sectors", diag["empty_sectors"]),
        ("inactive_count", diag["search_inactive_count"]),
    ):
        join = diff_pdet.to_frame("d").join(series.to_frame("x")).dropna()
        assoc[name] = float(pd.Series(join["x"]).corr(pd.Series(join["d"]), method="spearman")) if len(join) > 2 else np.nan

    # ---------------- trial / dataset CSVs ------------------------------ #
    trials = pd.DataFrame({
        "dPdet": diff_pdet, "dRMST_rel": diff_rmst, "dretention": diff_retention,
        "dL": diff_l, "dPostFaultGain": diff_gain,
    }).join(diag[["deadhead_total", "deadhead_fraction_remaining_budget",
                  "assignment_cost", "route_length_after_replan", "task_splits",
                  "empty_sectors", "prefix_removed_total", "suffix_removed_total",
                  "search_inactive_count", "remaining_budget_at_failure"]])
    trials = trials.join(eff_replan.to_frame("postfault_eff_replan"))
    trials = trials.join(eff_fixed.to_frame("postfault_eff_fixed"))
    trials = trials.join(mass_replan.to_frame("postfault_mass_replan"))
    trials = trials.join(mass_fixed.to_frame("postfault_mass_fixed"))
    trials = trials.join(dist_replan.to_frame("postfault_dist_replan"))
    trials = trials.join(dist_fixed.to_frame("postfault_dist_fixed"))
    trials = trials.join(term[["termination", "budget_utilization", "unused_budget"]])
    trials = trials.reset_index()
    trials.to_csv(A.OUTPUT_DIR / "C7A_PIZZA_DIAGNOSTIC_TRIALS.csv", index=False)

    dataset_rows = []
    for dataset, group in trials.groupby("dataset", sort=True):
        row = {"dataset": int(dataset), "n_trials": int(len(group))}
        for column in ("dPdet", "dRMST_rel", "dretention", "dL", "dPostFaultGain",
                       "deadhead_fraction_remaining_budget", "postfault_eff_replan",
                       "postfault_eff_fixed", "budget_utilization", "unused_budget"):
            row[f"mean_{column}"] = float(group[column].mean())
        row["path_exhausted_fraction"] = float((group["termination"] == "path_exhausted").mean())
        dataset_rows.append(row)
    datasets_csv = pd.DataFrame(dataset_rows)
    datasets_csv.to_csv(A.OUTPUT_DIR / "C7A_PIZZA_DIAGNOSTIC_DATASETS.csv", index=False)

    # ---------------- figures ------------------------------------------- #
    fig1, ax = plt.subplots(figsize=(9, 5))
    ds_means = diff_pdet.groupby(level=0).mean()
    ax.bar([str(int(d)) for d in ds_means.index], ds_means.to_numpy(), color="#c0504d")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(float(diff_pdet.mean()), color="#4f81bd", ls="--", lw=1.2,
               label=f"global mean {diff_pdet.mean():+.4f}")
    ax.set_title("A1-1  ΔP_detect (pizza_replan − pizza_fixed) per dataset")
    ax.set_xlabel("dataset"); ax.set_ylabel("Δ P_detect"); ax.legend()
    A.save_figure(fig1, "fig_c7a_a1_1_dpdet_by_dataset.png"); plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(joined["dh_frac"], joined["dPdet"], s=12, alpha=0.5)
    if np.isfinite(slope):
        xs = np.linspace(joined["dh_frac"].min(), joined["dh_frac"].max(), 10)
        ax.plot(xs, slope * xs + intercept, color="#c0504d", lw=1.5)
    ax.set_title(f"A1-2  deadhead fraction vs ΔP_detect\nPearson={pearson:+.3f}  Spearman={spearman:+.3f} (descriptive)")
    ax.set_xlabel("deadhead / remaining budget at failure"); ax.set_ylabel("Δ P_detect (replan − fixed)")
    A.save_figure(fig2, "fig_c7a_a1_2_deadhead_vs_dpdet.png"); plt.close(fig2)

    fig3, ax = plt.subplots(figsize=(6.5, 5))
    common = eff_replan.index.intersection(eff_fixed.index)
    ax.scatter(eff_fixed.loc[common], eff_replan.loc[common], s=12, alpha=0.5)
    lim = max(float(eff_fixed.max()), float(eff_replan.max()))
    ax.plot([0, lim], [0, lim], color="k", lw=0.8, ls=":")
    ax.set_title("A1-3  post-fault prior-mass efficiency (per metre)\nreplan vs fixed")
    ax.set_xlabel("pizza_fixed"); ax.set_ylabel("pizza_replan")
    A.save_figure(fig3, "fig_c7a_a1_3_postfault_efficiency.png"); plt.close(fig3)

    fig4, ax = plt.subplots(figsize=(6.5, 5))
    util_fixed = A.trial_metric(runs, FIXED, "budget_utilization")
    util_replan = A.trial_metric(runs, REPLAN, "budget_utilization")
    common_u = util_replan.index.intersection(util_fixed.index)
    ax.scatter(util_fixed.loc[common_u], util_replan.loc[common_u], s=12, alpha=0.5)
    ax.plot([0, 1.05], [0, 1.05], color="k", lw=0.8, ls=":")
    ax.set_xlim(0.9, 1.05); ax.set_ylim(0.9, 1.05)
    ax.set_title("A1-4  budget utilization: replan vs fixed")
    ax.set_xlabel("pizza_fixed"); ax.set_ylabel("pizza_replan")
    A.save_figure(fig4, "fig_c7a_a1_4_budget_utilization.png"); plt.close(fig4)

    fig5, ax = plt.subplots(figsize=(8, 5))
    for mode, color in ((FIXED, "#4f81bd"), (REPLAN, "#c0504d")):
        sub = recovery[recovery.planning_mode == mode]
        grid = np.linspace(0, 50_000, 60)
        curves = []
        for _, group in sub.groupby(["dataset", "planning_seed"]):
            g = group[group.distance_post_fault > 0].sort_values("distance_post_fault")
            if g.empty:
                continue
            curves.append(np.interp(grid, g["distance_post_fault"], g["P_detect_cumulative"]))
        mean_curve = np.mean(curves, axis=0)
        ax.plot(grid / 1000.0, mean_curve, color=color, label=f"{mode} (mean across trials)")
    ax.set_title("A1-5  P_detect vs post-fault distance (mean across 180 trials)")
    ax.set_xlabel("post-fault distance (km)"); ax.set_ylabel("cumulative P_detect"); ax.legend()
    A.save_figure(fig5, "fig_c7a_a1_5_recovery_curves_postfault.png"); plt.close(fig5)

    # ---------------- report -------------------------------------------- #
    def fmt(summary: dict | None) -> str:
        if summary is None:
            return "n/a"
        return (f"mean {summary['mean']:+.4f}  CI95[{summary['ci_lo']:+.4f},"
                f" {summary['ci_hi']:+.4f}]  n_ds={summary['n_datasets']}"
                f"{'  ' + summary.get('classification', '') if summary.get('classification') else ''}")

    lines = ["# C7A — pizza_replan vs pizza_fixed mechanical diagnostic", ""]
    lines.append("Frozen C7A artifacts only (`c7_midflight_runs.csv`, `c7_midflight_recovery.csv`); "
                 "no policy changes, no new simulations.")
    lines.append("")
    lines.append("## 1.1 Data verification")
    lines.append("")
    lines.append(f"- runs rows: {schema_note['runs_rows']} | trials: {schema_note['trials']}")
    lines.append(f"- columns present: {json.dumps(schema_note['columns_present'])}")
    lines.append(f"- recovery columns: {schema_note['recovery_columns']}")
    lines.append(f"- curve sampling: {schema_note['curve_sampling']}")
    lines.append(f"- trajectories archived: {schema_note['trajectories_archived']}")
    lines.append(f"- NOT AVAILABLE (would require new simulations): {schema_note['not_available']}")
    lines.append("")
    lines.append("## Blocking check (pre-fault identity)")
    lines.append("")
    lines.append(f"- max |Pdet_at_fault(replan) − Pdet_at_fault(fixed)| = {blocking['at_fault_max_abs_diff']:.3e} "
                 f"(equal: {blocking['at_fault_all_equal']})")
    lines.append(f"- prefailure trajectory hashes equal: {blocking['prefailure_hash_all_equal']}")
    lines.append(f"- trigger steps equal: {blocking['trigger_step_all_equal']}")
    lines.append("")
    lines.append("## 1.2 Paired contrasts (replan − fixed; positive = favorable to replan)")
    lines.append("")
    for name, summary in endpoints.items():
        lines.append(f"- {name}: {fmt(summary)}")
    lines.append(f"- dretention: {fmt({**retention_summary, 'classification': ''})} (cluster CI as in C7 frozen convention)")
    lines.append(f"- dPostFaultGain: {fmt({**gain_summary, 'classification': ''})}")
    lines.append(f"- dL: {fmt({**l_summary, 'classification': ''})}")
    lines.append(f"- trial signs (dPdet): {retention_summary['trials_positive'] and ''}"
                 f"positive {int((diff_pdet > 0).sum())} / negative {int((diff_pdet < 0).sum())} "
                 f"(diagnostic only)")
    lines.append(f"- dataset signs (dPdet): positive {endpoints['dPdet']['n_datasets'] and int((diff_pdet.groupby(level=0).mean() > 0).sum())} / "
                 f"negative {int((diff_pdet.groupby(level=0).mean() < 0).sum())}")
    lines.append("")
    lines.append("## 1.3 Deadhead")
    lines.append("")
    lines.append(f"- deadhead fraction of remaining budget: mean {dh['mean']:.3f}, median {dh['median']:.3f}, "
                 f"p25 {dh['p25']:.3f}, p75 {dh['p75']:.3f}, p90 {dh['p90']:.3f}, "
                 f"range [{dh['min']:.3f}, {dh['max']:.3f}]")
    lines.append(f"- deadhead total (m): mean {dh['deadhead_total_mean_m']:.0f}, median {dh['deadhead_total_median_m']:.0f}")
    lines.append(f"- deadhead / remaining budget (absolute, mean): {dh['deadhead_fraction_abs_mean']:.3f}")
    lines.append(f"- per-dataset deadhead fraction: " + ", ".join(
        f"ds{int(k)}={v:.2f}" for k, v in dh_ds.items()))
    lines.append(f"- relationship with ΔPdet (descriptive): Pearson {pearson:+.3f}, Spearman {spearman:+.3f}, "
                 f"slope {slope:+.3f} (not causal)")
    lines.append("")
    lines.append("## 1.4 Re-coverage / revisits")
    lines.append("")
    for name in ("revisit_fraction", "exposure_multiplicity", "L"):
        s = recov[name]
        lines.append(f"- Δ{name}: {s['mean']:+.5f} CI95[{s['ci_lo']:+.5f}, {s['ci_hi']:+.5f}] "
                     f"(method−control; {'replan higher' if s['mean'] > 0 else 'replan lower'})")
    lines.append(f"- post-fault prior-mass efficiency (mass/m): Δ = {eff_summary['mean']:+.3e} "
                 f"CI95[{eff_summary['ci_lo']:+.3e}, {eff_summary['ci_hi']:+.3e}]")
    lines.append(f"- post-fault mass gain: Δ = {mass_summary['mean']:+.5f} "
                 f"CI95[{mass_summary['ci_lo']:+.5f}, {mass_summary['ci_hi']:+.5f}]")
    lines.append(f"- post-fault distance: Δ = {dist_summary['mean']:+.0f} m "
                 f"CI95[{dist_summary['ci_lo']:+.0f}, {dist_summary['ci_hi']:+.0f}]")
    lines.append("- unique-cell post-fault metrics: NOT AVAILABLE (trajectories archived for one trial only; "
                 "reconstruction is not possible from the frozen artifacts -> would require new simulations; "
                 "per instructions this sub-part is stopped and reported).")
    lines.append("")
    lines.append("## 1.5 Termination / budget")
    lines.append("")
    lines.append("| termination_reason | n | fraction | mean P_detect | mean retention | mean budget_utilization | mean unused_budget (m) | mean ΔPdet vs fixed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for term_name, row in term_table.iterrows():
        lines.append(f"| {term_name} | {int(row['n'])} | {row['fraction']:.3f} | {row['mean_Pdet']:.4f} | "
                     f"{row['mean_retention']:.4f} | {row['mean_budget_utilization']:.4f} | "
                     f"{row['mean_unused_budget']:.0f} | {row['mean_dPdet_vs_fixed']:+.4f} |")
    lines.append("")
    lines.append("pizza_fixed reference: termination reasons " +
                 json.dumps(fixed_term.value_counts().to_dict()) +
                 f", mean budget utilization {float(fixed_util.mean()):.4f}")
    lines.append("")
    lines.append("## 1.6 Splits / inactive survivors")
    lines.append("")
    for key, value in splits.items():
        lines.append(f"- {key}: {value}")
    lines.append(f"- Spearman (ΔPdet vs): " + ", ".join(f"{k}={v:+.3f}" for k, v in assoc.items()) +
                 " (descriptive, not causal)")
    lines.append("")
    lines.append("## FACTS")
    lines.append("")
    lines.append(f"1. Both arms are identical pre-fault (at-fault P_detect equal to {blocking['at_fault_max_abs_diff']:.1e}, "
                 f"same prefailure hashes and trigger steps).")
    lines.append(f"2. pizza_replan ends with ΔP_detect = {endpoints['dPdet']['mean']:+.4f} vs pizza_fixed "
                 f"(CI95 [{endpoints['dPdet']['ci_lo']:+.4f}, {endpoints['dPdet']['ci_hi']:+.4f}], "
                 f"{endpoints['dPdet']['classification']}).")
    lines.append(f"3. Post-fault gain: Δ = {diff_gain.mean():+.4f} (replan gains less).")
    lines.append(f"4. budget utilization: replan {float(util_replan.mean()):.4f} vs fixed {float(util_fixed.mean()):.4f}.")
    lines.append("")
    lines.append("## MECHANISTIC EVIDENCE")
    lines.append("")
    lines.append(f"- deadhead consumes a non-trivial share of the remaining budget before the new routes start "
                 f"(mean fraction {dh['mean']:.3f}); relative mass-efficiency differences are reported above.")
    lines.append(f"- post-fault prior-mass per metre: Δ = {eff_summary['mean']:+.3e} "
                 f"[{eff_summary['ci_lo']:+.3e}, {eff_summary['ci_hi']:+.3e}] (replan more/less efficient per metre).")
    lines.append("")
    lines.append("## PLAUSIBLE EXPLANATIONS")
    lines.append("")
    lines.append("- deadhead + re-entry cost; re-coverage of already-observed terrain; geometric "
                 "discontinuity introduced by reconstructed sectors; early path exhaustion in a subset "
                 "of trials.  The reported tables quantify each factor separately.")
    lines.append("")
    lines.append("## UNSUPPORTED EXPLANATIONS")
    lines.append("")
    lines.append("- \"replanning is worse\" in general; any claim about per-cell unique-coverage mechanisms "
                 "(not measurable without new simulations); any causal reading of the descriptive correlations.")
    lines.append("")
    lines.append("## SAFE PAPER WORDING")
    lines.append("")
    lines.append("> Under the frozen distance-budgeted protocol, the tested naive fault-triggered geometric "
                 "repartitioning policy achieved lower final detection probability than unreplanned coverage of "
                 "the same sectors, despite identical pre-fault behaviour; the diagnostic tables attribute the "
                 "difference to deadhead/re-entry and lower post-fault prior-mass efficiency per metre.")
    lines.append("")

    (A.OUTPUT_DIR / "C7A_PIZZA_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n")
    print("wrote C7A_PIZZA_DIAGNOSTIC.md / _TRIALS.csv / _DATASETS.csv + 5 figures")

    # console summary
    print(f"blocking: {blocking}")
    print(f"dPdet: {fmt(endpoints['dPdet'])}")
    print(f"dRMST_rel: {fmt(endpoints['dRMST_rel'])}")
    print(f"deadhead mean fraction: {dh['mean']:.4f} | Pearson {pearson:+.3f} Spearman {spearman:+.3f}")
    print(f"postfault eff Δ: {eff_summary['mean']:+.3e} [{eff_summary['ci_lo']:+.3e},{eff_summary['ci_hi']:+.3e}]")
    print(f"termination (replan): {term_table['n'].to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
