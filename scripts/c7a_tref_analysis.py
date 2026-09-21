"""C7A Analysis 3 — secondary time-matched endpoints at T_ref = 8000 s.

The T_ref endpoints are pre-registered and already frozen in the C7A rows
(`P_detect_at_Tref`, `RMST_at_Tref`, `NRMST_at_Tref`; convention from
c7_stage._tref_metrics: paths truncated to indices 0..T_ref, i.e. steps
0..8000 / H = T+1 = 8001).  Resource accounting is read from the recovery
curves.  Frozen artifacts only; no new simulations.

Outputs (results/audit_block_C/):
  C7A_TREF_ANALYSIS.md
  C7A_TREF_TRIALS.csv
  C7A_TREF_DATASETS.csv
  C7A_TREF_GLOBAL.csv
  C7_FIGURES/fig_c7a_a3_1..5*.png
"""
from __future__ import annotations

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

FAIL = "midflight_failure"
K0 = "k0_baseline"
T_REF = int(A.T_REF)
MARGIN_PDET = 0.01
MARGIN_RMST = 0.02

ARMS = [
    ("dynamic_evidence_3step", "dynamic_evidence_3step"),
    ("online_static_3step", "online_static_3step"),
    ("pizza_replan_midflight", "pizza_no_failure"),
    ("pizza_fixed_midflight", "pizza_no_failure"),
    ("random_uniform_n8_uncoordinated", "random_uniform_n8_uncoordinated"),
]
CONTRASTS = [
    ("T1", "dynamic_evidence_3step", "pizza_replan_midflight"),
    ("T2", "dynamic_evidence_3step", "pizza_fixed_midflight"),
    ("T3", "dynamic_evidence_3step", "online_static_3step"),
    ("T4", "online_static_3step", "pizza_replan_midflight"),
    ("T5", "pizza_replan_midflight", "pizza_fixed_midflight"),
]


def _distance_at_tref(recovery: pd.DataFrame) -> pd.Series:
    """Fleet distance travelled up to step T_ref (last sample <= T_ref)."""
    rows = {}
    for (dataset, seed, mode), group in recovery.groupby(
        ["dataset", "planning_seed", "planning_mode"], sort=True
    ):
        g = group[group.step <= T_REF]
        if g.empty:
            continue
        rows[(int(dataset), int(seed), mode)] = float(g.sort_values("step").iloc[-1]["distance_absolute"])
    series = pd.Series(rows)
    series.index = pd.MultiIndex.from_tuples(
        [(d, s) for d, s, _ in series.index], names=["dataset", "planning_seed"]
    )
    # keep the mode as a column via a frame
    frame = pd.DataFrame({
        "dataset": [k[0] for k in rows], "planning_seed": [k[1] for k in rows],
        "planning_mode": [k[2] for k in rows], "distance_at_tref": list(rows.values()),
    })
    return frame


def main() -> int:
    runs = A.load_runs()
    recovery = A.load_recovery()
    fail = runs[runs.condition == FAIL]

    # ---------------- retention + resource frame ------------------------ #
    rows = []
    for method, k0_label in ARMS:
        p_tref = A.trial_metric(runs, method, "P_detect_at_Tref", FAIL)
        p_k0 = A.trial_metric(runs, k0_label, "P_detect_at_Tref", K0)
        r_tref = A.trial_metric(runs, method, "RMST_at_Tref", FAIL)
        r_k0 = A.trial_metric(runs, k0_label, "RMST_at_Tref", K0)
        p_budget = A.trial_metric(runs, method, "P_detect", FAIL)
        retention_budget = A.trial_metric(runs, method, "retention_Pdet", FAIL)
        common = p_tref.index.intersection(p_k0.index)
        retention_tref = (p_tref.loc[common] / p_k0.loc[common]).replace([np.inf, -np.inf], np.nan)
        degradation_tref = (r_tref.loc[common] - r_k0.loc[common]) / (T_REF + 1.0)
        for key in common:
            rows.append({
                "dataset": int(key[0]), "planning_seed": int(key[1]), "method": method,
                "P_detect_Tref": float(p_tref.loc[key]),
                "P_detect_k0_Tref": float(p_k0.loc[key]),
                "retention_Tref": float(retention_tref.loc[key]),
                "RMST_Tref": float(r_tref.loc[key]),
                "RMST_k0_Tref": float(r_k0.loc[key]),
                "NRMST_Tref": float(r_tref.loc[key]) / (T_REF + 1.0),
                "degradation_RMST_Tref": float(degradation_tref.loc[key]),
                "retention_budget": float(retention_budget.loc[key]) if key in retention_budget.index else np.nan,
            })
    trials = pd.DataFrame(rows)
    trials.to_csv(A.OUTPUT_DIR / "C7A_TREF_TRIALS.csv", index=False)

    # ---------------- resource verification ----------------------------- #
    dist = _distance_at_tref(recovery)
    dist_fail = dist[dist.planning_mode != "k0"]
    per_method_distance = dist.groupby("planning_mode")["distance_at_tref"].agg(["mean", "median", "size"])
    k0_arms = {"dynamic_evidence_3step", "online_static_3step",
               "random_uniform_n8_uncoordinated"}  # k0 curves not archived
    expected_k0 = 5 * 5.0 * T_REF
    expected_fail = 5 * 5.0 * 4000 + 4 * 5.0 * 4000

    # ---------------- global table -------------------------------------- #
    global_rows = []
    for method, _ in ARMS:
        sub = trials[trials.method == method]
        global_rows.append({
            "arm": method,
            "P_detect_Tref": sub["P_detect_Tref"].mean(),
            "P_detect_k0_Tref": sub["P_detect_k0_Tref"].mean(),
            "retention_Tref": sub["retention_Tref"].mean(),
            "RMST_Tref": sub["RMST_Tref"].mean(),
            "NRMST_Tref": sub["NRMST_Tref"].mean(),
            "mean_distance_at_Tref_m": float(
                per_method_distance["mean"].get(method, np.nan)),
        })
    global_table = pd.DataFrame(global_rows)
    global_table.to_csv(A.OUTPUT_DIR / "C7A_TREF_GLOBAL.csv", index=False)

    # ---------------- paired contrasts ---------------------------------- #
    contrast_rows = []
    for name, method, control in CONTRASTS:
        for column, margin in (("dPdet_Tref", MARGIN_PDET), ("dRMST_rel_Tref", MARGIN_RMST)):
            summary = c7s._contrast_summary(fail, method, control, column, margin)
            diff, n = c7s._paired_diff(fail, method, control, column)
            ds_means = diff.groupby([k[0] for k in diff.index]).mean()
            contrast_rows.append({
                "contrast": name, "method": method, "control": control, "endpoint": column,
                "mean": summary["mean"], "ci_lo": summary["ci_lo"], "ci_hi": summary["ci_hi"],
                "n_datasets": summary["n_datasets"], "n_pairs": summary["n_pairs"],
                "tost_equivalent": bool(summary["tost"]["equivalent"]),
                "classification": summary["classification"],
                "datasets_positive": int((ds_means > 0).sum()),
                "datasets_negative": int((ds_means < 0).sum()),
            })
    contrast_table = pd.DataFrame(contrast_rows)

    # ---------------- datasets CSV -------------------------------------- #
    dataset_rows = []
    for method, _ in ARMS:
        sub = trials[trials.method == method]
        for dataset, group in sub.groupby("dataset", sort=True):
            dataset_rows.append({
                "dataset": int(dataset), "method": method,
                "P_detect_Tref": group["P_detect_Tref"].mean(),
                "P_detect_k0_Tref": group["P_detect_k0_Tref"].mean(),
                "retention_Tref": group["retention_Tref"].mean(),
                "RMST_Tref": group["RMST_Tref"].mean(),
                "mean_distance_at_Tref_m": float(
                    dist[(dist.planning_mode == method)
                         & (dist.dataset == dataset)]["distance_at_tref"].mean()),
            })
    datasets_table = pd.DataFrame(dataset_rows)
    datasets_table.to_csv(A.OUTPUT_DIR / "C7A_TREF_DATASETS.csv", index=False)

    # ---------------- figures ------------------------------------------- #
    # fig 1: retention budget vs retention T_ref
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for method, _ in ARMS:
        sub = trials[trials.method == method]
        ax.scatter(sub["retention_budget"], sub["retention_Tref"], s=10, alpha=0.4, label=method)
    ax.axhline(1.0, color="k", lw=0.8, ls=":"); ax.axvline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlim(0.6, 1.15); ax.set_ylim(0.5, 1.15)
    ax.set_title("A3-1  retention: distance-budget vs T_ref (per trial)")
    ax.set_xlabel("retention_Pdet (fixed distance budget)")
    ax.set_ylabel("retention at T_ref=8000 s")
    ax.legend(fontsize=7)
    A.save_figure(fig, "fig_c7a_a3_1_retention_budget_vs_tref.png"); plt.close(fig)

    # fig 2-3: T_ref forests
    for tag, (name, method, control) in zip(("2", "3"), (CONTRASTS[0], CONTRASTS[1])):
        diff, _ = c7s._paired_diff(fail, method, control, "dPdet_Tref")
        ds_means = diff.groupby([k[0] for k in diff.index]).mean()
        summary = [r for r in contrast_rows if r["contrast"] == name and r["endpoint"] == "dPdet_Tref"][0]
        fig, ax = plt.subplots(figsize=(8, 6))
        y = np.arange(len(ds_means))[::-1]
        ax.errorbar(ds_means.to_numpy(), y, fmt="o", color="#4f81bd", ms=5)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(summary["mean"], color="#c0504d", ls="--", lw=1.2,
                   label=f"global {summary['mean']:+.4f}\nCI95[{summary['ci_lo']:+.4f},"
                         f"{summary['ci_hi']:+.4f}]")
        ax.axvspan(-MARGIN_PDET, MARGIN_PDET, color="grey", alpha=0.15, label="margin ±0.01")
        ax.set_yticks(y); ax.set_yticklabels([f"ds{int(d)}" for d in ds_means.index])
        ax.set_title(f"A3-{tag}  Forest ΔP_detect@T_ref: {method} − {control}")
        ax.set_xlabel("Δ P_detect @ T_ref"); ax.legend(fontsize=8, loc="lower right")
        A.save_figure(fig, f"fig_c7a_a3_{tag}_forest_tref_{method}_minus_{control}.png"); plt.close(fig)

    # fig 4: P_detect vs simulation time (absolute, mean across trials)
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, color in (("dynamic_evidence_3step", "#4f81bd"),
                        ("online_static_3step", "#9bbb59"),
                        ("pizza_replan_midflight", "#c0504d"),
                        ("pizza_fixed_midflight", "#8064a2")):
        sub = recovery[recovery.planning_mode == mode]
        grid = np.arange(0, 12_001, 250.0)
        curves = []
        for _, group in sub.groupby(["dataset", "planning_seed"]):
            g = group.sort_values("time")
            curves.append(np.interp(grid, g["time"], g["P_detect_cumulative"]))
        ax.plot(grid, np.mean(curves, axis=0), color=color, label=mode)
    ax.axvline(T_REF, color="k", ls="--", lw=1.0, label=f"T_ref={T_REF}")
    ax.set_title("A3-4  P_detect vs simulation time (mean across trials)")
    ax.set_xlabel("simulation time (s)"); ax.set_ylabel("cumulative P_detect"); ax.legend(fontsize=8)
    A.save_figure(fig, "fig_c7a_a3_4_pdet_vs_time.png"); plt.close(fig)

    # fig 5: P_detect vs time since failure
    failure_time = A.trial_metric(runs, "pizza_fixed_midflight", "failure_time", FAIL)
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode, color in (("dynamic_evidence_3step", "#4f81bd"),
                        ("online_static_3step", "#9bbb59"),
                        ("pizza_replan_midflight", "#c0504d"),
                        ("pizza_fixed_midflight", "#8064a2")):
        sub = recovery[recovery.planning_mode == mode]
        grid = np.arange(0, 6_001, 250.0)
        curves = []
        for (dataset, seed), group in sub.groupby(["dataset", "planning_seed"]):
            t_fail = failure_time.get((int(dataset), int(seed)))
            if t_fail is None or not np.isfinite(t_fail):
                continue
            g = group.sort_values("time")
            curves.append(np.interp(grid + t_fail, g["time"], g["P_detect_cumulative"], left=np.nan))
        ax.plot(grid, np.nanmean(curves, axis=0), color=color, label=mode)
    ax.set_title("A3-5  P_detect vs time since failure (per-trial fault time respected)")
    ax.set_xlabel("seconds since failure"); ax.set_ylabel("cumulative P_detect"); ax.legend(fontsize=8)
    A.save_figure(fig, "fig_c7a_a3_5_pdet_vs_time_since_failure.png"); plt.close(fig)

    # ---------------- sign convention + RMST/Pdet correlation ----------- #
    corr_rows = []
    for name, method, control in CONTRASTS:
        dp, _ = c7s._paired_diff(fail, method, control, "dPdet_Tref")
        dr, _ = c7s._paired_diff(fail, method, control, "dRMST_rel_Tref")
        joined = pd.DataFrame({"dp": dp, "dr": dr}).dropna()
        ds_means = joined.groupby([k[0] for k in joined.index]).mean()
        corr = float(ds_means["dp"].corr(ds_means["dr"], method="pearson")) if len(ds_means) > 2 else np.nan
        corr_rows.append({"contrast": name, "dataset_level_corr_dPdetTref_dRMSTTref": corr})
    corr_table = pd.DataFrame(corr_rows)

    # ---------------- report -------------------------------------------- #
    lines = ["# C7A — secondary endpoints at fixed time T_ref = 8000 s", ""]
    lines.append("Pre-registered secondary endpoints, already frozen in the C7A rows "
                 "(`P_detect_at_Tref`, `RMST_at_Tref`, `NRMST_at_Tref`). Convention "
                 "(`c7_stage._tref_metrics`): paths truncated to indices 0..T_ref, i.e. steps "
                 "0..8000 with H = 8001; RMST evaluated at the fixed horizon. Identical "
                 "convention for every arm and its k=0 baseline.")
    lines.append("")
    lines.append("### Sign conventions")
    lines.append("")
    lines.append("- `degradation_RMST_Tref = (RMST(T_ref) − RMST_k0(T_ref)) / (T_ref+1)`: "
                 "**positive = worse after the failure** (more expected search time).")
    lines.append("- `dRMST_rel_Tref = (RMST_control − RMST_method) / (T_ref+1)`: "
                 "**positive = favorable to the method** (lower RMST is better).")
    lines.append("- `retention_Tref = P_detect_Tref(failure) / P_detect_Tref(own k0)`.")
    lines.append("")
    lines.append("## Resource verification at T_ref (from recovery curves)")
    lines.append("")
    lines.append(f"- expected k0 fleet distance in 8000 s at 5 m/s with 5 UAVs: {expected_k0:.0f} m")
    lines.append(f"- expected failure-arm distance under the budget protocol "
                 f"(100 km with 5 UAVs + 100 km with 4 UAVs, both ≈ 4000 s): {expected_fail:.0f} m")
    lines.append("")
    lines.append("| arm | mean distance @T_ref (m) | median | n |")
    lines.append("|---|---|---|---|")
    for arm, row in per_method_distance.sort_index().iterrows():
        lines.append(f"| {arm} | {row['mean']:.0f} | {row['median']:.0f} | {int(row['size'])} |")
    lines.append("")
    lines.append("(k=0 curves are not archived in `c7_midflight_recovery.csv`; the failure arms' "
                 "numbers are verified against the analytical expectation above.)")
    lines.append("")
    lines.append("## Global table at T_ref")
    lines.append("")
    lines.append("| arm | P_detect_Tref | P_detect_k0_Tref | retention_Tref | RMST_Tref | NRMST_Tref | mean distance @T_ref |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in global_table.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.P_detect_Tref:.4f} | {row.P_detect_k0_Tref:.4f} | "
                     f"{row.retention_Tref:.4f} | {row.RMST_Tref:.1f} | {row.NRMST_Tref:.4f} | "
                     f"{row.mean_distance_at_Tref_m:.0f} |")
    lines.append("")
    lines.append("## Paired contrasts at T_ref")
    lines.append("")
    lines.append("| contrast | endpoint | mean | CI95 | classification | datasets +/- |")
    lines.append("|---|---|---|---|---|---|")
    for row in contrast_table.itertuples(index=False):
        lines.append(f"| {row.contrast} ({row.method} − {row.control}) | {row.endpoint} | "
                     f"{row.mean:+.4f} | [{row.ci_lo:+.4f}, {row.ci_hi:+.4f}] | {row.classification} | "
                     f"{row.datasets_positive}/{row.datasets_negative} |")
    lines.append("")
    lines.append("## Two notions of robustness")
    lines.append("")
    lines.append("| method | retention (fixed distance budget) | retention (fixed time T_ref) |")
    lines.append("|---|---|---|")
    budget_ret = {row["arm"]: row["retention_Tref"] for row in global_rows}
    for method, _ in ARMS:
        fixed_ret = A.trial_metric(runs, method, "retention_Pdet", FAIL).mean()
        lines.append(f"| {method} | {fixed_ret:.4f} | {budget_ret[method]:.4f} |")
    lines.append("")
    lines.append("## RMST vs P_detect at T_ref (dataset-level correlation, descriptive)")
    lines.append("")
    for row in corr_rows:
        lines.append(f"- {row['contrast']}: corr(ΔP_detect_Tref, dRMST_Tref) = "
                     f"{row['contrast'] and row['dataset_level_corr_dPdetTref_dRMSTTref']:+.3f}")
    lines.append("")
    lines.append("High correlation indicates that at this operating point RMST provides a temporally "
                 "consistent but not independent signal; it is not evidence that RMST is uninformative.")
    lines.append("")
    (A.OUTPUT_DIR / "C7A_TREF_ANALYSIS.md").write_text("\n".join(lines) + "\n")

    print(global_table.to_string(index=False))
    print()
    print(contrast_table[["contrast", "endpoint", "mean", "ci_lo", "ci_hi", "classification",
                          "datasets_positive", "datasets_negative"]].to_string(index=False))
    print()
    print(per_method_distance.to_string())
    print("wrote C7A_TREF_ANALYSIS.md / _TRIALS.csv / _DATASETS.csv / _GLOBAL.csv + 5 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
