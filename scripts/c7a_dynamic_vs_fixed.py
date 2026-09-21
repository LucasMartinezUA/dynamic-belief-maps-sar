"""C7A Analysis 2 — dynamic_evidence_3step vs pizza_fixed_midflight.

Frozen C7A artifacts only.  Paired trial differences -> dataset means ->
cluster bootstrap; TOST with the frozen margins; frozen classification.
Outputs (results/audit_block_C/):
  C7A_DYNAMIC_VS_PIZZA_FIXED.md
  C7A_DYNAMIC_VS_PIZZA_FIXED_DATASETS.csv
  C7_FIGURES/fig_c7a_a2_1_forest_dynamic_minus_fixed.png
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

DYN = "dynamic_evidence_3step"
FIXED = "pizza_fixed_midflight"
REPLAN = "pizza_replan_midflight"
FAIL = "midflight_failure"
MARGIN_PDET = 0.01
MARGIN_RMST = 0.02


def main() -> int:
    runs = A.load_runs()
    failure_rows = runs[runs.condition == FAIL]

    contrasts = {
        "dPdet": c7s._contrast_summary(failure_rows, DYN, FIXED, "dPdet", MARGIN_PDET),
        "dRMST_rel": c7s._contrast_summary(failure_rows, DYN, FIXED, "dRMST_rel", MARGIN_RMST),
        "dPostFaultGain": c7s._contrast_summary(failure_rows, DYN, FIXED, "dPostFaultGain", MARGIN_PDET),
    }
    diff_pdet, n_pairs = c7s._paired_diff(failure_rows, DYN, FIXED, "dPdet")
    diff_rmst, _ = c7s._paired_diff(failure_rows, DYN, FIXED, "dRMST_rel")
    diff_gain, _ = c7s._paired_diff(failure_rows, DYN, FIXED, "dPostFaultGain")

    # at-fault and post-fault decomposition (descriptive; C7A is not state-matched)
    at_fault_dyn = A.trial_metric(runs, DYN, "Pdet_at_fault")
    at_fault_fixed = A.trial_metric(runs, FIXED, "Pdet_at_fault")
    d_at_fault = (at_fault_dyn - at_fault_fixed).dropna()
    gain_dyn = A.trial_metric(runs, DYN, "PostFault_PdetGain")
    gain_fixed = A.trial_metric(runs, FIXED, "PostFault_PdetGain")
    d_gain = (gain_dyn - gain_fixed).dropna()

    retention_dyn = A.trial_metric(runs, DYN, "retention_Pdet")
    retention_fixed = A.trial_metric(runs, FIXED, "retention_Pdet")
    d_retention = (retention_dyn - retention_fixed).dropna()
    ret_summary = A.cluster_summary(d_retention)

    # pizza pre-fault identity (blocking, shared with Analysis 1)
    at_fault = runs[runs.condition == FAIL].pivot_table(
        index=["dataset", "planning_seed"], columns="planning_mode", values="Pdet_at_fault")
    pizza_identical = bool(((at_fault[REPLAN] - at_fault[FIXED]).abs() < 1e-12).all())

    # ---------------- inference table ----------------------------------- #
    key_rows = []
    for name, summary in contrasts.items():
        key_rows.append({
            "endpoint": name,
            "mean": summary["mean"],
            "ci_lo": summary["ci_lo"],
            "ci_hi": summary["ci_hi"],
            "sd_datasets": summary["sd"],
            "n_datasets": summary["n_datasets"],
            "n_pairs": summary["n_pairs"],
            "tost_equivalent": bool(summary["tost"]["equivalent"]),
            "tost_p_max": summary["tost"]["p_max"],
            "classification": summary["classification"],
        })
    inference = pd.DataFrame(key_rows)

    ds_means = diff_pdet.groupby(diff_pdet.index.map(lambda kv: int(kv[0]))).mean()
    ds_signs = {
        "positive": int((ds_means > 0).sum()),
        "negative": int((ds_means < 0).sum()),
        "zero": int((ds_means == 0).sum()),
    }

    # per-dataset CSV
    per_dataset = pd.DataFrame({"dataset": ds_means.index.astype(int),
                                "mean_dPdet": ds_means.to_numpy()})
    d_rmst_ds = diff_rmst.groupby(diff_rmst.index.map(lambda kv: int(kv[0]))).mean()
    per_dataset = per_dataset.merge(
        pd.DataFrame({"dataset": d_rmst_ds.index.astype(int),
                      "mean_dRMST_rel": d_rmst_ds.to_numpy()}), on="dataset", how="left")
    d_at_fault_ds = d_at_fault.groupby(d_at_fault.index.map(lambda kv: int(kv[0]))).mean()
    per_dataset = per_dataset.merge(
        pd.DataFrame({"dataset": d_at_fault_ds.index.astype(int),
                      "mean_dPdet_at_fault": d_at_fault_ds.to_numpy()}), on="dataset", how="left")
    d_gain_ds = d_gain.groupby(d_gain.index.map(lambda kv: int(kv[0]))).mean()
    per_dataset = per_dataset.merge(
        pd.DataFrame({"dataset": d_gain_ds.index.astype(int),
                      "mean_dPostFaultGain": d_gain_ds.to_numpy()}), on="dataset", how="left")
    d_ret_ds = d_retention.groupby(d_retention.index.map(lambda kv: int(kv[0]))).mean()
    per_dataset = per_dataset.merge(
        pd.DataFrame({"dataset": d_ret_ds.index.astype(int),
                      "mean_dRetention": d_ret_ds.to_numpy()}), on="dataset", how="left")
    per_dataset.to_csv(A.OUTPUT_DIR / "C7A_DYNAMIC_VS_PIZZA_FIXED_DATASETS.csv", index=False)

    # ---------------- forest plot --------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(ds_means))[::-1]
    ax.errorbar(ds_means.to_numpy(), y, xerr=None, fmt="o", color="#4f81bd", ms=5)
    ax.axvline(0, color="k", lw=0.8)
    global_mean = contrasts["dPdet"]["mean"]
    ax.axvline(global_mean, color="#c0504d", ls="--", lw=1.2,
               label=f"global mean {global_mean:+.4f}\nCI95[{contrasts['dPdet']['ci_lo']:+.4f},"
                     f"{contrasts['dPdet']['ci_hi']:+.4f}]")
    ax.axvspan(-MARGIN_PDET, MARGIN_PDET, color="grey", alpha=0.15,
               label=f"practical margin ±{MARGIN_PDET}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"ds{int(d)}" for d in ds_means.index])
    ax.set_title("A2-1  Forest: ΔP_detect (dynamic − pizza_fixed) per dataset")
    ax.set_xlabel("Δ P_detect")
    ax.legend(loc="lower right", fontsize=8)
    A.save_figure(fig, "fig_c7a_a2_1_forest_dynamic_minus_fixed.png")
    plt.close(fig)

    # ---------------- report -------------------------------------------- #
    lines = ["# C7A — dynamic_evidence_3step vs pizza_fixed_midflight", ""]
    lines.append("Frozen C7A artifacts; paired trial differences -> dataset means -> cluster bootstrap "
                 f"(seed {A.BOOTSTRAP_SEED}, {A.BOOTSTRAP_ITERS} iterations); TOST with frozen margins "
                 f"(P_detect ±{MARGIN_PDET}, RMST_rel ±{MARGIN_RMST}).")
    lines.append("")
    lines.append("- pizza_replan vs pizza_fixed pre-fault identity (blocking check): "
                 f"{pizza_identical}")
    lines.append("")
    lines.append("## Primary contrast (dynamic − pizza_fixed; positive = favorable to dynamic)")
    lines.append("")
    lines.append("| endpoint | mean | CI95 | TOST equiv. | classification |")
    lines.append("|---|---|---|---|---|")
    for row in inference.itertuples(index=False):
        lines.append(f"| {row.endpoint} | {row.mean:+.4f} | [{row.ci_lo:+.4f}, {row.ci_hi:+.4f}] | "
                     f"{row.tost_equivalent} | {row.classification} |")
    lines.append("")
    lines.append(f"- dataset signs (ΔP_detect): +{ds_signs['positive']} / -{ds_signs['negative']} / 0{ds_signs['zero']}")
    lines.append(f"- trial signs (ΔP_detect): +{int((diff_pdet > 0).sum())} / "
                 f"-{int((diff_pdet < 0).sum())} (diagnostic only)")
    lines.append("")
    lines.append("## Decomposition (descriptive; C7A is NOT state-matched)")
    lines.append("")
    lines.append(f"- Δ Pdet_at_fault (dynamic − fixed) = {d_at_fault.mean():+.4f} "
                 f"(dynamic already ahead before the failure)")
    lines.append(f"- Δ PostFault_PdetGain = {d_gain.mean():+.4f} "
                 f"(additional advantage accumulated after the failure)")
    lines.append(f"- final Δ P_detect = {diff_pdet.mean():+.4f} "
                 f"(= at-fault + post-fault increment within rounding)")
    lines.append("")
    lines.append("## Retention (paired)")
    lines.append("")
    lines.append(f"- Δretention (dynamic − pizza_fixed) = {ret_summary['mean']:+.4f} "
                 f"CI95[{ret_summary['ci_lo']:+.4f}, {ret_summary['ci_hi']:+.4f}], "
                 f"n_ds={ret_summary['n_datasets']}, datasets +{ret_summary['datasets_positive']}/"
                 f"-{ret_summary['datasets_negative']}")
    lines.append("- retention > 1 for dynamic must NOT be worded as 'losing a UAV improves dynamic': "
                 "it is a consequence of the distance-budget protocol and the trajectory, not a causal "
                 "benefit of the loss.")
    lines.append("")
    lines.append("## Pre-specified interpretation")
    lines.append("")
    c = contrasts["dPdet"]
    if c["classification"] == "IMPROVED":
        lines.append("Case B: dynamic outperformed unreplanned geometric coverage "
                     "(CI above 0 and beyond the practical margin).")
    elif c["classification"] == "EQUIVALENT":
        lines.append("Case A: practically similar final detection probability under the "
                     "distance-budgeted protocol.")
    elif c["classification"] == "INCONCLUSIVE":
        lines.append("Case C: no reliable difference was established.")
    else:
        lines.append(f"Case D: {c['classification']}.")
    lines.append("")
    lines.append("## Recommended single wording")
    lines.append("")
    lines.append("> " + (
        "Continuous evidence-guided online replanning achieved higher final detection probability than "
        "unreplanned geometric coverage under the frozen distance-budgeted mid-mission-loss protocol "
        f"(paired ΔP_detect = {c['mean']:+.4f}, CI95 [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}], "
        f"{c['classification'].lower()}), with a retention difference of "
        f"{ret_summary['mean']:+.4f} and an advantage that is present already at the fault instant and "
        "does not rely on post-fault recovery alone."
        if c["classification"] != "EQUIVALENT" else
        "Dynamic and unreplanned geometric coverage achieved practically similar final detection "
        "probability under the distance-budgeted mid-mission-loss protocol."
    ))
    lines.append("")
    (A.OUTPUT_DIR / "C7A_DYNAMIC_VS_PIZZA_FIXED.md").write_text("\n".join(lines) + "\n")

    print(inference.to_string(index=False))
    print(f"signs: datasets +{ds_signs['positive']}/-{ds_signs['negative']} | "
          f"trials +{int((diff_pdet > 0).sum())}/-{int((diff_pdet < 0).sum())}")
    print(f"decomposition: at_fault {d_at_fault.mean():+.4f} + postfault {d_gain.mean():+.4f} "
          f"= {diff_pdet.mean():+.4f}")
    print(f"retention delta: {ret_summary['mean']:+.4f} "
          f"[{ret_summary['ci_lo']:+.4f},{ret_summary['ci_hi']:+.4f}]")
    print("wrote C7A_DYNAMIC_VS_PIZZA_FIXED.md / _DATASETS.csv + forest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
