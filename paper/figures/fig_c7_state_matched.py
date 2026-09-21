#!/usr/bin/env python3
"""Figure 3: state-matched mid-mission UAV-loss recovery (C7B).

Main panel: absolute post-failure detection gain of each recovery branch
(mean with dataset-cluster bootstrap 95% CI), which reads directly as
evidence-guided > failure-triggered geometric > fixed-prior online >> random.
A boxed annotation highlights the confirmatory paired contrast
(evidence-guided - fixed-prior online). Pairing for every contrast is explicit:
differences are computed per trial (dataset, planning_seed), averaged within
dataset, and only then bootstrapped over the 18 datasets using the frozen
helper `scripts/paired_stats.py`.

Source (read-only):
  results/audit_block_C/c7-state-matched_20260910_070544_572098_9ae47248/
      c7_state_matched_runs.csv.gz
      900 rows = 180 prefix rows (`branch_label == 'prefix'`) + 720 branch rows
      (4 branches x 180 trials).
Output: output/fig_c7_recovery.pdf
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_DIR / "scripts"))

from paired_stats import cluster_bootstrap_ci  # noqa: E402  (frozen implementation)

CAMPAIGN = (
    REPO_DIR
    / "results"
    / "audit_block_C"
    / "c7-state-matched_20260910_070544_572098_9ae47248"
)
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_SEED = 20260826
BOOTSTRAP_ITERATIONS = 10_000

EVIDENCE = "dynamic_evidence_3step"
FIXED = "online_static_3step"
GEOMETRIC = "pizza_replan_midflight"
RANDOM = "random_uniform_n8_uncoordinated"

BRANCH_LABELS = {
    EVIDENCE: "evidence-guided",
    FIXED: "fixed-prior online",
    GEOMETRIC: "failure-triggered",
    RANDOM: "random",
}

CONTRASTS = [
    (EVIDENCE, FIXED, "evidence-guided $-$ fixed-prior online"),
    (EVIDENCE, GEOMETRIC, "evidence-guided $-$ failure-triggered repartitioning"),
    (EVIDENCE, RANDOM, "evidence-guided $-$ random"),
    (FIXED, GEOMETRIC, "fixed-prior online $-$ failure-triggered repartitioning"),
]

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=0.85)
plt.rcParams.update({
    "font.family": "serif",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.5,
})

COLOR_BAR = "#4c78a8"
COLOR_NULL = "#9d9d9d"


def load_branches():
    runs = pd.read_csv(CAMPAIGN / "c7_state_matched_runs.csv.gz")
    for column in (
        "PostFault_PdetGain",
        "ConditionalPostFaultPdet",
        "P_detect",
        "RMST",
        "H",
        "Pdet_at_fault",
        "revisit_fraction",
        "exposure_multiplicity",
        "actual_distance_total",
    ):
        runs[column] = pd.to_numeric(runs[column], errors="coerce")
    prefix = runs[runs["branch_label"] == "prefix"]
    branches = runs[runs["branch_label"] == "branch"].copy()
    return prefix, branches


def paired_differences(branches, method, control, column):
    """Per-trial paired differences (method - control), keyed by dataset+seed."""
    wide = branches.pivot_table(
        index=["dataset", "planning_seed"],
        columns="planning_mode",
        values=column,
        aggfunc="mean",
    )
    diff = (wide[method] - wide[control]).dropna()
    return diff


def contrast_summary(branches, method, control, column):
    diff = paired_differences(branches, method, control, column)
    frame = diff.rename("value").reset_index()
    return cluster_bootstrap_ci(
        frame,
        "value",
        "dataset",
        seed=BOOTSTRAP_SEED,
        iterations=BOOTSTRAP_ITERATIONS,
    )


def rmst_rel_differences(branches, method, control):
    """dRMST_rel = (RMST_control - RMST_method) / H_common, positive = earlier."""
    wide_rmst = branches.pivot_table(
        index=["dataset", "planning_seed"], columns="planning_mode",
        values="RMST", aggfunc="mean",
    )
    wide_h = branches.pivot_table(
        index=["dataset", "planning_seed"], columns="planning_mode",
        values="H", aggfunc="mean",
    )
    h_common = np.maximum(wide_h[method], wide_h[control])
    diff = ((wide_rmst[control] - wide_rmst[method]) / h_common).dropna()
    frame = diff.rename("value").reset_index()
    return cluster_bootstrap_ci(
        frame, "value", "dataset",
        seed=BOOTSTRAP_SEED, iterations=BOOTSTRAP_ITERATIONS,
    )


def branch_mean(branches, mode, column="PostFault_PdetGain"):
    """Branch mean with a dataset-cluster bootstrap CI."""
    frame = branches.loc[branches["planning_mode"] == mode, ["dataset", column]].copy()
    frame = frame.rename(columns={column: "value"})
    return cluster_bootstrap_ci(
        frame, "value", "dataset",
        seed=BOOTSTRAP_SEED, iterations=BOOTSTRAP_ITERATIONS,
    )


def main():
    prefix, branches = load_branches()
    n_prefix = len(prefix)
    n_branches = len(branches)
    n_trials = branches[["dataset", "planning_seed"]].drop_duplicates().shape[0]
    print(f"rows: prefix={n_prefix} branch={n_branches} unique trials={n_trials}")
    assert n_prefix == 180 and n_branches == 720 and n_trials == 180

    spread = (
        branches.groupby(["dataset", "planning_seed"])["Pdet_at_fault"]
        .agg(lambda values: values.max() - values.min())
        .max()
    )
    print(f"max within-trial Pdet_at_fault spread across branches: {spread:.3e}")
    assert spread < 1e-12, "checkpoint is not state-matched"

    print("\nper-branch means (branch rows only):")
    per_branch = branches.groupby("planning_mode")[
        ["PostFault_PdetGain", "ConditionalPostFaultPdet", "P_detect", "RMST",
         "revisit_fraction", "exposure_multiplicity"]
    ].mean()
    for mode, row in per_branch.iterrows():
        print(
            f"  {BRANCH_LABELS[mode]:26s} gain={row['PostFault_PdetGain']:.4f} "
            f"cond={row['ConditionalPostFaultPdet']:.4f} "
            f"Pdet={row['P_detect']:.4f} RMST={row['RMST']:.1f}"
        )

    print("\npaired contrasts (dataset-cluster bootstrap, "
          f"seed={BOOTSTRAP_SEED}, {BOOTSTRAP_ITERATIONS} iterations):")
    summaries = []
    for method, control, label in CONTRASTS:
        gain = contrast_summary(branches, method, control, "PostFault_PdetGain")
        cond = contrast_summary(branches, method, control, "ConditionalPostFaultPdet")
        rmst = rmst_rel_differences(branches, method, control)
        summaries.append((label, gain, cond, rmst))
        print(
            f"  {label:52s} gain={gain['mean']:+.4f} "
            f"[{gain['ci_lo']:+.4f}, {gain['ci_hi']:+.4f}] "
            f"cond={cond['mean']:+.4f} "
            f"dRMST_rel={rmst['mean']:+.4f} [{rmst['ci_lo']:+.4f}, {rmst['ci_hi']:+.4f}]"
        )

    print("\nbranch gains with dataset-cluster bootstrap CI:")
    branch_rows = []
    for mode in (EVIDENCE, GEOMETRIC, FIXED, RANDOM):
        stats = branch_mean(branches, mode)
        branch_rows.append((mode, stats))
        print(
            f"  {BRANCH_LABELS[mode]:26s} gain={stats['mean']:.4f} "
            f"[{stats['ci_lo']:.4f}, {stats['ci_hi']:.4f}]"
        )

    primary = next(s for s in summaries if s[0] == "evidence-guided $-$ fixed-prior online")

    fig = plt.figure(figsize=(3.5, 1.95))
    ax = fig.add_axes([0.30, 0.20, 0.68, 0.62])

    branch_rows.sort(key=lambda item: item[1]["mean"])
    y = np.arange(len(branch_rows), dtype=float)
    for yi, (mode, stats) in zip(y, branch_rows):
        ax.barh(
            yi, stats["mean"], height=0.52,
            color=COLOR_BAR if mode != RANDOM else COLOR_NULL,
            edgecolor="none", zorder=2,
        )
        ax.errorbar(
            stats["mean"], yi,
            xerr=[[stats["mean"] - stats["ci_lo"]], [stats["ci_hi"] - stats["mean"]]],
            fmt="none", ecolor="#333333", elinewidth=0.9, capsize=1.6, zorder=3,
        )
        ax.text(
            stats["ci_hi"] + 0.005, yi, f"{stats['mean']:.3f}",
            va="center", ha="left", fontsize=6.0, family="serif", zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [BRANCH_LABELS[mode] for mode, _ in branch_rows],
        fontsize=6.2, family="serif",
    )
    ax.set_xlim(0, max(stats["ci_hi"] for _, stats in branch_rows) * 1.22)
    ax.set_xlabel(r"post-failure detection gain, $\Delta P_{\mathrm{detect}}$")
    ax.set_title("Absolute branch gains (95% cluster CI)", loc="left", pad=2)
    ax.grid(True, alpha=0.25, axis="x")
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    outpath = OUTPUT_DIR / "fig_c7_recovery.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
