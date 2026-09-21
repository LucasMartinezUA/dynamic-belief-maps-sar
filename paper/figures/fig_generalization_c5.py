#!/usr/bin/env python3
"""Figure 2: evidence-guided vs fixed-prior online replanning across 18 scenarios.

Two side-by-side forest panels over the 18 prespecified SAREnv scenarios:
  (a) dPdet per dataset   (b) dRMST_rel per dataset
Per-dataset CIs are the frozen Student-t paired CIs of the campaign; the global
dataset-cluster bootstrap estimate is drawn as a diamond at the foot of each
panel (never as a vertical band, which would read as a per-dataset criterion).

Sources (read-only):
  results/audit_block_C/c5-generalization_20260827_181139_043334_662411fc/
      c5_dataset_effects.csv
      c5_global_summary.csv
Output: output/fig_generalization.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]
CAMPAIGN = (
    REPO_DIR
    / "results"
    / "audit_block_C"
    / "c5-generalization_20260827_181139_043334_662411fc"
)
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=0.85)
plt.rcParams.update({
    "font.family": "serif",
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
})

COLOR = "#1f77b4"
COLOR_INCONCLUSIVE = "#d62728"
COLOR_GLOBAL = "#2ca02c"


def _forest(ax, frame, mean_col, lo_col, hi_col, global_row, title, xlabel):
    """One forest panel: per-dataset rows plus the global estimate at the foot."""
    rows = frame.sort_values(mean_col).reset_index(drop=True)
    y = np.arange(len(rows), dtype=float)
    y_gap = 1.4
    y_global = len(rows) + y_gap

    for i, row in rows.iterrows():
        color = (
            COLOR_INCONCLUSIVE
            if str(row["state"]).upper() == "INCONCLUSIVE"
            else COLOR
        )
        ax.errorbar(
            row[mean_col],
            y[i],
            xerr=[[row[mean_col] - row[lo_col]], [row[hi_col] - row[mean_col]]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=3.2,
            elinewidth=0.9,
            capsize=1.4,
            zorder=3,
        )

    ax.errorbar(
        float(global_row[mean_col]),
        y_global,
        xerr=[
            [float(global_row[mean_col]) - float(global_row[lo_col])],
            [float(global_row[hi_col]) - float(global_row[mean_col])],
        ],
        fmt="D",
        color=COLOR_GLOBAL,
        ecolor=COLOR_GLOBAL,
        markersize=4.6,
        elinewidth=1.2,
        capsize=2.0,
        zorder=4,
    )

    labels = [f"DS{int(ds)}" for ds in rows["dataset"]]
    labels.append("Global")
    ticks = list(y) + [y_global]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, family="serif")
    ax.get_yticklabels()[-1].set_fontweight("bold")

    # Highlight the single inconclusive dataset row in bold.
    for tick, state in zip(ax.get_yticklabels(), rows["state"]):
        if str(state).upper() == "INCONCLUSIVE":
            tick.set_fontweight("bold")
            tick.set_color(COLOR_INCONCLUSIVE)

    ax.axvline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.6, zorder=1)
    ax.axhline(len(rows) + y_gap / 2.0, color="#bbbbbb", linewidth=0.6, zorder=1)
    ax.set_title(title, loc="left", pad=3)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25, axis="x")
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    lo = min(rows[lo_col].min(), float(global_row[lo_col]))
    hi = max(rows[hi_col].max(), float(global_row[hi_col]))
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    ax.set_xlim(min(lo - pad, -pad), hi + pad)
    ax.set_ylim(-0.8, y_global + 0.9)
    return rows


def main():
    effects = pd.read_csv(CAMPAIGN / "c5_dataset_effects.csv")
    global_summary = pd.read_csv(CAMPAIGN / "c5_global_summary.csv").set_index("endpoint")

    assert len(effects) == 18, f"expected 18 datasets, found {len(effects)}"

    def global_row_for(endpoint, mean_col, lo_col, hi_col):
        row = global_summary.loc[endpoint]
        return pd.Series(
            {
                mean_col: float(row["mean"]),
                lo_col: float(row["cluster_ci_lo"]),
                hi_col: float(row["cluster_ci_hi"]),
            }
        )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.95))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.145, wspace=0.30)

    rows_a = _forest(
        ax_a,
        effects,
        "dPdet_mean",
        "dPdet_ci_lo",
        "dPdet_ci_hi",
        global_row_for("dPdet", "dPdet_mean", "dPdet_ci_lo", "dPdet_ci_hi"),
        "(a) Detection probability",
        r"$\Delta P_{detect}$ (evidence-guided $-$ fixed-prior online)",
    )
    rows_b = _forest(
        ax_b,
        effects,
        "dRMST_rel_mean",
        "dRMST_rel_ci_lo",
        "dRMST_rel_ci_hi",
        global_row_for(
            "dRMST_rel", "dRMST_rel_mean", "dRMST_rel_ci_lo", "dRMST_rel_ci_hi"
        ),
        "(b) Time to detection",
        r"$\Delta RMST_{rel}$ (positive = earlier detection)",
    )

    outpath = OUTPUT_DIR / "fig_generalization.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Saved: {outpath}")
    print("C5 globals (cluster bootstrap, 18 datasets):")
    for endpoint, label in (("dPdet", "dPdet"), ("dRMST_rel", "dRMST_rel")):
        row = global_summary.loc[endpoint]
        print(
            f"  {label:10s} mean={row['mean']:.4f} "
            f"cluster CI [{row['cluster_ci_lo']:.4f}, {row['cluster_ci_hi']:.4f}] "
            f"margin={row['margin']:.2f} {row['classification']}"
        )
    print(
        "  per-dataset range: "
        f"min={rows_a['dPdet_mean'].min():.4f} (DS{int(rows_a.loc[rows_a['dPdet_mean'].idxmin(), 'dataset'])}) "
        f"max={rows_a['dPdet_mean'].max():.4f} (DS{int(rows_a.loc[rows_a['dPdet_mean'].idxmax(), 'dataset'])})"
    )
    print(
        "  inconclusive datasets: "
        + ", ".join(
            f"DS{int(ds)}" for ds in rows_a.loc[
                rows_a["state"].str.upper() == "INCONCLUSIVE", "dataset"
            ]
        )
    )
    positive = effects.loc[effects["dPdet_ci_lo"] > 0].sort_values("dPdet_mean")
    if len(positive):
        smallest = positive.iloc[0]
        print(
            "  smallest effect whose dPdet CI excludes zero: "
            f"DS{int(smallest['dataset'])} {smallest['dPdet_mean']:.4f} "
            f"[{smallest['dPdet_ci_lo']:.4f}, {smallest['dPdet_ci_hi']:.4f}]"
        )


if __name__ == "__main__":
    main()
