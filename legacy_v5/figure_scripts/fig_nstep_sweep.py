#!/usr/bin/env python3
"""N-step sweep figure: likelihood + wall_time vs lookahead depth N (dual axis)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "nstep_sweep.csv")
df = pd.read_csv(csv_path)
mode_to_n = {f"dynamic_{n}step": n for n in [2, 3, 4, 5, 6, 7]}
df["N"] = df["planning_mode"].map(mode_to_n)
df = df.dropna(subset=["N"])
df["N"] = df["N"].astype(int)
summary = (
    df.groupby("N")
    .agg(
        mean_lh=("likelihood", "mean"),
        sem_lh=("likelihood", "sem"),
        mean_wall=("wall_time", "mean"),
        sem_wall=("wall_time", "sem"),
    )
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(3.45, 1.9))
Ns = summary["N"].values
color_lh = "#1f77b4"
color_wt = "#d62728"

# Left axis: Likelihood with 95% CI
ax1.errorbar(
    Ns,
    summary["mean_lh"],
    yerr=1.96 * summary["sem_lh"],
    fmt="o-",
    color=color_lh,
    capsize=3,
    linewidth=1.2,
    markersize=4,
    label=r"Likelihood $L$ (left)",
    zorder=5,
)
ax1.set_xlabel("Lookahead Depth $N$")
ax1.set_ylabel(r"Likelihood $L$", color=color_lh)
ax1.tick_params(axis="y", labelcolor=color_lh)
ax1.set_xticks([2, 3, 4, 5, 6, 7])
ax1.set_xlim(1.5, 7.6)
# headroom on the left axis so nothing crowds the top
lh_lo = (summary["mean_lh"] - 1.96 * summary["sem_lh"]).min()
lh_hi = (summary["mean_lh"] + 1.96 * summary["sem_lh"]).max()
ax1.set_ylim(lh_lo - 0.004, lh_hi + 0.010)

# Right axis: Wall time with 95% CI
ax2 = ax1.twinx()
ax2.errorbar(
    Ns,
    summary["mean_wall"],
    yerr=1.96 * summary["sem_wall"],
    fmt="s--",
    color=color_wt,
    capsize=3,
    linewidth=1.2,
    markersize=4,
    label="Wall time (right)",
    zorder=5,
)
ax2.set_ylabel("Wall time (s)", color=color_wt)
ax2.tick_params(axis="y", labelcolor=color_wt)
wt_lo = (summary["mean_wall"] - 1.96 * summary["sem_wall"]).min()
wt_hi = (summary["mean_wall"] + 1.96 * summary["sem_wall"]).max()
ax2.set_ylim(wt_lo - 8, wt_hi + 22)  # headroom on top for the fit box

# Fit box: upper-left area, in axis-fraction coords, clear of both curves
# (both curves are LOW at left N=2..3, so upper-left is empty)
ax2.text(
    0.04,
    0.94,
    r"$\mathrm{time} = 18.6\,N + 76.8$" + "\n" + r"$R^2 = 0.998$",
    transform=ax2.transAxes,
    fontsize=6.5,
    color=color_wt,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color_wt, alpha=0.85, lw=0.5),
    zorder=6,
)

# Combined legend: lower-right, where the likelihood curve has flattened
# high and the time curve has risen high -> the lower-right band is empty
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="lower right",
    framealpha=0.9,
    edgecolor="gray",
    fontsize=6.5,
    handlelength=1.6,
    borderpad=0.4,
    labelspacing=0.3,
).set_zorder(7)

# N=3 annotation: place BELOW its point so it never collides with the
# fit box (upper-left) or legend (lower-right)
n3 = summary[summary["N"] == 3]
if not n3.empty:
    y3 = n3["mean_lh"].iloc[0]
    ax1.annotate(
        r"$N{=}3$",
        xy=(3, y3),
        xytext=(3.0, y3 - 0.012),
        fontsize=7,
        color="dimgray",
        ha="left",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.5),
        zorder=8,
    )

fig.tight_layout(pad=0.3)
out_path = os.path.join(script_dir, "fig_nstep_sweep.pdf")
fig.savefig(out_path)
fig.savefig(out_path.replace(".pdf", ".png"))  # for visual check
print("Saved", out_path)
print("\nSummary:")
print(summary.to_string(index=False))
